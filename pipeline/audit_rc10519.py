"""Audit existing RC archives and extracted annotations without modifying images."""
import csv
import hashlib
import json
import math
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile
from research import ROOT

BASE = ROOT / 'data/raw/external_cracks/Reinforced concrete structure crack detection data'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def shape_issues(shape, width, height):
    points = shape.get('points', [])
    if not isinstance(points, list) or not points:
        return ['missing_points']
    if any(not isinstance(p, (list, tuple)) or len(p) != 2 or
           any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in p) for p in points):
        return ['invalid_or_nonfinite_points']
    issues = []
    if shape.get('shape_type') == 'rectangle':
        if len(points) != 2:
            issues.append('rectangle_point_count')
        elif points[0][0] == points[1][0] or points[0][1] == points[1][1]:
            issues.append('degenerate_rectangle')
    if any(not 0 <= x <= width or not 0 <= y <= height for x, y in points):
        issues.append('out_of_image_bounds')
    return issues


def audit_archive(path, extracted, suffix):
    mismatches = []
    with ZipFile(path) as archive:
        members = [m for m in archive.infolist() if not m.is_dir() and m.filename.lower().endswith(suffix)]
        names = [m.filename.replace('\\', '/').split('/')[-1] for m in members]
        name_set = set(names)
        if len(set(names)) != len(names):
            raise ValueError(f'Duplicate archive basenames: {path}')
        for member, name in zip(members, names):
            file = extracted / name
            if not file.exists():
                mismatches.append(dict(name=name, reason='missing_extracted_file'))
                continue
            crc, size = 0, 0
            with file.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    crc = zlib.crc32(block, crc)
                    size += len(block)
            if (size, crc & 0xffffffff) != (member.file_size, member.CRC):
                mismatches.append(dict(name=name, reason='extracted_size_or_crc_mismatch'))
        extras = sorted(p.name for p in extracted.iterdir() if p.is_file() and p.suffix.lower() == suffix and p.name not in name_set)
    return dict(path=path.relative_to(ROOT).as_posix(), size_bytes=path.stat().st_size,
                sha256=sha(path), members=len(members), extracted_mismatches=mismatches,
                extracted_extras=extras,
                interpretation='CRC/size verifies extraction consistency, not publisher identity or cryptographic authenticity.')


def main():
    out = ROOT / 'datasets/source_provenance'
    out.mkdir(exist_ok=True)
    image_dir, json_dir = BASE / 'images', BASE / 'label_json'
    archives = [audit_archive(BASE / 'images.zip', image_dir, '.jpg'),
                audit_archive(BASE / 'label_json.zip', json_dir, '.json')]
    source = ROOT / 'datasets/source_content_v1/index.csv'
    with source.open(encoding='utf-8', newline='') as stream:
        indexed = {r['original_path']: r for r in csv.DictReader(stream)}
    rows, labels, types, issues = [], Counter(), Counter(), Counter()
    images = sorted(image_dir.glob('*.jpg'))
    for image in images:
        annotation = json_dir / (image.stem + '.json')
        key = image.relative_to(ROOT).as_posix()
        record = indexed[key]
        if (image.stat().st_size, image.stat().st_mtime_ns) != (int(record['size_bytes']), int(record['mtime_ns'])):
            raise ValueError(f'Image changed since exact-content audit: {image}')
        flags = []
        if not annotation.exists():
            flags.append('missing_annotation')
            data = {}
        else:
            data = json.loads(annotation.read_text(encoding='utf-8-sig'))
        width, height = int(record['width']), int(record['height'])
        if annotation.exists() and (data.get('imageWidth'), data.get('imageHeight')) != (width, height):
            flags.append('annotation_image_dimensions_mismatch')
        if annotation.exists() and Path(str(data.get('imagePath', '')).replace('\\', '/')).name != image.name:
            flags.append('annotation_image_filename_mismatch')
        shapes = data.get('shapes', [])
        local_labels, local_types = Counter(), Counter()
        for shape in shapes:
            local_labels[str(shape.get('label', 'MISSING'))] += 1
            local_types[str(shape.get('shape_type', 'MISSING'))] += 1
            flags.extend(shape_issues(shape, width, height))
        labels.update(local_labels)
        types.update(local_types)
        issues.update(set(flags))
        rows.append(dict(image=key, annotation=annotation.relative_to(ROOT).as_posix(),
                         shape_count=len(shapes), labels=json.dumps(dict(local_labels), sort_keys=True),
                         shape_types=json.dumps(dict(local_types), sort_keys=True),
                         issue_flags=';'.join(sorted(set(flags)))))
    csv_path = out / 'rc10519_annotation_inventory.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    extra_annotations = sorted(p.name for p in json_dir.glob('*.json') if not (image_dir / (p.stem + '.jpg')).exists())
    summary = dict(status='completed', created_utc=datetime.now(timezone.utc).isoformat(),
                   local_image_files=len(images), annotation_files=len(list(json_dir.glob('*.json'))),
                   images_with_no_shapes=sum(r['shape_count'] == 0 for r in rows),
                   shape_labels=dict(labels), shape_types=dict(types),
                   images_with_issues=sum(bool(r['issue_flags']) for r in rows), issue_counts=dict(issues),
                   extra_annotations=extra_annotations, archives=archives,
                   source_index_sha256=sha(source), annotation_inventory_sha256=sha(csv_path),
                   publisher_record='datasets/source_provenance/rc10519_publisher_record.json',
                   label_policy='No labels changed. The existing Exp3 builder assigns every local RC image to cracked. Annotation presence alone does not independently validate those labels or failure mechanisms.',
                   limitation='Local archive/extraction consistency and annotation geometry only; no publisher archive digest, original acquisition history, pixel-mask validity or mechanism-label validation established.')
    (out / 'rc10519_local_inventory.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
