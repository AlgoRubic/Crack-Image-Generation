"""Fixed detector versus DeepCrack pixel masks on prospective held-out originals.

Uses native, nonoverlapping 256px tiles; does not tune on test masks.
Historical generative models already saw these originals. This is detector testing,
not an independent validation of those historical generative models.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from crack_segmentation import fissure_mask


def metrics(tp, fp, fn):
    def ratio(a, b):
        return a / b if b else None
    return dict(precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn),
                iou=ratio(tp, tp+fp+fn), dice=ratio(2*tp, 2*tp+fp+fn))


def main():
    manifest = ROOT / 'datasets/exp5_exp6_level_bc_mechanisms_256/research_v1/test.csv'
    with manifest.open(encoding='utf-8') as stream:
        originals = sorted({r['original_path'] for r in csv.DictReader(stream) if r['source'] == 'DeepCrack'})
    out = ROOT / 'experiments/exp6_level_c_spatial_physics/detector_validation'
    out.mkdir(exist_ok=True)
    records, panels = [], []
    for original in originals:
        path = ROOT / original
        lab = path.parent.parent / path.parent.name.replace('_img', '_lab') / (path.stem + '.png')
        with Image.open(path) as image, Image.open(lab) as mask:
            rgb, gt = np.array(image.convert('RGB')), np.array(mask.convert('L')) > 127
        if rgb.shape[:2] != gt.shape:
            raise ValueError(f'Image/mask shape mismatch: {path}')
        tp = fp = fn = tiles = pixels = 0
        for y in range(0, rgb.shape[0]-255, 256):
            for x in range(0, rgb.shape[1]-255, 256):
                tile, target = rgb[y:y+256,x:x+256], gt[y:y+256,x:x+256]
                pred = fissure_mask(tile)
                a, b, c = int((pred & target).sum()), int((pred & ~target).sum()), int((~pred & target).sum())
                tp += a; fp += b; fn += c; tiles += 1; pixels += target.size
                if tiles == 1:
                    overlay = tile.copy()
                    overlay[pred & ~target] = [235,60,60]
                    overlay[target & ~pred] = [40,220,100]
                    overlay[target & pred] = [30,180,240]
                    panels.append((metrics(a,b,c)['iou'] or 0, original, tile.copy(), overlay))
        if not tiles:
            raise ValueError(f'No native 256px tiles available: {path}')
        records.append(dict(original_path=original, tiles=tiles, evaluated_pixels=pixels,
                            tp=tp, fp=fp, fn=fn, **metrics(tp,fp,fn)))
    if not records:
        raise ValueError('No DeepCrack test originals found')
    with (out / 'per_original.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    counts = np.array([[r[k] for k in ('tp','fp','fn')] for r in records])
    aggregate = metrics(*counts.sum(axis=0))
    rng = np.random.default_rng(2026)
    draws = {k: [] for k in aggregate}
    for _ in range(2000):
        sampled = counts[rng.integers(0, len(records), len(records))].sum(axis=0)
        for k,v in metrics(*sampled).items():
            if v is not None:
                draws[k].append(v)
    ci = {k: np.quantile(v, [.025,.975]).tolist() for k,v in draws.items()}
    summary = dict(original_images=len(records), tiles=sum(r['tiles'] for r in records),
                   detector='fixed local window 23, dark contrast >24.5, 2x2 opening',
                   mask_threshold=127, aggregation='pixel-micro metrics; original-image cluster bootstrap, 2000 draws',
                   selection='DeepCrack originals in prospective research_v1/test.csv; native nonoverlapping 256px tiles, incomplete edge strips excluded',
                   tuning='none; fixed historical detector', metrics=aggregate, bootstrap_95_ci=ci,
                   caveats=['Real-mask performance does not establish accuracy on generated images.',
                            'Masks annotate visible cracks, not structural failure mechanisms.',
                            'Historical generator exposure means this is not an unseen-data test of historical generators.'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(8,5), layout='constrained')
    keys = list(aggregate)
    vals = np.array([aggregate[k] for k in keys])
    bounds = np.array([ci[k] for k in keys])
    ax.bar(keys, vals, color='#237a89')
    ax.errorbar(range(len(keys)), vals, yerr=[vals-bounds[:,0], bounds[:,1]-vals], fmt='none', color='black', capsize=5)
    ax.set(ylim=(0,1), ylabel='Pixel-level metric', title=f'Fixed crack detector versus human masks\n{len(records)} prospective test originals; cluster-bootstrap 95% intervals')
    for ext in ('png','svg'):
        fig.savefig(out / ('detector_metrics.'+ext), dpi=200)
    plt.close(fig)
    panels.sort(key=lambda v:v[0])
    selected = panels[:3] + panels[-3:]
    canvas = Image.new('RGB',(512, len(selected)*286),'white')
    draw = ImageDraw.Draw(canvas)
    for i,(score,name,tile,overlay) in enumerate(selected):
        canvas.paste(Image.fromarray(tile),(0,i*286+30))
        canvas.paste(Image.fromarray(overlay),(256,i*286+30))
        draw.text((4,i*286+5), f'{Path(name).name} | tile IoU {score:.3f}', fill='black')
    canvas.save(out / 'examples.png')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
