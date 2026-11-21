"""Prospective v2 grouping: SDNET filename-prefix acquisitions, other source images.

SDNET grouping is inferred from filenames and corroborated against the dataset's
documented surface acquisition counts (54 decks, 72 walls, 104 pavements).
It does not establish site/structure independence or rule out near duplicates.
"""
import csv
import hashlib
import json
import re
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from research import ROOT, assign_split, write_csv


def acquisition(row):
    path=Path(row['original_path'])
    source=row.get('source') or ('SDNET2018' if 'SDNET2018' in path.parts else None)
    if source=='SDNET2018':
        surface=path.parent.parent.name
        prefix,sep,tile=path.stem.partition('-')
        if surface not in ('D','W','P') or not sep or not re.fullmatch(r'\d+(?:_\d+)?',tile):
            raise ValueError(f'Cannot infer SDNET acquisition: {path}')
        return f'SDNET2018:{surface}:{prefix}'
    if source is None:
        raise ValueError(f'Missing dataset identity: {path}')
    return source+':'+path.as_posix()


def main(version='research_v4'):
    raw=ROOT/'data/raw/SDNET2018/images'
    groups={acquisition(dict(source='SDNET2018',original_path=str(p.relative_to(ROOT)))) for p in raw.rglob('*.jpg')}
    counts=Counter(g.split(':')[1] for g in groups)
    if counts!={'D':54,'W':72,'P':104}:
        raise ValueError(f'Acquisition grouping does not match source documentation: {counts}')
    all_assignments={}
    reports=[]
    prepared={}
    for dataset in ('exp1_exp2_sdnet_64','exp3_augmented_128','exp4_level_a_morphology_256','exp5_exp6_level_bc_mechanisms_256'):
        folder=ROOT/'datasets'/dataset
        if (folder/'research_v1/provenance.csv').exists():
            inputs=[folder/'research_v1/provenance.csv']
        else:
            inputs=[folder/'train.csv']
            if (folder/'validation.csv').exists():
                inputs.append(folder/'validation.csv')
        rows=[]
        for file in inputs:
            with file.open(encoding='utf-8-sig') as stream:
                rows.extend(csv.DictReader(stream))
        prepared[dataset]=(folder,inputs,rows)
    # Link source acquisitions and all known identical original-image contents.
    # Canonical component IDs are deterministic lexical minima, shared across experiments.
    parents={}
    def find(key):
        parents.setdefault(key,key)
        if parents[key]!=key:
            parents[key]=find(parents[key])
        return parents[key]
    def union(a,b):
        a,b=find(a),find(b)
        parents[max(a,b)]=min(a,b)
    known_by_path={r['original_path']:r['original_sha256'] for _,_,rows in prepared.values() for r in rows if r.get('original_sha256')}
    content_index={}
    if version=='research_v4':
        index=ROOT/'datasets/source_content_v1/index.csv'
        summary=json.loads(index.with_name('summary.json').read_text())
        if summary['status']!='completed' or hashlib.sha256(index.read_bytes()).hexdigest()!=summary['index_sha256']:
            raise ValueError('Source-content audit is incomplete or index changed')
        with index.open(encoding='utf-8') as stream:
            content_index={r['original_path']:r for r in csv.DictReader(stream)}
        for _,_,rows in prepared.values():
            for row in rows:
                record=content_index[row['original_path']]
                row['original_sha256']=record['sha256']
                row['original_rgb_sha256']=record['decoded_rgb_sha256']
    first_by_content={}
    for _,_,rows in prepared.values():
        for row in rows:
            group=acquisition(row)
            find(group)
            content=row.get('original_sha256') or known_by_path.get(row['original_path'])
            if content:
                row['original_sha256']=content
                union(group,first_by_content.setdefault(content,group))
            if row.get('original_rgb_sha256'):
                key='rgb:'+row['original_rgb_sha256']
                union(group,first_by_content.setdefault(key,group))
    linked_acquisitions=sum(find(k)!=k for k in list(parents))
    for dataset,(folder,inputs,rows) in prepared.items():
        by_group=defaultdict(set)
        for row in rows:
            original_group=acquisition(row)
            group=find(original_group)
            split=assign_split(group,2026)
            if all_assignments.setdefault(group,split)!=split:
                raise ValueError('Cross-experiment group assignment mismatch')
            row.update(acquisition_group=group,source_acquisition_group=original_group,split=split,historical_training_exposure='possible_or_known',
                       split_version=version)
            by_group[group].add(split)
        out=folder/version;out.mkdir(exist_ok=False)
        coverage={}
        for split in ('train','validation','test'):
            selected=[r for r in rows if r['split']==split]
            write_csv(out/(split+'.csv'),selected)
            coverage[split]=dict(Counter(r.get('class_name',r.get('label','unknown')) for r in selected))
        classes={r.get('class_name',r.get('label','unknown')) for r in rows}
        report=dict(dataset=dataset,rows=len(rows),unique_groups=len(by_group),
                    acquisition_overlap=sum(len(v)>1 for v in by_group.values()),
                    split_counts=dict(Counter(r['split'] for r in rows)),class_counts=coverage,
                    missing_classes={s:sorted(classes-set(c)) for s,c in coverage.items()},
                    input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                    grouping='Connected components of SDNET acquisition IDs, other source-image paths, and known original-content SHA256 aliases',
                    linked_acquisitions=linked_acquisitions,
                    source_content_audit_complete=bool(content_index),
                    source_content_index_sha256=summary['index_sha256'] if content_index else None,
                    sdnet_source='https://digitalcommons.usu.edu/all_datasets/48/',
                    inference='SDNET group identities inferred from filenames; counts match 54 deck/72 wall/104 pavement source photographs',
                    caveats=['Structure/site independence not established','Near-duplicate audit outstanding',
                             'Existing trained checkpoints are exposed; these are prospective splits only'])
        (out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        reports.append(report)
        print(json.dumps(report),flush=True)
    (ROOT/'datasets/acquisition_split_audit.json').write_text(json.dumps(dict(reports=reports,
        sdnet_acquisition_groups=dict(counts),cross_experiment_assignment_consistent=True),indent=2),encoding='utf-8')
    (ROOT/'datasets/current_research_split.json').write_text(json.dumps(dict(version=version,
        datasets={report['dataset']:report['split_counts'] for report in reports},
        exact_rgb_content_audit_complete=bool(content_index)),indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',choices=['research_v3','research_v4'],default='research_v4')
    main(parser.parse_args().version)
