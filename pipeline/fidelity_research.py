"""Leakage-checked, class-balanced FID/KID with one identical preprocessing rule.

Default is selection/preflight only. --compute explicitly runs the metric backend.
Smoke samples may only be inspected with --allow-smoke-preflight, never scored.
"""
import argparse
import csv
import json
import os
import math
import importlib.metadata
import random
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
from checkpoints import ROOT,sha
from evaluate_research import verify_samples
from hash_source_images import hash_original


def assert_disjoint(train,reference):
    for key in ('acquisition_group','original_sha256','original_rgb_sha256'):
        if any(not r.get(key) for r in train+reference):raise ValueError(f'Missing leakage identity: {key}')
        if {r[key] for r in train}&{r[key] for r in reference}:raise ValueError(f'Training/reference overlap: {key}')


def balanced_select(records,classes,count,rng):
    selected=[]
    for cid in range(classes):
        rows=[r for r in records if int(r['class_id'])==cid]
        if len(rows)<count:raise ValueError(f'Class {cid}: only {len(rows)} images available, {count} requested')
        rng.shuffle(rows)
        # Spread first choices across acquisitions; then fill with remaining patches.
        first=[];rest=[];seen=set()
        for row in rows:
            key=row.get('acquisition_group',row.get('path'))
            (rest if key in seen else first).append(row);seen.add(key)
        selected.extend((first+rest)[:count])
    return selected


def main(args):
    folder=Path(args.samples).resolve()
    data,fingerprint=verify_samples(folder,args.allow_smoke_preflight)
    if data['smoke_only'] and args.compute:raise ValueError('Smoke checkpoints cannot produce scientific fidelity scores')
    if args.per_class<1 or (args.compute and args.per_class<2):raise ValueError('Use positive per-class count; metric runs require at least 2')
    if not 0<=args.selection_seed<2**32:raise ValueError('Selection seed must be 0..2^32-1')
    config=data['training_config'];experiment=data['experiment']
    dataset='exp4_level_a_morphology_256' if experiment==4 else 'exp5_exp6_level_bc_mechanisms_256'
    base=ROOT/'datasets'/dataset/config['split_version']
    if config.get('quality_view'):
        if config['quality_view']!='strict_rc_geometry_v1':raise ValueError('Unsupported dataset quality view')
        base=base/config['quality_view']
    tables={}
    for role in ('train',args.reference_split):
        path=base/(role+'.csv')
        if sha(path)!=config['manifests'][role]['sha256']:raise ValueError(f'Reference/training manifest changed: {role}')
        with path.open(encoding='utf-8') as stream:tables[role]=list(csv.DictReader(stream))
    assert_disjoint(tables['train'],tables[args.reference_split])
    rng=random.Random(args.selection_seed);classes=5 if experiment==4 else 7
    real=balanced_select(tables[args.reference_split],classes,args.per_class,rng)
    fake=balanced_select(data['images'],classes,args.per_class,random.Random(args.selection_seed))
    references=[]
    for row in real:
        path=ROOT/row['path']
        with Image.open(path) as image:
            if image.size!=(256,256):raise ValueError('Unexpected native reference dimensions')
        hashed,_=hash_original(row['path'],{})
        references.append(dict(path=row['path'],class_id=int(row['class_id']),acquisition_group=row['acquisition_group'],sha256=hashed['sha256'],rgb_sha256=hashed['decoded_rgb_sha256']))
    selected_pixels={r['rgb_sha256'] for r in references}
    for i,row in enumerate(tables['train'],1):
        hashed,_=hash_original(row['path'],{})
        if hashed['decoded_rgb_sha256'] in selected_pixels:raise ValueError('Selected reference exactly duplicates a decoded training patch')
        if i%2000==0:print(f'Checked {i} training patches for selected-reference pixel overlap',flush=True)
    if Path(args.run_id).name!=args.run_id or args.run_id in ('.','..'):raise ValueError('Invalid run ID')
    out=folder/args.run_id;out.mkdir(exist_ok=False)
    report=dict(status='preflight_complete',smoke_only=data['smoke_only'],reference_split=args.reference_split,
                sampling_manifest_sha256=fingerprint,per_class=args.per_class,real_images=len(real),fake_images=len(fake),
                unique_reference_acquisitions=len({r['acquisition_group'] for r in real}),
                unique_reference_rgb_images=len(selected_pixels),selected_reference_training_patch_overlap=0,
                selection_seed=args.selection_seed,preprocessing=args.preprocessing,reference_selection=references,
                fake_selection=[dict(path=r['path'],sha256=r['sha256'],class_id=r['class_id']) for r in fake],
                caveats=['Patch samples can share acquisitions; image count is not independent-original count.',
                         'Small-sample FID is sensitive to sample budget; KID subset std is not a confidence interval.',
                         'Morphological labels do not independently establish structural mechanism.'])
    record=out/'FIDELITY_METRICS.json';record.write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not args.compute:
        print(f'Preflight passed; no FID/KID computed: {record}');return
    os.environ['TORCH_HOME']=str(ROOT/'.cache/torch')
    import torch
    import torch_fidelity
    from train_exp6_spatial_physics_256 import normalize_chromatic_cast
    try:
        with tempfile.TemporaryDirectory(dir=out) as work:
            dirs={kind:Path(work)/kind for kind in ('real','fake')}
            for dest in dirs.values():dest.mkdir()
            for kind,selection in [('real',references),('fake',fake)]:
                for i,row in enumerate(selection):
                    path=ROOT/row['path'] if kind=='real' else folder/row['path']
                    with Image.open(path) as image:
                        rgb=image.convert('RGB')
                        if args.preprocessing=='common_chroma':rgb=Image.fromarray(normalize_chromatic_cast(np.array(rgb)))
                        rgb.save(dirs[kind]/f'{i:06d}.png')
            metrics=torch_fidelity.calculate_metrics(input1=str(dirs['fake']),input2=str(dirs['real']),
                fid=True,kid=True,cuda=False,batch_size=8,kid_subsets=8,
                kid_subset_size=min(100,len(real)),rng_seed=args.selection_seed,verbose=False)
        if not all(math.isfinite(float(v)) for v in metrics.values()):raise ValueError('Nonfinite fidelity metric')
        report.update(status='completed',metrics={k:float(v) for k,v in metrics.items()},
                      backend='torch_fidelity',feature_device='cpu',feature_precision='float32',
                      backend_version=importlib.metadata.version('torch-fidelity'),
                      feature_preprocessing='Same backend and pixel transform on real and fake inputs')
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__)
        record.write_text(json.dumps(report,indent=2),encoding='utf-8');raise
    record.write_text(json.dumps(report,indent=2),encoding='utf-8');print(f'Saved {record}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--samples',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--per-class',type=int,required=True);p.add_argument('--reference-split',choices=['validation','test'],default='validation')
    p.add_argument('--selection-seed',type=int,default=2026);p.add_argument('--preprocessing',choices=['rgb','common_chroma'],default='rgb')
    p.add_argument('--compute',action='store_true');p.add_argument('--allow-smoke-preflight',action='store_true')
    main(p.parse_args())
