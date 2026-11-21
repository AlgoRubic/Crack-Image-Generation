"""Select detector parameters on validation only; freeze before exploratory test comparison."""
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from validate_detector import ROOT, metrics, fissure_mask

DATA = ROOT / 'datasets/exp5_exp6_level_bc_mechanisms_256/research_v1'
OUT = ROOT / 'experiments/exp6_level_c_spatial_physics/detector_calibration_v1'
BASELINE = dict(window=23, threshold=24.5, opening=2)
CANDIDATES = [dict(window=w, threshold=t, opening=o) for w,t,o in
              itertools.product([15,23,31], [12.,18.,24.5,30.], [1,2])]


def originals(split):
    with (DATA / (split+'.csv')).open(encoding='utf-8') as stream:
        return {r['original_sha256']: r['original_path'] for r in csv.DictReader(stream) if r['source']=='DeepCrack'}


def tiles(path):
    path = ROOT / path
    label = path.parent.parent / path.parent.name.replace('_img','_lab') / (path.stem+'.png')
    with Image.open(path) as im, Image.open(label) as gt:
        rgb, target = np.array(im.convert('RGB')), np.array(gt.convert('L')) > 127
    if rgb.shape[:2] != target.shape:
        raise ValueError('Mask shape mismatch')
    for y in range(0,rgb.shape[0]-255,256):
        for x in range(0,rgb.shape[1]-255,256):
            yield rgb[y:y+256,x:x+256],target[y:y+256,x:x+256]


def count(rgb, gt, config):
    pred = fissure_mask(rgb, **config)
    return np.array([(pred & gt).sum(),(pred & ~gt).sum(),(~pred & gt).sum()],dtype=np.int64)


def select(validation_scores):
    # Deterministic first-candidate tie break; no test scores are accepted here.
    return max(validation_scores, key=lambda r:r['iou'])


def main():
    OUT.mkdir(exist_ok=False)
    val, test = originals('validation'), originals('test')
    if not val or not test or set(val)&set(test):
        raise ValueError('Missing data or validation/test original-content overlap')
    protocol = dict(candidates=CANDIDATES, objective='validation pixel-micro IoU',
                    tile_policy='native nonoverlapping 256px; incomplete edge strips excluded',
                    validation_originals=len(val), test_originals=len(test),
                    caveat='Exploratory: baseline test results were previously inspected. No test thresholds enter selection.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    totals = np.zeros((len(CANDIDATES),3),dtype=np.int64)
    for path in val.values():
        for rgb, gt in tiles(path):
            for i, config in enumerate(CANDIDATES):
                totals[i] += count(rgb,gt,config)
    scores = [dict(config=config,**metrics(*total)) for config,total in zip(CANDIDATES,totals)]
    chosen = select(scores)
    # Persist the selection before loading any test-image pixels.
    (OUT/'validation_search.json').write_text(json.dumps(scores,indent=2),encoding='utf-8')
    locked = json.dumps(chosen,sort_keys=True,indent=2)
    (OUT/'selected_detector.json').write_text(locked,encoding='utf-8')
    rows, paired = [], []
    for group,path in sorted(test.items()):
        totals = np.zeros((2,3),dtype=np.int64)
        for rgb,gt in tiles(path):
            for i, config in enumerate([BASELINE,chosen['config']]):
                totals[i] += count(rgb,gt,config)
        paired.append(totals)
        rows.append(dict(original_path=path,original_sha256=group,
                         baseline_iou=metrics(*totals[0])['iou'],selected_iou=metrics(*totals[1])['iou']))
    counts = np.array(paired)
    baseline, selected = [metrics(*c) for c in counts.sum(axis=0)]
    rng = np.random.default_rng(2026)
    differences = []
    for _ in range(2000):
        draw = counts[rng.integers(0,len(counts),len(counts))].sum(axis=0)
        differences.append(metrics(*draw[1])['iou']-metrics(*draw[0])['iou'])
    summary = dict(validation_originals=len(val),test_originals=len(test), selected_config=chosen['config'],
                   selection_sha256=hashlib.sha256(locked.encode()).hexdigest(),
                   validation_iou=chosen['iou'],test_baseline=baseline,test_selected=selected,
                   paired_iou_gain=selected['iou']-baseline['iou'],
                   paired_original_bootstrap_95_ci=np.quantile(differences,[.025,.975]).tolist(),
                   interpretation='Exploratory detector improvement only; not validated generated-image mechanism accuracy',
                   historical_evaluator_changed=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (OUT/'per_original_test.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    keys=list(baseline); x=np.arange(len(keys))
    axes[0].bar(x-.18,[baseline[k] for k in keys],.36,label='Fixed baseline',color='#697278')
    axes[0].bar(x+.18,[selected[k] for k in keys],.36,label='Validation-selected',color='#237a89')
    axes[0].set_xticks(x,keys);axes[0].set(ylim=(0,1),title='Exploratory test pixel metrics');axes[0].legend()
    axes[1].scatter([r['baseline_iou'] for r in rows],[r['selected_iou'] for r in rows],alpha=.7,color='#237a89')
    axes[1].plot([0,1],[0,1],linestyle='--',color='gray')
    axes[1].set(xlim=(0,1),ylim=(0,1),xlabel='Baseline original-image IoU',ylabel='Selected original-image IoU',title='Paired original-image comparison')
    fig.suptitle('Detector calibration: validation-only selection, frozen test comparison')
    for ext in ('png','svg'):
        fig.savefig(OUT/('calibration_comparison.'+ext),dpi=200)
    plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
