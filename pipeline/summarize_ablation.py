"""Audit paired ablations and compare fixed/calibrated segmentation sensitivity.

Missing detected cracks are counted explicitly; no fabricated distance penalty is used.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validate_detector import ROOT, fissure_mask


def geometry(mask,command):
    core=command[0]>.22
    if not core.any():
        return dict(recall=None,mae=None,no_detection=not mask.any())
    if not mask.any():
        return dict(recall=0.,mae=None,no_detection=True)
    distances=np.concatenate([distance_transform_edt(~mask)[core],distance_transform_edt(~core)[mask]])
    return dict(recall=float(binary_dilation(mask,iterations=4)[core].mean()),
                mae=float(distances.mean()),no_detection=False)


def main(run):
    with (run/'per_sample.csv').open() as stream:
        records=list(csv.DictReader(stream))
    config=json.loads((run/'config.json').read_text())
    if config['status']!='completed':
        raise ValueError('Run is not completed')
    groups={}
    for row in records:
        key=(int(row['class_id']),int(row['seed']),int(row['index']))
        groups.setdefault(key,{})[row['variant']]=row
    expected=len(config['classes'])*len(config['seeds'])*config['count']
    if len(groups)!=expected:
        raise ValueError('Incomplete groups')
    for group in groups.values():
        if set(group)!={'correct','zero','shuffled'} or len({r['initial_rng_sha256'] for r in group.values()})!=1:
            raise ValueError('Unmatched variants or initial RNG states')
    controls_equal=True
    for (cid,seed,i),group in groups.items():
        if cid==0:
            folder=run/f'non_cracked_seed{seed}'
            hashes={hashlib.sha256((folder/f'{mode}_{i:03d}.png').read_bytes()).hexdigest() for mode in group}
            controls_equal &= len(hashes)==1
    if not controls_equal:
        raise ValueError('Identical non-cracked control conditions produced different outputs')
    calibrated=json.loads((ROOT/'experiments/exp6_level_c_spatial_physics/detector_calibration_v1/selected_detector.json').read_text())['config']
    detectors={'fixed':dict(window=23,threshold=24.5,opening=2),'calibrated':calibrated}
    measured=[]
    for row in records:
        if int(row['class_id'])==0: continue
        folder=run/f'{row["class_name"]}_seed{row["seed"]}'
        i=int(row['index'])
        with Image.open(folder/f'{row["variant"]}_{i:03d}.png') as image:
            rgb=np.array(image.convert('RGB'))
        command=np.load(folder/f'command_{i:03d}.npy')
        for detector,params in detectors.items():
            measured.append(dict(class_name=row['class_name'],seed=row['seed'],index=i,variant=row['variant'],
                                 detector=detector,**geometry(fissure_mask(rgb,**params),command)))
    summary={}
    for detector in detectors:
        summary[detector]={}
        for variant in ('correct','zero','shuffled'):
            selected=[r for r in measured if r['detector']==detector and r['variant']==variant]
            distances=[r['mae'] for r in selected if r['mae'] is not None]
            summary[detector][variant]=dict(n=len(selected),mean_recall=float(np.mean([r['recall'] for r in selected])),
                no_detection=sum(r['no_detection'] for r in selected),
                conditional_mae_px=float(np.mean(distances)) if distances else None,
                distance_samples=len(distances))
    report=dict(paired_rng_verified=True,noncracked_controls_byte_identical=controls_equal,
                generated_images=len(records),seeds=config['seeds'],summary=summary,
                limitations=['Small historical-checkpoint pilot, not clean-lineage generalization.',
                             'Zero-field/requested-class pairing may be outside the training distribution.',
                             'Calibrated detector has only exploratory real-mask evidence, not generated-domain validation.',
                             'Distances are conditional on nonempty detections; compare alongside no-detection counts.',
                             'Legacy summary MAE used a 50px penalty for absent detections; this report omits those distances.'])
    (run/'audited_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (run/'detector_sensitivity.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(measured[0]));writer.writeheader();writer.writerows(measured)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,detector in zip(axes,detectors):
        classes=sorted({r['class_name'] for r in measured})
        x=np.arange(len(classes))
        for j,variant in enumerate(('correct','zero','shuffled')):
            vals=[np.mean([r['recall'] for r in measured if r['detector']==detector and r['variant']==variant and r['class_name']==c]) for c in classes]
            ax.bar(x+(j-1)*.25,vals,.25,label=variant)
        ax.set_xticks(x,[c.replace('_','\n') for c in classes],fontsize=8)
        ax.set(ylim=(0,1.05),ylabel='Commanded-trajectory recall',title=detector+' detector')
        ax.legend()
    fig.suptitle('Paired spatial-control pilot | detector sensitivity')
    for ext in ('png','svg'):
        fig.savefig(run/('audited_comparison.'+ext),dpi=200)
    plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    main(p.parse_args().run)
