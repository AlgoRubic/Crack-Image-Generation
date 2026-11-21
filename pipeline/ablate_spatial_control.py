"""Paired spatial-conditioning ablation using a fixed historical checkpoint.

All variants use identical initial noise; every output is scored against the
original correct command. This measures sensitivity, not causal physics validity.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from model_attention import AttentionUNet256
from diffusion import GaussianDiffusion
from physics_trajectory_solver import solve_physical_trajectory
from sample_exp6_spatial_physics_256 import ddim_sample_dual_cfg
from evaluate_exp6_spatial_physics_256 import analyze_sample, CLASSES


def variants(fields):
    if len(fields)<2:
        raise ValueError('At least two fields needed for a within-class shuffle')
    return dict(correct=fields,zero=torch.zeros_like(fields),shuffled=fields.roll(1,0))


def main(args):
    if not 2 <= args.count <= 4 or not 2 <= args.steps <= 1000:
        raise ValueError('Use count 2..4 for bounded mini-batches and steps 2..1000')
    if Path(args.run_id).name!=args.run_id or args.run_id in ('.','..'):
        raise ValueError('Run ID must be a directory name')
    out=ROOT/'experiments/exp6_level_c_spatial_physics/ablations'/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    checkpoint=ROOT/'models/weights/exp6_level_c_spatial_physics_256/exp6_level_c_spatial_physics_256_epoch015.pt'
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    model=AttentionUNet256(in_channels=5,out_channels=3,base_channels=40,num_classes=7,cfg_dropout=0.).to(device)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model']);del state
    diffusion=GaussianDiffusion(device=device)
    config=dict(vars(args),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                protocol='Same initial noise, class and checkpoint; only conditioning fields vary. All scores use original correct command.',
                shuffle='Cyclic permutation across same-class field batch; requested class is retained for zero-field variant',
                zero_field_caveat='Zero spatial field with a present requested class can be out-of-distribution because historical Dual-CFG jointly dropped class and fields.',
                status='running',scope='Pilot historical-checkpoint sensitivity analysis; not matched clean-lineage class-only training comparison',
                detector='Fixed historical 23px / 24.5 / 2x2 heuristic; calibration uncertainty applies')
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    records=[]
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats()
    for cid in args.classes:
        for seed in args.seeds:
            folder=out/f'{CLASSES[cid]}_seed{seed}';folder.mkdir()
            commands=[]
            for i in range(args.count):
                field,_=solve_physical_trajectory(cid,seed=seed+1974+cid*10000+i)
                commands.append(torch.from_numpy(field))
                np.save(folder/f'command_{i:03d}.npy',field)
            fields=torch.stack(commands)
            for mode,control in variants(fields).items():
                torch.manual_seed(seed)
                if device.type=='cuda':
                    torch.cuda.manual_seed_all(seed)
                rng_state=torch.cuda.get_rng_state() if device.type=='cuda' else torch.get_rng_state()
                rng_hash=hashlib.sha256(rng_state.cpu().numpy().tobytes()).hexdigest()
                field_hash=hashlib.sha256(control.numpy().tobytes()).hexdigest()
                started=time.perf_counter()
                output=ddim_sample_dual_cfg(model,diffusion,cid,control,steps=args.steps,cfg_scale=args.cfg_scale,device=device).cpu()
                elapsed=time.perf_counter()-started
                save_image(output,folder/f'{mode}_grid.png',nrow=args.count)
                for i in range(args.count):
                    path=folder/f'{mode}_{i:03d}.png';save_image(output[i],path)
                    stat=analyze_sample(path,folder/f'command_{i:03d}.npy',cid)
                    records.append(dict(class_id=cid,class_name=CLASSES[cid],seed=seed,index=i,variant=mode,
                                        trajectory_recall=stat['traj_recall'],trajectory_mae_px=stat['traj_mae_px'],
                                        trajectory_rmse_px=stat['traj_rmse_px'],predicted_class=stat['pred_class'],seconds_per_batch=elapsed,
                                        initial_rng_sha256=rng_hash, conditioning_sha256=field_hash))
                print(f'{CLASSES[cid]} seed={seed} {mode}: {elapsed:.2f}s',flush=True)
    with (out/'per_sample.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    summary={}
    for mode in ('correct','zero','shuffled'):
        selected=[r for r in records if r['variant']==mode and r['class_id']!=0]
        summary[mode]=dict(n=len(selected),mean_trajectory_recall=float(np.mean([r['trajectory_recall'] for r in selected])),
                           mean_trajectory_mae_px=float(np.mean([r['trajectory_mae_px'] for r in selected]))) if selected else {}
    paired=[]
    for cid in args.classes:
        if cid==0: continue
        for seed in args.seeds:
            for i in range(args.count):
                group={r['variant']:r for r in records if r['class_id']==cid and r['seed']==seed and r['index']==i}
                paired.append(dict(class_id=cid,seed=seed,index=i,
                                   correct_minus_zero_recall=group['correct']['trajectory_recall']-group['zero']['trajectory_recall'],
                                   correct_minus_shuffled_recall=group['correct']['trajectory_recall']-group['shuffled']['trajectory_recall']))
    summary['paired_differences']=paired
    peak=torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else 0
    summary['peak_gpu_gib']=peak
    summary['publication_ready']=False
    if peak>4:
        raise RuntimeError('GPU budget exceeded')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    modes=['correct','zero','shuffled']
    for ax,key,title in zip(axes,['mean_trajectory_recall','mean_trajectory_mae_px'],['Trajectory recall','Trajectory MAE (pixels)']):
        ax.bar(modes,[summary[m].get(key,0) for m in modes],color=['#237a89','#697278','#d97732'])
        ax.set_title(title)
    fig.suptitle('Paired spatial-control pilot | identical initial noise')
    fig.supxlabel('Historical checkpoint; heuristic measurements; small pilot sample, not a physical-validity claim.')
    for ext in ('png','svg'):
        fig.savefig(out/('ablation_comparison.'+ext),dpi=200)
    plt.close(fig)
    config['status']='completed';(out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',required=True)
    p.add_argument('--classes',nargs='+',type=int,choices=list(CLASSES),default=[1,2])
    p.add_argument('--seeds',nargs='+',type=int,default=[2026])
    p.add_argument('--count',type=int,default=2)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--cfg-scale',type=float,default=2.5)
    main(p.parse_args())
