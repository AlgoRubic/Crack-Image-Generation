"""Short native-resolution throughput/memory profile; never saves trained weights."""
import copy
import json
import math
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from train_research import ROOT,Images,verify_manifests,AttentionUNet256,GaussianDiffusion


def main(batch_size=4,report_name='training_resource_profile.json'):
    if Path(report_name).name!=report_name:raise ValueError('Use a report filename, not a path')
    path=ROOT/'experiments'/report_name
    if path.exists():raise FileExistsError('Preserve the earlier profile; select a new report filename')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA profiling requires the project GPU')
    results=[]
    for experiment in (4,5,6):
        random.seed(2026);np.random.seed(2026);torch.manual_seed(2026)
        dataset='exp4_level_a_morphology_256' if experiment==4 else 'exp5_exp6_level_bc_mechanisms_256'
        rows=verify_manifests(ROOT/'datasets'/dataset/'research_v4')
        spatial=experiment==6
        model=AttentionUNet256(in_channels=5 if spatial else 3,base_channels=40,num_classes=5 if experiment==4 else 7,cfg_dropout=0.).cuda()
        ema=copy.deepcopy(model).eval();ema.requires_grad_(False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        scaler=torch.amp.GradScaler('cuda')
        diffusion=GaussianDiffusion(device='cuda')
        torch.cuda.reset_peak_memory_stats()
        timings={}
        for phase in ('train','validation'):
            training=phase=='train';net=model if training else ema;net.train(training)
            loader=iter(DataLoader(Images(rows[phase],spatial,training),batch_size=batch_size,shuffle=training,num_workers=0))
            times=[]
            for step in range(7 if training else 4):
                torch.cuda.synchronize();start=time.perf_counter()
                x,p,c=next(loader);x=x.cuda();p=p.cuda();c=c.cuda()
                t=torch.randint(0,1000,(len(x),),device='cuda');noisy,noise=diffusion.add_noise(x,t)
                with torch.set_grad_enabled(training),torch.amp.autocast('cuda',dtype=torch.float16):
                    output=net(torch.cat([noisy,p],1) if spatial else noisy,t,c,force_dropout=False)
                    loss=torch.nn.functional.mse_loss(output.float(),noise)
                if training:
                    optimizer.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(optimizer);scaler.update()
                    with torch.no_grad():
                        for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,.001)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                if step>=(2 if training else 1):times.append(elapsed)
            timings[phase]=dict(measured_batches=len(times),median_batch_seconds=float(np.median(times)),min_batch_seconds=min(times),max_batch_seconds=max(times))
        peak=torch.cuda.max_memory_allocated()/2**30
        epoch=sum(math.ceil(len(rows[s])/batch_size)*timings[s]['median_batch_seconds'] for s in timings)
        report=dict(experiment=experiment,batch_size=batch_size,train_images=len(rows['train']),validation_images=len(rows['validation']),
                    timings=timings,peak_allocated_gpu_gib=peak,estimated_epoch_minutes=epoch/60,
                    estimated_50_epoch_hours=epoch*50/3600,smoke_profile_only=True)
        if peak>4.: raise RuntimeError('Measured memory exceeds 4 GiB budget')
        results.append(report);print(json.dumps(report),flush=True)
        del model,ema,optimizer,loader,x,p,c,noisy,noise,output,loss
        torch.cuda.empty_cache()
    summary=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,results=results,
                 estimate_limitations='Short-batch extrapolation, not observed full-epoch runtime. Includes data loading, field extraction and EMA; excludes checkpoint I/O and long-run thermal/power changes.',
                 trained_weights_saved=False)
    path.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(f'Saved {path}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size',type=int,choices=[1,2,4,6],default=4)
    parser.add_argument('--output',default='training_resource_profile.json')
    args=parser.parse_args();main(args.batch_size,args.output)
