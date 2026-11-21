"""Sample new clean-lineage research checkpoints, defaulting to EMA weights.

Requires explicit opt-in for smoke checkpoints. No hidden color postprocessing;
class-only and spatial controls use the same sampler and initial-noise protocol.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
from torchvision.utils import save_image
from checkpoints import ROOT,sha
sys.path.insert(0,str(ROOT/'src'))
from model_attention import AttentionUNet256
from diffusion import GaussianDiffusion
from physics_trajectory_solver import solve_physical_trajectory

MECHANISMS=['non_cracked','flexural','flexure_shear','web_shear','torsion','corrosion_splitting','shrinkage_map']
MORPHOLOGY=['non_cracked','longitudinal','transverse','diagonal','alligator_map']


def validate_config(config,allow_smoke):
    if config.get('initialization')!='scratch' or config.get('transfer_ancestors')!=[]:
        raise ValueError('Checkpoint is not a clean scratch research lineage')
    if config.get('smoke_only') and not allow_smoke:raise ValueError('Smoke checkpoint requires --allow-smoke')
    if config.get('experiment') not in (4,5,6) or config.get('conditioning') not in ('class_only','spatial'):
        raise ValueError('Research checkpoint configuration is missing or unsupported')
    if config['experiment']!=6 and config['conditioning']!='class_only':raise ValueError('Invalid experiment/conditioning pairing')
    if config.get('split_version') not in ('research_v4','research_v5'):raise ValueError('Research sampling requires an audited research_v4/v5 lineage')
    if not config.get('smoke_only') and config.get('split_version')!='research_v5':
        raise ValueError('Full research sampling requires similarity-isolated research_v5 lineage')


@torch.no_grad()
def sample(model,diffusion,initial,labels,fields,steps,guidance):
    image=initial.clone();model.eval()
    null_labels=torch.full_like(labels,model.null_token_idx)
    times=torch.linspace(999,0,steps,dtype=torch.long,device=image.device)
    for i,t in enumerate(times):
        batch_t=torch.full_like(labels,t)
        cond=torch.cat((image,fields),1) if fields is not None else image
        null=torch.cat((image,torch.full_like(fields,-1)),1) if fields is not None else image
        with torch.amp.autocast('cuda',enabled=image.device.type=='cuda',dtype=torch.float16):
            eps_c=model(cond,batch_t,labels,force_dropout=False)
            eps_u=model(null,batch_t,null_labels,force_dropout=True)
            eps=eps_u+guidance*(eps_c-eps_u)
        alpha=diffusion.alpha_bars[t]
        x0=((image-torch.sqrt(1-alpha)*eps)/torch.sqrt(alpha)).clamp(-1,1)
        image=torch.sqrt(diffusion.alpha_bars[times[i+1]])*x0+torch.sqrt(1-diffusion.alpha_bars[times[i+1]])*eps if i+1<len(times) else x0
    return ((image.clamp(-1,1)+1)/2).cpu()


def main(args):
    if args.count<1 or not 1<=args.batch_size<=4 or not 2<=args.steps<=1000 or not math.isfinite(args.guidance) or args.guidance<0:
        raise ValueError('Invalid count, mini-batch, sampling steps or guidance')
    if not 0<=args.seed<2**32 or Path(args.run_id).name!=args.run_id or args.run_id in ('.','..'):
        raise ValueError('Invalid seed or run ID')
    checkpoint=Path(args.checkpoint).resolve()
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=state.get('config',{});validate_config(config,args.allow_smoke)
    experiment=config['experiment'];classes=MORPHOLOGY if experiment==4 else MECHANISMS
    names=['crack_type_conditional','structural_mechanism_ddpm','exp6_level_c_spatial_physics']
    out=ROOT/'experiments'/names[experiment-4]/'research_samples'/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model=AttentionUNet256(in_channels=5 if experiment==6 else 3,base_channels=40,num_classes=len(classes),cfg_dropout=0.).to(device)
    model.load_state_dict(state[args.weights]);del state
    diffusion=GaussianDiffusion(device=device)
    record=dict(status='running',experiment=experiment,conditioning=config['conditioning'],weights=args.weights,
                smoke_only=config['smoke_only'],checkpoint_sha256=sha(checkpoint),training_config=config,
                count_per_class=args.count,seed=args.seed,steps=args.steps,guidance=args.guidance,
                sampler_sha256=sha(Path(__file__)),postprocessing='Fixed [-1,1] to [0,1] scaling only; no chroma transform',images=[])
    manifest=out/'sampling_manifest.json'
    manifest.write_text(json.dumps(record,indent=2),encoding='utf-8')
    for cid,cname in enumerate(classes):
        folder=out/cname;folder.mkdir()
        physics=out/(cname+'_physics_masks')
        if experiment!=4:physics.mkdir()
        for start in range(0,args.count,args.batch_size):
            count=min(args.batch_size,args.count-start)
            noise=[];fields=[];commands=[]
            for index in range(start,start+count):
                gen=torch.Generator(device=device).manual_seed(args.seed+cid*1000000+index)
                noise.append(torch.randn((3,256,256),generator=gen,device=device))
                if experiment!=4:
                    command,_=solve_physical_trajectory(cid,seed=args.seed+1974+cid*100000+index)
                    commands.append(command)
                    fields.append(torch.from_numpy(command if config['conditioning']=='spatial' else np.zeros_like(command)))
            initial=torch.stack(noise)
            conditioning=(torch.stack(fields).to(device)*2-1) if experiment==6 else None
            outputs=sample(model,diffusion,initial,torch.full((count,),cid,dtype=torch.long,device=device),conditioning,args.steps,args.guidance)
            for offset,image in enumerate(outputs):
                index=start+offset;path=folder/f'sample_{index:04d}.png';save_image(image,path)
                item=dict(class_id=cid,class_name=cname,index=index,path=path.relative_to(out).as_posix(),sha256=sha(path),
                          initial_noise_sha256=hashlib.sha256(initial[offset].cpu().numpy().tobytes()).hexdigest())
                if experiment!=4:
                    command_path=physics/f'phys_{index:04d}.npy';np.save(command_path,commands[offset])
                    item.update(command_path=command_path.relative_to(out).as_posix(),command_sha256=sha(command_path))
                record['images'].append(item)
            manifest.write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(f'{cname}: {args.count} images',flush=True)
    record['status']='completed';manifest.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(f'Saved {out}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--weights',choices=['ema','model'],default='ema');p.add_argument('--allow-smoke',action='store_true')
    p.add_argument('--count',type=int,default=200);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--steps',type=int,default=50);p.add_argument('--guidance',type=float,default=2.5);p.add_argument('--seed',type=int,default=2026)
    main(p.parse_args())
