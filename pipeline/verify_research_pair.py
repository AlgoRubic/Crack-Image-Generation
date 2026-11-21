"""Verify class-only/spatial research samples use matched protocols, commands and noise."""
import argparse
import json
from pathlib import Path
from checkpoints import ROOT,sha


def main(control,spatial):
    a=json.loads((control/'sampling_manifest.json').read_text());b=json.loads((spatial/'sampling_manifest.json').read_text())
    if a['conditioning']!='class_only' or b['conditioning']!='spatial' or a['experiment']!=6 or b['experiment']!=6:
        raise ValueError('Expected matched five-channel Experiment 6 controls')
    for key in ('status','smoke_only','weights','count_per_class','seed','steps','guidance','sampler_sha256','postprocessing'):
        if a[key]!=b[key]:raise ValueError(f'Unmatched sampling protocol: {key}')
    if a['status']!='completed':raise ValueError('Sampling is incomplete')
    for key in ('seed','batch_size','lr','ema_decay','split_version','manifests','script_sha256'):
        if a['training_config'][key]!=b['training_config'][key]:raise ValueError(f'Unmatched training protocol: {key}')
    first={(r['class_id'],r['index']):r for r in a['images']}
    second={(r['class_id'],r['index']):r for r in b['images']}
    if first.keys()!=second.keys():raise ValueError('Sample pairs differ')
    for key in first:
        for fingerprint in ('initial_noise_sha256','command_sha256'):
            if first[key][fingerprint]!=second[key][fingerprint]:raise ValueError(f'Unmatched sample {key}: {fingerprint}')
    report=dict(passed=True,paired_images=len(first),smoke_only=a['smoke_only'],
                control_manifest_sha256=sha(control/'sampling_manifest.json'),spatial_manifest_sha256=sha(spatial/'sampling_manifest.json'),
                caveat='Protocol pairing verification only; no scientific quality or mechanism-validity claim')
    out=ROOT/'experiments/research_pair_verification.json';out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('control',type=Path);p.add_argument('spatial',type=Path)
    args=p.parse_args();main(args.control,args.spatial)
