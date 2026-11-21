"""Generate one sample per experiment in an isolated source/weight copy without datasets."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from PIL import Image
from checkpoints import ROOT

NAMES=['unconditional_ddpm','conditional_ddpm','conditional_ddpm_128_augmented','crack_type_conditional','structural_mechanism_ddpm','exp6_level_c_spatial_physics']


def main(run_id):
    if Path(run_id).name!=run_id or run_id in ('.','..'): raise ValueError('Invalid run ID')
    destination=Path('D:/Downloads/Crack-Image-Generation-sharing')/run_id
    destination.mkdir(exist_ok=False)
    for folder in ('pipeline','src'):
        for path in (ROOT/folder).rglob('*.py'):
            target=destination/path.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    bundle=destination.parent/'crack-generation-final-models-v1.zip'
    with zipfile.ZipFile(bundle) as archive:
        for name in archive.namelist():
            path=Path(name)
            if path.is_absolute() or '..' in path.parts or path.parts[0]!='models':
                raise ValueError(f'Unexpected model archive member: {name}')
        archive.extractall(destination)
    if (destination/'data').exists() or (destination/'datasets').exists():
        raise ValueError('Isolated inference copy must not have raw/processed datasets')
    checks=[]
    for n in range(1,7):
        output=destination/'outputs'/f'exp{n}'
        category='all' if n==1 else 'cracked' if n<4 else 'non_cracked' if n==4 else 'flexural'
        command=[sys.executable,str(destination/'pipeline/generate_cracks.py'),'--experiment',str(n),
                 '--crack-type',category,'--count','1','--seed','2026','--output-dir',str(output)]
        started=time.perf_counter()
        result=subprocess.run(command,cwd=destination,capture_output=True,text=True,timeout=600)
        images=[p for p in output.rglob('*.png') if 'grid' not in p.name]
        expected=64 if n<3 else 128 if n==3 else 256
        valid=len(images)==1
        for path in images:
            with Image.open(path) as im:
                valid &= im.size==(expected,expected) and im.mode=='RGB'
        record=dict(experiment=n,passed=result.returncode==0 and valid,returncode=result.returncode,
                    output_images=len(images),expected_resolution=expected,seconds=time.perf_counter()-started,
                    image_sha256=hashlib.sha256(images[0].read_bytes()).hexdigest() if images else None,
                    stderr=result.stderr if result.returncode else '')
        checks.append(record);print(json.dumps(record),flush=True)
        evidence=dict(isolated_source_root=str(destination),datasets_present=False,checks=checks,
                      environment='Existing installed environment reused; not a fresh dependency installation',
                      scope='Pretrained inference smoke checks; no training or scientific quality claim')
        (ROOT/'experiments/pretrained_package_verification.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
        if not record['passed']: raise RuntimeError(f'Inference package failed for Experiment {n}')
    print('All six pretrained inference checks passed without datasets.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True)
    main(parser.parse_args().run_id)
