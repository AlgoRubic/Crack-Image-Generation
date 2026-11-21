"""Package only final historical inference weights; preserve repository-relative paths."""
import hashlib
import json
import zipfile
from pathlib import Path
from checkpoints import ROOT,MANIFEST,verify


def main():
    verify()
    out=Path('D:/Downloads/Crack-Image-Generation-sharing')
    out.mkdir(exist_ok=True)
    bundle=out/'crack-generation-final-models-v1.zip'
    if bundle.exists():
        raise FileExistsError('Preserve the existing versioned bundle; select a new version for changed weights')
    manifest=json.loads(MANIFEST.read_text())
    with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        archive.write(MANIFEST,'models/checkpoint_manifest.json')
        for entry in manifest['checkpoints']:
            archive.write(ROOT/entry['path'],entry['path'])
    with zipfile.ZipFile(bundle) as archive:
        failure=archive.testzip()
        if failure: raise RuntimeError(f'Archive integrity failure: {failure}')
    h=hashlib.sha256()
    with bundle.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    report=dict(file=bundle.name,size_bytes=bundle.stat().st_size,sha256=h.hexdigest(),
                contains='Six final historical inference checkpoints and portable checksum manifest',
                extraction='Extract into repository root, preserving models/weights paths',
                datasets_required_for_inference=False,
                lineage='Historical models, not newly trained research_v4 models')
    (out/'model_bundle_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
