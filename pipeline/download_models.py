"""Download, verify and install historical pretrained weights without overwriting any model."""
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path,PurePosixPath
from checkpoints import ROOT,sha


def install_archive(bundle,root,manifest):
    entries=manifest['checkpoints']
    for entry in entries:
        path=PurePosixPath(entry['path'])
        if path.is_absolute() or '..' in path.parts or path.parts[:2]!=('models','weights') or path.suffix!='.pt':
            raise ValueError('Unexpected model destination')
        target=root/entry['path']
        if target.exists() and (target.stat().st_size!=entry['size_bytes'] or sha(target)!=entry['sha256']):
            raise ValueError(f'Existing weight differs; preserve it and resolve manually: {target}')
    cache=root/'.cache/model-downloads';cache.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(bundle) as archive, tempfile.TemporaryDirectory(dir=cache) as temp:
        expected={e['path'] for e in entries}|{'models/checkpoint_manifest.json'}
        if set(archive.namelist())!=expected or len(archive.namelist())!=len(expected):
            raise ValueError('Unexpected or duplicate archive members')
        if json.loads(archive.read('models/checkpoint_manifest.json'))!=manifest:
            raise ValueError('Archive model manifest differs from repository manifest')
        # Verify every staged checkpoint before exposing any new model path.
        staged=[]
        for i,entry in enumerate(entries):
            path=Path(temp)/str(i)
            with archive.open(entry['path']) as source,path.open('xb') as dest: shutil.copyfileobj(source,dest)
            if path.stat().st_size!=entry['size_bytes'] or sha(path)!=entry['sha256']:
                raise ValueError('Extracted weight checksum mismatch')
            staged.append((entry,path))
        for entry,path in staged:
            target=root/entry['path'];target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if sha(target)!=entry['sha256']: raise ValueError('Model changed during installation')
                continue
            # Same-filesystem hard link exposes verified bytes atomically and refuses overwrite.
            os.link(path,target)


def main():
    manifest=json.loads((ROOT/'models/checkpoint_manifest.json').read_text())
    missing=[]
    for entry in manifest['checkpoints']:
        path=ROOT/entry['path']
        if path.exists():
            if path.stat().st_size!=entry['size_bytes'] or sha(path)!=entry['sha256']:
                raise ValueError(f'Existing weight differs; will not overwrite: {path}')
        else: missing.append(entry)
    if not missing:
        print('All six historical models are present and checksum-verified. No download needed.');return
    release=json.loads((ROOT/'models/release_manifest.json').read_text())
    cache=ROOT/'.cache/model-downloads';cache.mkdir(parents=True,exist_ok=True)
    bundle=cache/'historical-models-v1.zip'
    if not bundle.exists() or sha(bundle)!=release['bundle_sha256']:
        part=cache/'historical-models-v1.zip.part'
        request=urllib.request.Request(release['bundle_url'],headers={'User-Agent':'Crack-Image-Generation-model-download'})
        total=0;last=0
        with urllib.request.urlopen(request,timeout=60) as response,part.open('wb') as stream:
            while True:
                block=response.read(2*1024*1024)
                if not block:break
                stream.write(block);total+=len(block)
                if total>release['bundle_size_bytes']:raise ValueError('Download exceeds expected size')
                if total-last>=32*1024*1024:
                    print(f'Downloaded {total/2**20:.0f} MiB',flush=True);last=total
        if total!=release['bundle_size_bytes'] or sha(part)!=release['bundle_sha256']:
            raise ValueError('Downloaded bundle failed size/checksum validation')
        # Cache files are task-owned; model paths remain immutable.
        part.replace(bundle)
    install_archive(bundle,ROOT,manifest)
    print('Installed and verified six historical pretrained models. No datasets or retraining required.')


if __name__=='__main__':main()
