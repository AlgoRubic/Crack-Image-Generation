"""Resumable byte and decoded-RGB content audit for all manifest originals."""
import csv
import hashlib
import io
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from research import ROOT

DATASETS=['exp1_exp2_sdnet_64','exp3_augmented_128','exp4_level_a_morphology_256','exp5_exp6_level_bc_mechanisms_256']


def hash_original(original,previous):
    path=ROOT/original;stat=path.stat();prior=previous.get(original)
    if prior and int(prior['size_bytes'])==stat.st_size and int(prior['mtime_ns'])==stat.st_mtime_ns:
        return prior,False
    content=path.read_bytes()
    with Image.open(io.BytesIO(content)) as image:
        rgb=image.convert('RGB')
        shape=f'{rgb.width}x{rgb.height}:RGB:'.encode()
        pixelhash=hashlib.sha256(shape+rgb.tobytes()).hexdigest()
        width,height=rgb.size
    after=path.stat()
    if after.st_size!=stat.st_size or after.st_mtime_ns!=stat.st_mtime_ns:
        raise RuntimeError(f'Source changed during hashing: {original}')
    return dict(original_path=original,sha256=hashlib.sha256(content).hexdigest(),decoded_rgb_sha256=pixelhash,
                size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,width=width,height=height),True


def main():
    originals=set()
    for dataset in DATASETS:
        folder=ROOT/'datasets'/dataset
        inputs=[folder/'research_v1/provenance.csv'] if (folder/'research_v1/provenance.csv').exists() else [folder/'train.csv']+([folder/'validation.csv'] if (folder/'validation.csv').exists() else [])
        for file in inputs:
            with file.open(encoding='utf-8-sig') as stream:
                originals.update(r['original_path'] for r in csv.DictReader(stream))
    out=ROOT/'datasets/source_content_v1';out.mkdir(exist_ok=True)
    index=out/'index.csv'
    previous={}
    if index.exists():
        with index.open(encoding='utf-8') as stream:
            previous={r['original_path']:r for r in csv.DictReader(stream)}
    fields=['original_path','sha256','decoded_rgb_sha256','size_bytes','mtime_ns','width','height']
    results={}
    with index.open('a',encoding='utf-8',newline='') as stream, ThreadPoolExecutor(max_workers=4) as workers:
        writer=csv.DictWriter(stream,fieldnames=fields)
        if stream.tell()==0: writer.writeheader()
        for i,(record,changed) in enumerate(workers.map(lambda p:hash_original(p,previous),sorted(originals)),1):
            if changed:
                writer.writerow(record)
            results[record['original_path']]=record
            if i%1000==0:
                stream.flush();print(f'Hashed/verified {i}/{len(originals)} source originals',flush=True)
    duplicates=Counter(r['decoded_rgb_sha256'] for r in results.values())
    summary=dict(status='completed',original_files=len(results),decoded_unique_images=len(duplicates),
                 decoded_duplicate_groups=sum(n>1 for n in duplicates.values()),
                 duplicate_files_beyond_first=sum(n-1 for n in duplicates.values()),
                 index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
                 limitation='Exact byte and decoded RGB identity only; near-duplicates and structure/site identity remain unverified')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
