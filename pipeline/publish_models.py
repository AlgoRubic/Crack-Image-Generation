"""Publish the verified historical-model bundle to the selected repository's Release.

Requires explicit --commit and --publish. Creates a draft first, uploads assets,
then publishes only after successful server-side size/digest verification.
Credentials stay in process memory; existing mismatched assets are never replaced.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import urlencode
from github_access_check import authenticated_request
from checkpoints import ROOT

API='https://api.github.com/repos/AlgoRubic/Crack-Image-Generation'


def main(commit,publish):
    if not publish: raise ValueError('Explicit --publish is required')
    if not re.fullmatch('[0-9a-f]{40}',commit): raise ValueError('Use full verified commit SHA')
    remote=subprocess.run(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.split()[0]
    if remote!=commit: raise ValueError('Selected source commit is not pushed main HEAD')
    out=Path('D:/Downloads/Crack-Image-Generation-sharing')
    bundle=out/'crack-generation-final-models-v1.zip'
    manifest=out/'model_bundle_manifest.json'
    expected=json.loads(manifest.read_text())
    h=hashlib.sha256()
    with bundle.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    if h.hexdigest()!=expected['sha256'] or bundle.stat().st_size!=expected['size_bytes']:
        raise ValueError('Model bundle checksum does not match verified manifest')
    releases=authenticated_request(API+'/releases')
    tag='historical-models-v1'
    release=next((r for r in releases if r['tag_name']==tag),None)
    body=('Six final historical pretrained checkpoints for Experiments 1–6. These models do not require retraining or raw datasets for inference. '
          'They are not new research_v4-clean models.\n\n'
          'Extract crack-generation-final-models-v1.zip into the repository root, preserving models/weights paths. '
          'Install dependencies using README.md, then run python pipeline/checkpoints.py verify and pipeline/generate_cracks.py.\n\n'
          'SHA256: '+expected['sha256']+'\n\n'
          'All six models passed one-image inference checks in an isolated source/weight copy without datasets, using the existing installed environment. '
          'Scientific quality and independent mechanism validation remain limited; see the research audit.')
    headers={'Content-Type':'application/json'}
    if release is None:
        release=authenticated_request(API+'/releases',method='POST',data=json.dumps(dict(tag_name=tag,target_commitish=commit,
             name='Historical pretrained models — Experiments 1–6',body=body,draft=True,prerelease=False)).encode(),extra_headers=headers)
    if release.get('target_commitish')!=commit:
        raise ValueError('Existing release targets a different source commit; preserve it')
    print(f'Draft release ready: {release["id"]}',flush=True)
    uploaded=[]
    for path,mime in ((bundle,'application/zip'),(manifest,'application/json')):
        existing=next((a for a in release['assets'] if a['name']==path.name),None)
        if existing is None:
            url=release['upload_url'].split('{')[0]+'?'+urlencode({'name':path.name})
            print(f'Uploading {path.name} ({path.stat().st_size} bytes)',flush=True)
            with path.open('rb') as stream:
                existing=authenticated_request(url,method='POST',data=stream,
                    extra_headers={'Content-Type':mime,'Content-Length':str(path.stat().st_size)})
        if existing['size']!=path.stat().st_size: raise ValueError('Uploaded asset size mismatch')
        if path==bundle and existing.get('digest') not in (None,'sha256:'+expected['sha256']):
            raise ValueError('Uploaded asset digest mismatch')
        uploaded.append(dict(name=path.name,url=existing['browser_download_url'],size=existing['size'],digest=existing.get('digest')))
    published=authenticated_request(API+'/releases/'+str(release['id']),method='PATCH',
                                   data=json.dumps(dict(draft=False,body=body)).encode(),extra_headers=headers)
    report=dict(release_url=published['html_url'],tag=tag,source_commit=commit,assets=uploaded,
                status='published' if not published['draft'] else 'draft')
    (out/'published_release.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--commit',required=True);parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    try: main(args.commit,args.publish)
    except Exception as exc:
        print(json.dumps(dict(status='failed',error_type=type(exc).__name__)),flush=True)
        raise SystemExit(1)
