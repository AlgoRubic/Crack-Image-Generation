"""Read-only GitHub release-access check using existing noninteractive Git credentials.

Credential values stay in process memory and are never printed or written.
"""
import json
import os
import subprocess
import urllib.request
from urllib.parse import urlsplit


def authenticated_request(url,method='GET',data=None,extra_headers=None):
    parsed=urlsplit(url)
    if parsed.scheme!='https' or parsed.hostname not in ('api.github.com','uploads.github.com') or not parsed.path.startswith('/repos/AlgoRubic/Crack-Image-Generation'):
        raise ValueError('Request is outside the selected GitHub repository')
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never')
    result=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',
                          env=env,capture_output=True,text=True,timeout=30)
    if result.returncode:
        raise RuntimeError('Noninteractive GitHub credentials unavailable')
    values=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    token=values.get('password')
    if not token: raise RuntimeError('GitHub credential has no token')
    headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
             'X-GitHub-Api-Version':'2026-03-10','User-Agent':'Crack-Image-Generation-research-handoff'}
    headers.update(extra_headers or {})
    request=urllib.request.Request(url,headers=headers,method=method,data=data)
    with urllib.request.urlopen(request,timeout=600 if method=='POST' else 30) as response:
        return json.load(response)


if __name__=='__main__':
    try:
        repo=authenticated_request('https://api.github.com/repos/AlgoRubic/Crack-Image-Generation')
        releases=authenticated_request('https://api.github.com/repos/AlgoRubic/Crack-Image-Generation/releases')
        print(json.dumps(dict(repo=repo['full_name'],push_permission=repo.get('permissions',{}).get('push'),
                              releases=[dict(tag=r['tag_name'],url=r['html_url']) for r in releases]),indent=2))
    except Exception as error:
        # Do not print exception/request representations that could contain credentials.
        print(json.dumps(dict(access_check='failed',error_type=type(error).__name__)))
        raise SystemExit(1)
