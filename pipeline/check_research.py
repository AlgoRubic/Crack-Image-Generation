"""Run bounded infrastructure checks and separately report scientific readiness."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CHECKS=['pipeline/test_research.py','pipeline/test_acquisition_splits.py','pipeline/test_source_hashes.py','pipeline/test_near_duplicate_screen.py','pipeline/test_rc10519_audit.py','pipeline/test_quality_view.py','pipeline/test_detector.py','pipeline/test_ablation.py','pipeline/test_model_install.py','pipeline/test_research_handoff.py','pipeline/test_fidelity_protocol.py','pipeline/test_reproduction.py','src/test_evaluation_independence.py','pipeline/verify_research_reports.py','website/verify_site.py']


def main():
    checks=[]
    for script in CHECKS:
        result=subprocess.run([sys.executable,script],cwd=ROOT,capture_output=True,text=True)
        checks.append(dict(script=script,passed=result.returncode==0,exit_code=result.returncode,
                           stdout=result.stdout,stderr=result.stderr))
    output=dict(created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,
                infrastructure_checks_passed=all(c['passed'] for c in checks),
                publication_ready=False,
                outstanding=['Clean full training and multi-seed evaluation',
                             'Independent expert review of mechanism labels',
                             'Generated-domain detector validation',
                             'Controlled spatial ablations and external validation'],
                limitation='Passing these infrastructure checks does not establish scientific validity.')
    path=ROOT/'experiments/research_quality_checks.json'
    path.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))
    if not output['infrastructure_checks_passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
