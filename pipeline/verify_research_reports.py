"""Check six revised reports against saved evidence, original archives and sample logs."""
import csv
import hashlib
import json
import re
from pathlib import Path
from build_research_reports import ROOT,NAMES


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    revision=json.loads((ROOT/'experiments/report_revision_audit.json').read_text(encoding='utf-8'))
    errors=[];checks=[];line_endings=[]
    for record in revision['reports']:
        n=record['experiment'];folder=ROOT/'experiments'/NAMES[n-1]
        path=ROOT/record['report'];text=path.read_text(encoding='utf-8')
        evidence=json.loads((folder/'report_evidence.json').read_text(encoding='utf-8'))
        if digest(path)!=record['sha256']:errors.append(f'Exp{n}: revised report changed after build')
        for section in ('Abstract',)+tuple(f'{i}.' for i in range(1,15)):
            if f'## {section}' not in text:errors.append(f'Exp{n}: missing section {section}')
        if text.count('$$')%2 or text.count('```')%2:errors.append(f'Exp{n}: unbalanced math or code delimiter')
        if 'file:///' in text:errors.append(f'Exp{n}: nonportable file URL')
        for relative,expected in evidence['input_sha256'].items():
            source=ROOT/relative
            if digest(source)!=expected:
                normalized=hashlib.sha256(source.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                if normalized!=evidence['input_lf_sha256'].get(relative):errors.append(f'Exp{n}: source changed: {relative}')
                else:line_endings.append(relative)
        for source in evidence['archived_sources']:
            if digest(ROOT/source['archive'])!=source['sha256']:errors.append(f'Exp{n}: original archive changed')
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            if target.startswith(('https://','http://')):continue
            if not (path.parent/target).is_file():errors.append(f'Exp{n}: missing report artifact {target}')
        source=folder/('evaluation_corrected/EVALUATION_METRICS.json' if n==6 else 'EVALUATION_METRICS.json')
        metrics=json.loads(source.read_text(encoding='utf-8'))
        sample=folder/('evaluation_corrected/PER_SAMPLE_ERROR_LOG.csv' if n==6 else 'PER_SAMPLE_ERROR_LOG.csv')
        rows=None
        if sample.exists():
            with sample.open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
            expected=metrics.get('total_evaluated_samples',1400 if n==6 else None)
            if expected is not None and len(rows)!=expected:errors.append(f'Exp{n}: per-sample denominator mismatch')
            if n in (2,3,6):
                expected=metrics['confusion_matrix']['matrix_counts'];actual=[[0]*len(expected) for _ in expected]
                key='class_id' if n==6 else 'commanded_class_id'
                for row in rows:actual[int(row[key])][int(row['predicted_class_id'])]+=1
                if actual!=expected:errors.append(f'Exp{n}: saved confusion matrix disagrees with per-sample log')
        if n==6 and ('1,230 / 1,400 (87.86%)' not in text or 'superseded' not in text):errors.append('Exp6: corrected agreement/supersession absent')
        checks.append(dict(experiment=n,words=len(text.split()),archive_files=len(evidence['archived_sources']),input_files=len(evidence['input_sha256']),per_sample_rows=len(rows) if rows is not None else None))
    result=dict(passed=not errors,reports_checked=len(checks),reports=checks,errors=errors,
                text_inputs_with_line_ending_conversion=sorted(set(line_endings)),
                scope='Evidence consistency, source/archive integrity, structure, links and sample-log aggregation; not independent scientific validation')
    (ROOT/'experiments/report_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
