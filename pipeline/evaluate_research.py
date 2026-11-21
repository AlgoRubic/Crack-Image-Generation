"""Validate generated artifacts and evaluate new seven-class research samples.

Smoke metrics remain explicitly marked and cannot establish model quality.
"""
import argparse
import json
import subprocess
import sys
import csv
from pathlib import Path
from checkpoints import ROOT,sha
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from validate_detector import fissure_mask
from summarize_ablation import geometry

CLASSES=['non_cracked','flexural','flexure_shear','web_shear','torsion','corrosion_splitting','shrinkage_map']
MORPHOLOGY=['non_cracked','longitudinal','transverse','diagonal','alligator_map']


def validate_command(field,class_id):
    if field.shape!=(2,256,256) or not np.isfinite(field).all() or field.min()<0 or field.max()>1:
        raise ValueError('Invalid command field shape or range')
    if class_id and not (field[0]>.22).any():raise ValueError('Cracked command contains no reference trajectory')


def geometry_scores(mask,command):
    core=command[0]>.22;values=geometry(mask,command)
    precision=float(binary_dilation(core,iterations=4)[mask].mean()) if core.any() and mask.any() else 0. if core.any() else None
    recall=values['recall']
    f1=2*precision*recall/(precision+recall) if precision is not None and precision+recall else 0. if precision is not None else None
    return dict(trajectory_precision=precision,trajectory_recall=recall,trajectory_f1=f1,
                conditional_distance_mae_px=values['mae'],no_detection=values['no_detection'])


def verify_samples(folder,allow_smoke):
    manifest=folder/'sampling_manifest.json'
    data=json.loads(manifest.read_text())
    if data['status']!='completed' or data['experiment'] not in (4,5,6):
        raise ValueError('Expected completed research samples')
    classes=MORPHOLOGY if data['experiment']==4 else CLASSES
    if data['smoke_only'] and not allow_smoke:raise ValueError('Smoke samples require --allow-smoke')
    ids={r['class_id'] for r in data['images']}
    if ids!=set(range(len(classes))):raise ValueError('All experiment classes required')
    if len(data['images'])!=len(classes)*data['count_per_class']:raise ValueError('Unexpected image-record count')
    expected={}
    for item in data['images']:
        if item['class_name']!=classes[item['class_id']] or not 0<=item['index']<data['count_per_class']:
            raise ValueError('Class/index metadata mismatch')
        pairs=[('path','sha256')]+([('command_path','command_sha256')] if data['experiment']!=4 else [])
        for path_key,hash_key in pairs:
            path=(folder/item[path_key]).resolve()
            if not path.is_relative_to(folder.resolve()):raise ValueError('Sample path escapes run')
            if not path.is_file() or sha(path)!=item[hash_key]:raise ValueError('Missing/changed generated image or command')
        with Image.open(folder/item['path']) as image:
            if image.size!=(256,256) or image.mode!='RGB':raise ValueError('Invalid generated image format')
        if data['experiment']!=4:validate_command(np.load(folder/item['command_path']),item['class_id'])
        expected.setdefault(item['class_name'],set()).add(item['path'])
    for name,paths in expected.items():
        actual={p.relative_to(folder).as_posix() for p in (folder/name).glob('sample_*.png')}
        if paths!=actual or len(paths)!=data['count_per_class']:raise ValueError('Sample count/membership mismatch')
    return data,sha(manifest)


def main(args):
    folder=Path(args.samples).resolve();data,fingerprint=verify_samples(folder,args.allow_smoke)
    if Path(args.evaluation_id).name!=args.evaluation_id or args.evaluation_id in ('.','..'):raise ValueError('Invalid evaluation ID')
    out=folder/args.evaluation_id
    out.mkdir(exist_ok=False)
    if data['experiment']==4:
        evaluate_morphology(folder,data,fingerprint,out)
        return
    command=[sys.executable,str(ROOT/'src/evaluate_exp6_spatial_physics_256.py'),
             '--samples-dir',str(folder),'--output-dir',str(out),'--figure-note',
             'SMOKE PIPELINE CHECK — NOT A MODEL QUALITY RESULT' if data['smoke_only'] else 'Research checkpoint; morphology heuristics require independent validation']
    subprocess.run(command,cwd=ROOT,check=True)
    path=out/'EVALUATION_METRICS.json';metrics=json.loads(path.read_text())
    metrics['protocol'].update(sampling_manifest_sha256=fingerprint,weights=data['weights'],
        checkpoint_sha256=data['checkpoint_sha256'],smoke_only=data['smoke_only'],
        trained_experiment=data['experiment'],conditioning=data['conditioning'],
        note='Seven-class morphology heuristics; not independent physical-mechanism accuracy',
        legacy_distance_warning='Legacy trajectory fields may contain 50px missing-detection penalties; use balanced_geometry for conditional distances')
    balanced=[]
    for item in data['images']:
        with Image.open(folder/item['path']) as image:mask=fissure_mask(np.array(image.convert('RGB')))
        command=np.load(folder/item['command_path'])
        balanced.append(dict(class_name=item['class_name'],index=item['index'],**geometry_scores(mask,command)))
    with (out/'balanced_geometry.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(balanced[0]));writer.writeheader();writer.writerows(balanced)
    cracked=[r for r in balanced if r['trajectory_recall'] is not None]
    distances=[r['conditional_distance_mae_px'] for r in cracked if r['conditional_distance_mae_px'] is not None]
    metrics['balanced_geometry']=dict(tolerance='4 binary-dilation iterations at native 256px',
        mean_trajectory_precision=float(np.mean([r['trajectory_precision'] for r in cracked])),
        mean_trajectory_recall=float(np.mean([r['trajectory_recall'] for r in cracked])),
        mean_trajectory_f1=float(np.mean([r['trajectory_f1'] for r in cracked])),
        no_detection_count=sum(r['no_detection'] for r in cracked),
        conditional_distance_mae_px=float(np.mean(distances)) if distances else None,
        caveat='Symmetric trajectory proxy metrics, not human-mask segmentation quality or mechanism accuracy')
    path.write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(f'Provenance-linked evaluation saved: {out}; smoke_only={data["smoke_only"]}')


def evaluate_morphology(folder,data,fingerprint,out):
    from evaluate_crack_type_256 import analyze_crack_morphology
    import matplotlib.pyplot as plt
    matrix=np.zeros((5,5),dtype=int);rows=[]
    for item in data['images']:
        with Image.open(folder/item['path']) as image:mask=fissure_mask(np.array(image.convert('RGB')))
        stat=analyze_crack_morphology(mask)
        predicted=stat['predicted_class'] if stat['has_crack'] else 'non_cracked'
        matrix[item['class_id'],MORPHOLOGY.index(predicted)]+=1
        rows.append(dict(requested_class=item['class_name'],predicted_class=predicted,path=item['path'],
                         has_crack=bool(stat['has_crack']),continuity_ratio=float(stat['c_ratio']),max_span_px=float(stat['max_length'])))
    report=dict(experiment=4,smoke_only=data['smoke_only'],sampling_manifest_sha256=fingerprint,
                checkpoint_sha256=data['checkpoint_sha256'],weights=data['weights'],
                class_order=MORPHOLOGY,confusion_counts=matrix.tolist(),
                requested_class_agreement=float(np.trace(matrix)/matrix.sum()),
                detector='Fixed shared local-darkness segmentation (23px,24.5,2x2 opening) with historical morphology rules',
                caveat='Descriptive morphology agreement; detector thresholds and orientation labels require independent validation')
    (out/'EVALUATION_METRICS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (out/'PER_SAMPLE_ERROR_LOG.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,ax=plt.subplots(figsize=(8,7),layout='constrained');image=ax.imshow(matrix,cmap='Blues');fig.colorbar(image,ax=ax)
    ax.set_xticks(range(5),MORPHOLOGY,rotation=35,ha='right');ax.set_yticks(range(5),MORPHOLOGY)
    ax.set(xlabel='Heuristic prediction',ylabel='Requested class',title='SMOKE PIPELINE CHECK — NO QUALITY CLAIM' if data['smoke_only'] else 'Experiment 4 | descriptive morphology agreement')
    for i in range(5):
        for j in range(5):ax.text(j,i,str(matrix[i,j]),ha='center',va='center')
    fig.savefig(out/'morphology_confusion.png',dpi=200);plt.close(fig)
    print(f'Morphology evaluation saved: {out}; smoke_only={data["smoke_only"]}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--samples',required=True);p.add_argument('--allow-smoke',action='store_true')
    p.add_argument('--evaluation-id',default='evaluation')
    main(p.parse_args())
