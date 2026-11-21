"""Create an unlabelled review form and a separate key; never fabricate expert labels."""
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['non_cracked','flexural','flexure_shear','web_shear','torsion','corrosion_splitting','shrinkage_map']


def main():
    folder = ROOT / 'experiments/exp6_level_c_spatial_physics'
    out = folder / 'blind_review_v1'
    # A populated pack may already contain human annotations; never overwrite it.
    out.mkdir(exist_ok=False)
    images = out / 'images'
    images.mkdir()
    rng = random.Random(2026)
    selected = []
    for cname in CLASSES:
        pool = sorted((folder / 'results' / cname).glob('sample_*.png'))
        if len(pool) < 10:
            raise ValueError(f'Insufficient examples in {cname}')
        selected.extend((cname, p) for p in rng.sample(pool, 10))
    rng.shuffle(selected)
    keys, review = [], []
    cards = []
    for i,(cname,path) in enumerate(selected,1):
        rid = f'R{i:03d}'
        shutil.copyfile(path, images / (rid+'.png'))
        keys.append(dict(review_id=rid, requested_class=cname, source_path=str(path.relative_to(ROOT)),
                         image_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        review.append(dict(review_id=rid, reviewer_id='', crack_present='', morphology='',
                           mechanism='', confidence='', artifacts='', notes=''))
        cards.append(f'<article><h2>{rid}</h2><img src="images/{rid}.png" alt="Unlabelled review image {rid}" width="256" height="256"></article>')
    for name, rows in [('reviewer_form.csv',review),('PRIVATE_review_key.csv',keys)]:
        with (out/name).open('w',newline='',encoding='utf-8') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    instructions = ('Review visible crack presence and morphology first. Mark mechanism as indeterminate when loading, member geometry or reinforcement context is unavailable. '
                    'Use confidence low/medium/high. Do not infer physical causation solely from dark-line orientation. '
                    'Record reviewer identity and artifacts in reviewer_form.csv. Reviewers must not receive PRIVATE_review_key.csv. '
                    'Two independent reviewers and adjudication are required before agreement statistics or accuracy claims.')
    doc = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Blinded image review</title><style>body{font:18px/1.6 system-ui;max-width:1100px;margin:2rem auto;padding:1rem}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:1rem}article{border:1px solid #ccc;padding:1rem}img{max-width:100%;height:auto}</style></head><body><h1>Blinded crack-image review</h1><p>'+instructions+'</p><main>'+''.join(cards)+'</main></body></html>'
    (out/'review.html').write_text(doc,encoding='utf-8')
    (out/'protocol.json').write_text(json.dumps(dict(seed=2026, images=70, per_requested_class=10,
           status='awaiting_independent_reviewers', instructions=instructions,
           blinding='Randomized order and neutral IDs; separate key must remain with study coordinator',
           limitation='A balanced illustration/review sample; not a population estimate; image-only mechanism may be indeterminate.'),indent=2),encoding='utf-8')
    print(f'Prepared 70 unlabelled images and blank reviewer form: {out}')


if __name__ == '__main__':
    main()
