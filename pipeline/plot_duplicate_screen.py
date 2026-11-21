"""Create an aggregate screening plot and local-only source review contact sheet."""
import csv
import json
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps

from research import ROOT


def main():
    folder = ROOT / 'datasets/near_duplicate_screen_v1'
    summary = json.loads((folder / 'research_v4_summary.json').read_text())
    with (folder / 'research_v4_candidates.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    current = json.loads((folder / 'research_v5_summary.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')
    distances = Counter(int(r['phash_distance']) for r in rows)
    axes[0].bar(range(4), [distances[i] for i in range(4)], color='#277a78')
    axes[0].set(xlabel='pHash Hamming distance', ylabel='Candidate pairs', xticks=range(4))
    pairs = summary['split_pair_counts']
    axes[1].barh(list(pairs), list(pairs.values()), color='#ca663b')
    axes[1].set(xlabel='Candidate pairs')
    axes[2].bar(['v4', 'v5'], [len(rows), current['cross_split_candidate_pairs']], color=['#ca663b', '#277a78'])
    axes[2].set(ylabel='Cross-split candidates', title='Same declared search after grouping')
    fig.suptitle(f'Cross-split source similarity screening: {len(rows):,} unreviewed pairs', fontsize=13)
    fig.supxlabel('pHash ≤3 AND dHash ≤6; candidates are not confirmed duplicates. Crops/rotations may be missed.', fontsize=9)
    for suffix in ('png', 'svg'):
        fig.savefig(folder / f'research_v4_screen.{suffix}', dpi=180)
    plt.close(fig)
    # Raw source images remain local; do not redistribute these sheets with the website.
    selected = rows[:12]
    if selected:
        sheet = Image.new('RGB', (1000, 230 * len(selected) + 50), '#f5f3ee')
        draw = ImageDraw.Draw(sheet)
        draw.text((15, 12), 'UNREVIEWED similarity candidates — first 12 by hash distance; not a random sample', fill='black')
        for i, row in enumerate(selected):
            top = 50 + 230 * i
            draw.text((15, top), f'Pair {i+1}: pHash={row["phash_distance"]}; dHash={row["dhash_distance"]}', fill='black')
            for column, side in enumerate(('a', 'b')):
                x = 15 + column * 495
                with Image.open(ROOT / row[f'path_{side}']) as image:
                    thumb = ImageOps.contain(image.convert('RGB'), (475, 160))
                sheet.paste(thumb, (x, top + 20))
                label = row[f'path_{side}'].split('/')[-1]
                draw.text((x, top + 185), f'{row[f"split_{side}"]}: {label}', fill='black')
        sheet.save(folder / 'LOCAL_ONLY_review_first12.jpg', quality=90)
    print(json.dumps(dict(candidate_pairs=len(rows), plot=str(folder / 'research_v4_screen.png'),
                          review_sheet='Local only; raw-source redistribution not authorized'), indent=2))


if __name__ == '__main__':
    main()
