"""Plot saved measurements only; never runs generation or changes metrics."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPS = ['unconditional_ddpm', 'conditional_ddpm', 'conditional_ddpm_128_augmented',
        'crack_type_conditional', 'structural_mechanism_ddpm', 'exp6_level_c_spatial_physics']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def bars(ax, names, vals, title, ylabel='', percent=False):
    ax.bar(range(len(names)), np.asarray(vals) * (100 if percent else 1), color='#237a89')
    ax.set_xticks(range(len(names)), [n.replace('_', '\n') for n in names], fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    if percent:
        ax.set_ylim(0, 110)


def main():
    for num, name in enumerate(EXPS, 1):
        folder = ROOT / 'experiments' / name
        source = folder / 'EVALUATION_METRICS.json'
        corrected = num == 6 and (folder / 'evaluation_corrected' / 'EVALUATION_METRICS.json').exists()
        if corrected:
            source = folder / 'evaluation_corrected' / 'EVALUATION_METRICS.json'
        data = json.loads(source.read_text(encoding='utf-8'))
        out = folder / 'plots'
        out.mkdir(exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
        ax = axes.ravel()
        log = next(iter(sorted((folder / 'logs').glob('*loss*.csv'))), None)
        points = []
        if log:
            for row in csv.DictReader(log.open(encoding='utf-8-sig')):
                try:
                    points.append((int(row['epoch']), float(row['mean_loss'])))
                except (ValueError, KeyError):
                    pass
        if points:
            ax[0].plot(*zip(*points), color='#237a89', marker='.', markersize=3)
            ax[0].set(xlabel='Logged epoch (local to this training phase)', ylabel='Mean noise MSE', title='Training loss (no held-out loss logged)')
            ax[0].grid(alpha=.2)
        else:
            ax[0].text(.5, .5, 'Training log unavailable', ha='center')
        classes = data.get('classes')
        if num == 4:
            classes = {k: v for k, v in data.items() if isinstance(v, dict) and 'sample_count' in v}
        if classes:
            keys = list(classes)
            rates = [next((classes[k][r] for r in ('target_adherence_rate', 'mechanism_adherence_rate',
                     'clean_surface_rate', 'orientation_alignment_accuracy', 'mesh_detection_rate') if r in classes[k]), 0) for k in keys]
            bars(ax[1], keys, rates, 'Heuristic target / mechanism adherence', 'Percent', True)
            counts = [classes[k].get('samples', classes[k].get('sample_count', 0)) for k in keys]
            bars(ax[2], keys, counts, 'Evaluated images per requested class', 'Images')
        else:
            prior = data['dataset_prior_comparison']
            bars(ax[1], ['Dataset label prior', 'Generated detector rate'],
                 [prior['real_sdnet2018_cracked_rate'], prior['generated_cracked_rate']],
                 'Crack prevalence (different label sources)', 'Percent', True)
            stats = data['surface_and_morphology_error_statistics']
            bars(ax[2], ['Luminance MAE', 'Luminance RMSE', 'Contrast MAE', 'Contrast RMSE'],
                 [stats[k] for k in ['luma_error_mae', 'luma_error_rmse', 'contrast_error_mae', 'contrast_error_rmse']],
                 'Deviation from reference summary statistics', 'Intensity units')
        cm = data.get('confusion_matrix', {})
        if 'matrix_counts' in cm:
            matrix = np.asarray(cm['matrix_counts'])
            labels = cm['class_order']
            im = ax[3].imshow(matrix, cmap='Blues')
            fig.colorbar(im, ax=ax[3], shrink=.7, label='Image count')
            ax[3].set_xticks(range(len(labels)), [s.replace('_', '\n') for s in labels], rotation=45, ha='right', fontsize=7)
            ax[3].set_yticks(range(len(labels)), [s.replace('_', ' ') for s in labels], fontsize=8)
            ax[3].set(xlabel='Heuristic assigned class', ylabel='Requested class',
                      title='Saved confusion counts' + (' (target-dependent rule)' if num == 6 and not corrected else ' (heuristic detector)'))
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax[3].text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=8,
                               color='white' if matrix[i, j] > matrix.max()/2 else 'black')
        elif classes:
            bars(ax[3], keys, [classes[k].get('mean_continuity_ratio', classes[k].get('mean_c_ratio', np.nan)) for k in keys],
                 'Mean continuity proxy (blank = unavailable)', 'Largest-component fraction')
        else:
            ax[3].axis('off')
            ax[3].text(.1, .7, 'Unconditional generation has no requested class.\nA classification confusion matrix is not applicable.\n\nThese are detector-based descriptive measurements.', va='top')
        fig.suptitle(f'Experiment {num} | Saved evaluation and training record', fontsize=17)
        fig.supxlabel('Heuristic scores are not independently validated classification accuracy. Sample sizes and protocols differ across experiments.', fontsize=9)
        for ext in ('png', 'svg'):
            fig.savefig(out / f'exp{num}_overview.{ext}', dpi=200)
        plt.close(fig)
        if num == 6:
            fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
            keys = [k for k in classes if k != 'non_cracked']
            for axis, metric, title, unit in zip(axes.ravel(),
                    ['trajectory_mae_px', 'angle_error_mae_deg', 'curvature_error_mae_deg', 'taper_error_mae'],
                    ['Trajectory MAE', 'Orientation MAE', 'Curvature MAE', 'Width taper MAE'],
                    ['Pixels', 'Degrees', 'Degrees', 'Ratio units']):
                bars(axis, keys, [classes[k][metric] for k in keys], title, unit)
            fig.suptitle('Experiment 6 | Commanded versus detected geometry', fontsize=17)
            fig.supxlabel('Saved heuristic geometry measurements; not a structural safety assessment.', fontsize=9)
            for ext in ('png', 'svg'):
                fig.savefig(out / ('exp6_geometry_errors.' + ext), dpi=200)
            plt.close(fig)
        manifest = {'metrics_source': str(source), 'training_log': str(log),
                    'metrics_modified': source.stat().st_mtime, 'training_rows_plotted': len(points),
                    'caveat': 'Saved heuristic evaluation, not independently validated accuracy.',
                    'corrected_target_independent_confusion': corrected}
        (out / 'plot_provenance.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'Experiment {num}: {out}', flush=True)


if __name__ == '__main__':
    main()
