"""Build current evidence addenda from project artifacts; retain original reports as archives."""
import argparse
import copy
import html
import json
import re
import shutil
from pathlib import Path
import build_experiment_pages as base

SITE = Path(__file__).resolve().parent
NAMES = ['unconditional_ddpm', 'conditional_ddpm', 'conditional_ddpm_128_augmented',
         'crack_type_conditional', 'structural_mechanism_ddpm', 'exp6_level_c_spatial_physics']
esc = html.escape


def table(headers, rows):
    return '<table class="research-table"><thead><tr>' + ''.join(f'<th>{esc(str(h))}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{esc(str(v))}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table>'


def supporting_evidence(project, assets):
    exp=project/'experiments/exp6_level_c_spatial_physics'
    sources={
        'source_content_summary.json':project/'datasets/source_content_v1/summary.json',
        'detector_baseline.json':exp/'detector_validation/summary.json',
        'detector_calibration.json':exp/'detector_calibration_v1/summary.json',
        'ablation_audit.json':exp/'ablations/paired-seven-classes-three-seeds-v2/audited_summary.json',
        'detector_metrics.png':exp/'detector_validation/detector_metrics.png',
        'detector_errors.png':exp/'detector_validation/examples.png',
        'detector_calibration.png':exp/'detector_calibration_v1/calibration_comparison.png',
        'ablation_comparison.png':exp/'ablations/paired-seven-classes-three-seeds-v2/audited_comparison.png',
        'references.bib':project/'REFERENCES.bib',
        'references.json':project/'pipeline/research_references.json',
    }
    for name,source in sources.items():
        shutil.copyfile(source,assets/name)
    baseline=json.loads(sources['detector_baseline.json'].read_text())
    calibrated=json.loads(sources['detector_calibration.json'].read_text())
    ablation=json.loads(sources['ablation_audit.json'].read_text())
    rows=[[k.capitalize(),f'{baseline["metrics"][k]*100:.1f}%',f'{calibrated["test_selected"][k]*100:.1f}%'] for k in ('precision','recall','iou','dice')]
    section='<section id="detector-validation" class="research-section"><p class="eyebrow">EVALUATOR CHECK</p><h2>How reliable is the crack detector?</h2><p>Pixel-mask comparison on 51 DeepCrack originals (102 native tiles). Calibration searched 24 settings using 46 validation originals only. Both studies used the archived research_v1 selection, not the newer research_v4 split. Test results had already been inspected, so the calibrated comparison is exploratory.</p>'+table(['Pixel metric','Fixed detector','Validation-selected detector'],rows)
    section+='<p>Exact mask correspondence measures visible cracks, not structural failure mechanism. Real-image mask performance does not establish generated-image detector validity. Original-image cluster-bootstrap intervals and the full protocol are retained in the metric records.</p><img class="research-plot" src="assets/research/detector_calibration.png" alt="Exploratory detector calibration comparison"><details><summary>Inspect representative detector errors</summary><p>Left: original tile. Right: red false positives, green missed mask pixels, cyan overlap. Examples are the three lowest and three highest first-tile IoU values, not a random sample.</p><img class="research-plot" src="assets/research/detector_errors.png" alt="Crack segmentation errors against human pixel masks"></details><p><a href="assets/research/detector_baseline.json">Baseline record</a> · <a href="assets/research/detector_calibration.json">Calibration record</a></p></section>'
    ar=[]
    for detector in ('fixed','calibrated'):
        for mode in ('correct','zero','shuffled'):
            record=ablation['summary'][detector][mode]
            ar.append([detector,mode,record['n'],f'{record["mean_recall"]*100:.1f}%',record['no_detection'],
                       f'{record["conditional_mae_px"]:.2f}' if record['conditional_mae_px'] is not None else 'No detected crack; distance undefined'])
    section+='<section id="spatial-ablation" class="research-section"><p class="eyebrow">PAIRED CONTROL STUDY</p><h2>Does the spatial field change the output?</h2><p>126 images across seven classes, three seeds, and correct / zero / shuffled fields. Each cracked class contributes two commands per seed. Outputs are scored against the original correct command. The audit verified identical initial RNG states within each pair and byte-identical non-cracked controls.</p>'+table(['Detector','Field','Cracked images','Trajectory recall','Empty detections','Conditional distance MAE (px)'],ar)
    section+='<img class="research-plot" src="assets/research/ablation_comparison.png" alt="Spatial-field sensitivity by crack class and detector"><p>This is a small historical-checkpoint sensitivity pilot. The zero-field/requested-class pairing may be outside the training distribution, and shuffled-field mismatch is expected to reduce correspondence to the original command. These results do not establish physical causation or replace a separately trained class-only baseline.</p><p>Missing cracks are counted explicitly. Distance means exclude empty detections; they are not assigned the legacy fixed 50-pixel penalty.</p><p><a href="assets/research/ablation_audit.json">Inspect paired audit and limitations</a></p>'
    for cname in ('non_cracked','flexural','flexure_shear','web_shear','torsion','corrosion_splitting','shrinkage_map'):
        section+=f'<details><summary>{esc(cname.replace("_"," "))}: paired examples</summary><p>Seed 2026, both sampled commands, shown in the same order. This fixed-seed display is illustrative.</p><div class="paired-gallery">'
        for mode in ('correct','zero','shuffled'):
            target=f'ablation_{cname}_{mode}.png'
            shutil.copyfile(exp/f'ablations/paired-seven-classes-three-seeds-v2/{cname}_seed2026/{mode}_grid.png',assets/target)
            section+=f'<figure><img src="assets/research/{target}" alt="{esc(cname)} with {mode} spatial fields" loading="lazy"><figcaption>{mode} field</figcaption></figure>'
        section+='</div></details>'
    return section+'</section>'


def build(project):
    base.PROJECT = project
    assets = SITE / 'assets/research'
    assets.mkdir(parents=True, exist_ok=True)
    supporting=supporting_evidence(project,assets)
    archive_images = {
        **{name + '_256_grid.png': 'crack_type_conditional/generated_samples/' + name + '_256_grid.png'
           for name in ('longitudinal', 'transverse', 'diagonal', 'alligator_map')},
        'dataset_verification_grid.png': 'structural_mechanism_ddpm/dataset_verification_grid.png',
        'physics_trajectories_verification.png': 'exp6_level_c_spatial_physics/physics_trajectories_verification.png',
        'exp6_error_analysis_plots.png': 'exp6_level_c_spatial_physics/exp6_error_analysis_plots.png',
    }
    for target, source in archive_images.items():
        shutil.copyfile(project / 'experiments' / source, SITE / 'assets/images' / target)
    rows, panels = [], []
    for n, name in enumerate(NAMES, 1):
        folder = project / 'experiments' / name
        source = folder / ('evaluation_corrected/EVALUATION_METRICS.json' if n == 6 else 'EVALUATION_METRICS.json')
        data = json.loads(source.read_text(encoding='utf-8'))
        shutil.copyfile(source, assets / f'exp{n}_metrics.json')
        shutil.copyfile(folder / 'plots' / f'exp{n}_overview.png', assets / f'exp{n}_overview.png')
        cm = data.get('confusion_matrix', {})
        classes = data.get('classes', {})
        if n == 4:
            classes = {k:v for k,v in data.items() if isinstance(v, dict) and 'sample_count' in v}
        count = data.get('total_evaluated_samples') or sum(c.get('samples', c.get('sample_count', 0)) for c in classes.values())
        agreement = f"{cm['overall_accuracy'] * 100:.2f}%" if cm else 'Not reported'
        if n == 1:
            text = f"{count:,} images evaluated. The morphology detector flags {data['dataset_prior_comparison']['generated_cracked_rate']*100:.1f}% as cracked. This is detector prevalence, not a ground-truth accuracy score."
        elif n in (2,3,6):
            text = f'{count:,} generated images evaluated. Requested-class agreement with the morphology heuristic: {agreement}. The detector has not been independently calibrated for this generated-image domain.'
        elif n == 4:
            text = '80 generated images evaluated (16 per class). The heuristic clean-surface rate is 0%; longitudinal, transverse and diagonal alignment rates are 0%. These findings require detector calibration and visual review; continuity improvement is not established by these measurements.'
        else:
            text = '112 generated images evaluated (16 per class). Heuristic adherence varies by class; flexure-shear is 50% and web-shear is 62.5%. These small-sample proxy results do not establish structural failure mechanisms.'
        if n == 6:
            text += ' The former 99.29% score used requested-class-dependent assignments and is superseded by this corrected 87.86% result.'
        class_rows = []
        for cname, record in classes.items():
            rate = next((record[k] for k in ['target_adherence_rate','mechanism_adherence_rate','clean_surface_rate','orientation_alignment_accuracy','mesh_detection_rate'] if k in record), None)
            class_rows.append([cname.replace('_',' '), record.get('samples', record.get('sample_count')), f'{rate*100:.2f}%' if rate is not None else 'Unavailable'])
        panel = '<section class="research-section"><p class="eyebrow">CURRENT EVALUATION ADDENDUM</p><h2>Measured results and their limits</h2><p>' + esc(text) + '</p>'
        panel += table(['Requested class','Images','Heuristic adherence / alignment'], class_rows) if classes else ''
        panel += f'<p><a href="assets/research/exp{n}_metrics.json">Inspect the metric record</a> · <a href="research.html">Research protocol and readiness</a></p><img class="research-plot" src="assets/research/exp{n}_overview.png" alt="Experiment {n} training and evaluation overview">'
        panel += '<p>Fidelity scores in historical reports used differing reference samples or implementations. They are not a controlled cross-experiment leaderboard. New clean training and independent validation remain outstanding.</p></section>'
        if n in (3,5,6):
            panel+='<section class="research-section"><h2>Dataset source and annotation audit</h2><p>The RC source audit found annotation geometry issues. The prospective Experiment 5/6 pipeline uses a conservative qualified subset; historical results are preserved. See the source audit for counts, attribution and limitations.</p><a href="research.html#source-quality">Inspect source quality and qualification</a></section>'
        if n==6:
            panel+=supporting
        d = copy.deepcopy(base.common[n])
        d.update(report_kind='md', status='HISTORICAL RUN · EVALUATION AUDITED', results=text,
                 observations='See the current evaluation addendum for per-class measurements and plots. Generated examples illustrate model output; they are not independently annotated validation data.',
                 conclusion='This experiment provides evidence about image synthesis and heuristic control. Claims about physical mechanism, generalization and downstream usefulness require independent labels and controlled held-out evaluation.',
                 errors='Geometry-based labels and detectors are proxies. Historical checkpoint exposure, small or unequal sample sizes, differing fidelity implementations and domain shifts limit interpretation. See the research audit for the corrected protocol.')
        if n in (2, 3):
            previous = (SITE / f'experiment-{n}.html').read_text(encoding='utf-8')
            match = re.search(r'<div class="report-content">(.*?)</div></details>', previous, re.S)
            if not match:
                raise ValueError(f'Missing archival technical record for Experiment {n}')
            d.update(report_kind='archive', archive_body=match.group(1))
        d['facts'] = [('Canvas', '64 × 64' if n < 3 else '128 × 128' if n == 3 else '256 × 256'),
                      ('Evaluated images', str(count)), ('Heuristic agreement', agreement), ('Independent validation', 'Pending')]
        if n == 6:
            d['facts'] = [('Canvas','256 × 256'),('Evaluated images','1,400'),('Heuristic agreement',agreement),('Independent validation','Pending')]
            d['procedure'] = 'Historical transfer training used two spatial conditioning channels. The corrected evaluation analyses 200 saved images per class and predicts each class from image morphology independently of its requested class. Spatial measurements compare generated fissures against commanded masks.'
        elif n == 3:
            d['facts'] = [('Canvas','128 × 128'),('Evaluated images',str(count)),('Heuristic agreement',agreement),('Independent validation','Pending')]
        page = base.page(n, d)
        page = page.replace('</head>', '<link rel="stylesheet" href="research.css"></head>')
        page = page.replace('<section class="facts">', '<div class="research-notice">Historical training record with a current evaluation addendum. Proxy scores are not validated structural-mechanism accuracy. <a href="research.html">View research status</a></div><section class="facts">', 1)
        page = page.replace('<section class="full-report">', panel + '<section class="full-report">', 1)
        page = page.replace('<details open><summary>Expand / collapse the original report</summary>', '<p>Archival report: it may contain superseded measurements or interpretations. The current addendum above takes precedence.</p><details><summary>Read the historical technical record</summary>')
        (SITE / f'experiment-{n}.html').write_text(page, encoding='utf-8')
        rows.append([f'Experiment {n}', count, agreement, 'Descriptive heuristic evaluation'])
        panels.append(f'<section class="research-section"><h2>Experiment {n}</h2><p>{esc(text)}</p><a href="experiment-{n}.html">Read complete experiment page</a><img class="research-plot" src="assets/research/exp{n}_overview.png" alt="Experiment {n} results"></section>')
    protocol = json.loads((project / 'pipeline/research_protocol.json').read_text())
    shutil.copyfile(project / 'pipeline/research_protocol.json', assets / 'protocol.json')
    audits = []
    current=json.loads((project/'datasets/current_research_split.json').read_text())['version']
    for name in ['exp1_exp2_sdnet_64','exp3_augmented_128','exp4_level_a_morphology_256','exp5_exp6_level_bc_mechanisms_256']:
        source = project / 'datasets' / name / current / 'audit.json'
        audit = json.loads(source.read_text())
        shutil.copyfile(source, assets / (name + '_audit.json'))
        audits.append([name, audit['rows'],audit['split_counts']['train'],audit['split_counts']['validation'],audit['split_counts']['test'],audit['acquisition_overlap']])
    content = '<p class="eyebrow">RESEARCH AUDIT · 21 NOVEMBER 2026</p><h1>Evidence you can inspect.</h1><p class="lede">Six historical diffusion experiments, corrected evaluation, and a reproducible protocol for the next research lineage.</p><div class="research-notice"><strong>Publication readiness: incomplete.</strong> Exact-duplicate audits, prospective splits, real-mask detector calibration, and a three-seed historical ablation pilot are available. Full clean training, expert labels, generated-image detector validation, controlled clean-lineage comparisons and external evaluation remain outstanding.</div>'
    content += '<section class="research-section"><h2>What has actually been measured</h2>' + table(['Experiment','Generated images','Heuristic agreement','Evidence type'],rows) + '<p>The experiments differ in resolution, data and evaluation protocol. These numbers are descriptive, not a controlled ranking.</p></section>'
    source_audit=json.loads((project/'datasets/source_content_v1/summary.json').read_text())
    screen=project/'datasets/near_duplicate_screen_v1'
    before=json.loads((screen/'research_v4_summary.json').read_text())
    after=json.loads((screen/'research_v5_summary.json').read_text())
    links=json.loads((screen/'conservative_links_summary.json').read_text())
    for filename in ('research_v4_summary.json','research_v5_summary.json','conservative_links_summary.json','research_v4_screen.png'):
        shutil.copyfile(screen/filename,assets/filename)
    content += f'<section class="research-section"><h2>Current {esc(current)} splits</h2>' + table(['Dataset','Rows','Train','Validation','Test','Acquisition overlap'],audits) + f'<p>Audited {source_audit["original_files"]:,} original files: {source_audit["decoded_unique_images"]:,} distinct decoded images and {source_audit["duplicate_files_beyond_first"]:,} duplicate copies beyond first occurrences. Connected groups link inferred SDNET acquisitions, other original-image identities, exact byte/decoded-RGB aliases, and conservative similarity candidates consistently across all six experiments.</p><p>Earlier split versions remain archival. Existing checkpoints were exposed to historical images, so these splits support a new training lineage only.</p><a href="assets/research/source_content_summary.json">Inspect content audit</a></section>'
    content += f'<section class="research-section"><h2>Cross-split similarity screening</h2><p>The v4 split contained {before["cross_split_candidate_pairs"]:,} cross-split similarity candidate pairs: 169 training–test, 195 training–validation and 22 validation–test. V5 conservatively groups all {links["pairs"]:,} declared-threshold links, including links originally within a split, before assigning connected components. Repeating the same screen on v5 found {after["cross_split_candidate_pairs"]} cross-split candidates.</p><img class="research-plot" src="assets/research/research_v4_screen.png" alt="Similarity screening candidates by hash distance and split pairing, with v4 and v5 counts"><h3>Assumptions and limits</h3><p>Candidate thresholds are pHash Hamming distance ≤3 and dHash distance ≤6. The thresholds are uncalibrated on this dataset. Grouping suspected aliases does not label them confirmed duplicates; false-positive links can reduce the number of independent groups. The search may miss crops, rotations, flips and larger edits. Site and structure independence remain unverified. Zero candidates under this search is not proof of complete independence.</p><p>Full research training now requires v5. A v5 one-batch training/validation smoke run passed; full clean models and research evaluations remain pending.</p><p><a href="assets/research/research_v4_summary.json">Archived v4 screening</a> · <a href="assets/research/research_v5_summary.json">V5 verification</a> · <a href="assets/research/conservative_links_summary.json">Grouping protocol</a></p></section>'
    content+=supporting
    provenance=project/'datasets/source_provenance'
    rc=json.loads((provenance/'rc10519_local_inventory.json').read_text())
    qualified_path=project/'datasets/exp5_exp6_level_bc_mechanisms_256/research_v5/strict_rc_geometry_v1/audit.json'
    qualified=json.loads(qualified_path.read_text())
    for filename in ('rc10519_local_inventory.json','rc10519_publisher_record.json','rc10519_quality.png'):
        shutil.copyfile(provenance/filename,assets/filename)
    shutil.copyfile(qualified_path,assets/'strict_rc_geometry_audit.json')
    qrows=[[s,qualified['base_split_counts'][s],qualified['excluded_by_split'][s],qualified['qualified_split_counts'][s]] for s in ('train','validation','test')]
    content+=f'<section id="source-quality" class="research-section"><h2>RC source and annotation quality</h2><p>The matching publisher record is <a href="https://data.mendeley.com/datasets/3ghxck2wd5/1">Wang (2026), Mendeley Data v1</a>, DOI 10.17632/3ghxck2wd5.1. The publisher requests the <a href="https://onlinelibrary.wiley.com/doi/10.1002/suco.202300937">Wang et al. (2024) paper</a> as well and declares CC BY 4.0. Archive identity against publisher checksums and upstream acquisition lineage remain unverified.</p><p>The local subset contains {rc["local_image_files"]:,} images and matching JSON files, with {sum(rc["shape_labels"].values()):,} crack-labeled rectangles and no empty annotations. Extracted file CRCs and sizes match the local ZIP records; archive SHA256 values are saved. This verifies local consistency, not publisher identity or label correctness.</p><p>{rc["images_with_issues"]:,} source annotations have a geometry or metadata flag: 1,122 contain coordinates outside image bounds and 150 have annotation/image dimension mismatches (categories overlap). Of these, {qualified["excluded_total"]} records feed the base Experiment 5/6 dataset. The strict_rc_geometry_v1 view excludes them conservatively, while retaining v5 component IDs and split assignments.</p>'+table(['Partition','Base v5 records','Excluded','Qualified records'],qrows)+'<img class="research-plot" src="assets/research/rc10519_quality.png" alt="RC annotation flags and prospective Experiment 5/6 qualified dataset counts"><h3>Assumptions and limits</h3><p>Any flagged RC annotation record is excluded from the prospective box-dependent Exp5/6 view. This includes small coordinate overruns; coordinates are not automatically repaired or rescaled. All seven proxy classes remain present. Passing geometry checks does not validate crack presence, bounding boxes, segmentation masks or structural failure mechanisms. Exclusion does not prove every flagged label is wrong. Exp3 uses image-level labels without boxes and is not filtered by this geometry policy. Historical models and datasets remain archival.</p><p><a href="assets/research/rc10519_local_inventory.json">Local inventory and archive hashes</a> · <a href="assets/research/rc10519_publisher_record.json">Publisher record and identity limits</a> · <a href="assets/research/strict_rc_geometry_audit.json">Qualification policy and manifest signatures</a></p></section>'
    content += '<section class="research-section"><h2>Reproducible pipeline checks</h2><p>Experiments 4–6 passed one-batch training and validation smoke runs. An Experiment 6 checkpoint restart matched uninterrupted model and EMA tensors exactly on this machine. These checks establish pipeline behavior, not scientific model quality.</p><h3>Requirements before strong research claims</h3><ul>' + ''.join('<li>'+esc(s)+'</li>' for s in protocol['publication_gates']) + '</ul><a href="assets/research/protocol.json">Inspect the prospective protocol</a></section>'
    content += ''.join(panels)
    references=json.loads((assets/'references.json').read_text())
    content+='<section class="research-section"><h2>Method and dataset sources</h2><ul>'+''.join(f'<li><a href="{esc(r["url"])}">{esc(r["key"])}</a>: {esc(r["role"])}</li>' for r in references['references']+references.get('additional_verified_sources',[]))+'</ul><p><a href="assets/research/references.bib">Bibliography file</a>. RC archive byte identity, upstream acquisition lineage and engineering-context sources remain outstanding; none of these method references establishes physical causation in generated patches.</p></section>'
    document = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Research status | Crack Synthesis</title><link rel="stylesheet" href="experiment.css"><link rel="stylesheet" href="research.css"></head><body><header class="detail-header"><a href="index.html">CRACK / SYNTHESIS</a><nav><a href="index.html#experiments">Experiment archive</a></nav></header><main>'+content+'</main><footer class="detail-footer">Sources: saved project evaluations and dataset audit records.</footer></body></html>'
    (SITE / 'research.html').write_text(document, encoding='utf-8')
    index = (SITE / 'index.html').read_text(encoding='utf-8')
    if 'id="current-research-status"' not in index:
        index = index.replace('<main id="top">', '<main id="top"><section id="current-research-status" class="section-shell research-notice"><strong>Current evidence and research status</strong><p>Corrected results, all six experiment plots, dataset audits and remaining validation requirements are now available.</p><a href="research.html">Read the current research audit</a></section>')
    index = index.replace('</head>', '<link rel="stylesheet" href="research.css"></head>') if 'href="research.css"' not in index else index
    # Mark the legacy overview as historical; corrected pages govern all current claims.
    index = index.replace('<section id="experiments"', '<section id="experiments"')
    index = index.replace('RESEARCH ARCHIVE · 2026','HISTORICAL RESEARCH ARCHIVE · CURRENT AUDIT AVAILABLE')
    index = index.replace('CLOSED · NO FINAL QUALITY METRICS','EVALUATION AVAILABLE')
    index = index.replace('Training completed; the user closed the experiment before the large evaluation run.', 'Training completed; a subsequent 400-image heuristic evaluation is available on the experiment page.')
    index = index.replace('Experiment 3 has no final quality metrics.', 'See the current research audit for updated measurements and protocol differences.')
    index = index.replace('Experiment 3 was closed before full generation and formal evaluation.', 'Experiment 3 now has a 400-image heuristic evaluation; independent validation is pending.')
    cards = []
    for n, row in enumerate(rows, 1):
        cards.append(f'<article class="experiment-card"><div class="exp-number">{n:02d}</div><div class="exp-main"><p class="eyebrow">HISTORICAL RUN · AUDITED EVALUATION</p><h3>{esc(base.common[n]["title"])}</h3><p>{row[1]:,} saved images evaluated. Independent validation remains pending.</p><div class="chip-row"><span>Heuristic agreement: {esc(str(row[2]))}</span></div></div><a class="exp-open" href="experiment-{n}.html" aria-label="Read Experiment {n}">Read</a></article>')
    index = re.sub(r'(<div class="experiment-list">).*?(</div>\s*</section>)', lambda m: m[1]+''.join(cards)+m[2], index, count=1, flags=re.S)
    index = re.sub(r'(<table class="comparison-table">).*?</table>', lambda m: '<table class="comparison-table"><thead><tr><th>Experiment</th><th>Evaluated images</th><th>Heuristic agreement</th><th>Evidence type</th></tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>', index, count=1, flags=re.S)
    index = re.sub(r'<div id="fid-chart".*?</div>', '<p>Historical FID protocols differ. Inspect each experiment record and the current research audit; no cross-protocol ranking is shown.</p>', index, count=1, flags=re.S)
    index = index.replace('Experiment 6 improved physical adherence but worsened FID/KID and retained low torsion continuity.', 'Experiment 6 has a corrected 87.86% morphology-heuristic agreement score. Physical-mechanism validity is unproven.')
    index = index.replace('RTX 3070', 'CUDA GPU').replace('latest spatial-physics run', 'hardware details retained in run records')
    (SITE / 'index.html').write_text(index, encoding='utf-8')
    print('Built six audited experiment pages and research.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    build(parser.parse_args().project_root.resolve())
