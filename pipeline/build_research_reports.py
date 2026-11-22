"""Build evidence-linked retrospective research reports; never trains or changes metrics."""
import csv
import hashlib
import json
import shutil
import math
import re
import importlib.util
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REVISION='2026-11-22'
NAMES=['unconditional_ddpm','conditional_ddpm','conditional_ddpm_128_augmented','crack_type_conditional','structural_mechanism_ddpm','exp6_level_c_spatial_physics']
DATASETS=['exp1_exp2_sdnet_64','exp1_exp2_sdnet_64','exp3_augmented_128','exp4_level_a_morphology_256','exp5_exp6_level_bc_mechanisms_256','exp5_exp6_level_bc_mechanisms_256']
TRAINERS=['train.py','train_conditional.py','train_conditional_128_augmented.py','train_crack_type_256.py','train_exp5_mechanisms_256.py','train_exp6_spatial_physics_256.py']
EVALUATORS=['evaluate_exp1_2_3_errors.py']*3+['evaluate_crack_type_256.py','evaluate_exp5_mechanisms_256.py','evaluate_exp6_spatial_physics_256.py']
PARAMETERS=[3258115,3258627,3258627,13820003,13820323,13821043]
TITLES=['Unconditional concrete surface synthesis at 64 by 64 pixels','Binary crack status conditioning at 64 by 64 pixels','Higher resolution binary conditioning with RC data augmentation','Five class morphology conditioning at 256 by 256 pixels','Seven mechanism inspired morphology classes with transfer learning','Spatial trajectory and aperture field conditioned crack synthesis']


def read_json(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x).replace('|',' / ').replace('\n',' ') for x in row)+' |' for row in rows])
def pct(value):return f'{100*value:.2f}%'
def link(path,label=None):
    path=Path(path);relative=path.relative_to(ROOT).as_posix()
    return f'[{label or path.name}](../../{relative})'


def preserve(folder,n):
    archive=folder/'report_archive/20261122-original'
    archive.mkdir(parents=True,exist_ok=True)
    registry=archive/'archive_index.json'
    if registry.exists():
        result=read_json(registry)
        for entry in result['entries']:
            if sha(ROOT/entry['archive'])!=entry['sha256']:raise ValueError('Original report archive changed')
        for source in sorted((folder/'plots').glob('*')):
            if not source.is_file():continue
            target=archive/'plots'/source.name
            if target.exists():continue
            target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target)
            result['entries'].append(dict(source=str(source.relative_to(ROOT)),archive=str(target.relative_to(ROOT)),sha256=sha(source)))
        if n==6:
            source=folder/'EVALUATION_METRICS.json';target=archive/'original_stored_metrics.json'
            if not target.exists():
                shutil.copyfile(source,target)
                result['entries'].append(dict(source=str(source.relative_to(ROOT)),archive=str(target.relative_to(ROOT)),sha256=sha(source)))
        registry.write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result
    entries=[]
    for source in sorted(folder.glob('*.md')):
        if any(k in source.name for k in ('REPORT','DOCUMENTATION','STATUS')):
            target=archive/source.name
            shutil.copyfile(source,target)
            entries.append(dict(source=str(source.relative_to(ROOT)),archive=str(target.relative_to(ROOT)),sha256=sha(source)))
    # The existing website is the recoverable technical record for Exp2/3.
    source=ROOT/f'website/experiment-{n}.html'
    target=archive/'previous_webpage.html';shutil.copyfile(source,target)
    entries.append(dict(source=str(source.relative_to(ROOT)),archive=str(target.relative_to(ROOT)),sha256=sha(source)))
    if n in (2,3):
        source=ROOT/f'website/assets/reports/experiment_{n}_formula_report.pdf'
        target=archive/source.name;shutil.copyfile(source,target)
        entries.append(dict(source=str(source.relative_to(ROOT)),archive=str(target.relative_to(ROOT)),sha256=sha(source)))
    result=dict(revision=REVISION,policy='Exact pre-revision bytes; historical statements may be superseded by the current report.',entries=entries)
    registry.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def training(folder):
    path=next(iter(sorted((folder/'logs').glob('*loss*.csv'))))
    valid=[];ignored=0
    with path.open(encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            try:valid.append(dict(epoch=int(row['epoch']),loss=float(row['mean_loss'])))
            except (ValueError,KeyError):ignored+=1
    unique={r['epoch']:r for r in valid}
    rows=sorted(unique.values(),key=lambda r:r['epoch'])
    return dict(path=path,rows=rows,raw_numeric_rows=len(valid),repeated_epochs=len(valid)-len(rows),ignored_rows=ignored,
                final=rows[-1],minimum=min(rows,key=lambda r:r['loss']))


def dataset_rows(n):
    path=ROOT/'datasets'/DATASETS[n-1]/'train.csv'
    with path.open(encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream))
    return path,rows


NARRATIVE={
1:dict(
 abstract='A compact unconditional DDPM provides a baseline for generating concrete surface patches without a crack-status input. Fifty logged training epochs end at noise-prediction MSE 0.015996. A later analysis of 1,000 saved outputs flags 771 images as cracked using a fixed image heuristic. This 77.10% detector prevalence is not an independently labeled crack rate and cannot be directly equated with the 15.13% cracked-label prevalence in the training manifest. The experiment demonstrates executable surface synthesis, while controlled crack generation and engineering validity remain unestablished.',
 aim='Can a compact unconditional diffusion model learn concrete surface appearance, and what does its output distribution reveal about the need for explicit crack conditioning?',
 hypothesis='A model trained on the mixed image distribution can synthesize concrete-like texture. Because no condition is supplied, this experiment cannot test requested crack-class compliance.',
 procedure=['Index SDNET2018 and retain cracked/non-cracked and surface metadata; resize RGB patches to 64 by 64 and normalize image tensors.','Train SmallUNet to predict Gaussian noise using randomly sampled diffusion timesteps. Labels remain metadata and are not model inputs.','Save the 50-epoch checkpoint and generate the existing unconditional evaluation collection using the historical sampler.','Analyse all 1,000 saved images with the 64-pixel detector. Compute reference-summary deviations and a separate 160-real/160-generated feature diagnostic.'],
 interpretation='The fixed heuristic responds to local dark structures, including texture, shadows and aggregate boundaries. Its large prevalence relative to the dataset label prior therefore supports a detector/distribution mismatch concern, not a claim that the model generates five times as many genuine cracks. The low continuity proxy (0.116) and increased contrast relative to reference summaries are consistent with fragmented or over-detected dark texture. They do not locate a uniquely identified model failure without independently labeled generated images.',
 assumptions=[('Image-level labels','SDNET labels describe patch status; they are not pixel masks.','Useful metadata, not supplied to the unconditional model.'),('Resolution','64-pixel resizing retains enough texture for a baseline.','Thin fissures may disappear or merge; results do not establish topology preservation.'),('Prior comparison','Dataset labels and a generated-image heuristic measure comparable crack presence.','Unverified; their 15.13% and 77.10% rates use different label sources.'),('Reference statistics','Differences from reference mean luminance/contrast describe distributional deviation.','They are not paired reconstruction errors against a correct target image.')],
 errors=[('Earlier FID 5.604571 presented alongside later FID 128.886','Different feature dimensions, reference choices and preprocessing are used; keep both protocols separate.'),('Detector prevalence treated as genuine crack correctness','There is no requested class and no generated-image ground truth.'),('Repeated epoch 1 and an embedded CSV header','Use the last numeric record per epoch for the report; preserve the raw log.')],
 conclusion='Experiment 1 establishes an unconditional synthesis baseline and its reproducible checkpoint. It does not establish a controllable crack generator. Its main research value is to motivate explicit conditioning and to expose how texture-sensitive detectors can distort crack-prevalence interpretation.'),
2:dict(
 abstract='Binary class embeddings and inverse-frequency sampling introduce a requested crack-status input into the 64-pixel DDPM. The historical checkpoint completes 50 logged epochs with final noise MSE 0.015247. Later heuristic analysis covers 1,000 outputs per requested class. Requested-label agreement is 58.45% overall: 84.40% for non-cracked requests and 32.50% for cracked requests. These are detector agreement measurements, not independently validated crack-classification accuracy. Conditioning is implemented, but reliable crack-specific control is not established.',
 aim='Does supplying a binary crack-status label produce output groups that a fixed image-only crack heuristic distinguishes in the requested direction?',
 hypothesis='A learned binary embedding and balanced training sampler should provide a class-control signal. Support requires observing both requested groups and measuring error asymmetry; a low distribution distance alone cannot verify crack presence.',
 procedure=['Reuse the processed 64-pixel SDNET training manifest and encode non-cracked as 0 and cracked as 1.','Use inverse class-frequency sample weights with replacement. Balancing changes exposure probabilities, not the number of unique cracked images.','Add the learned class embedding to the timestep representation and train the conditional SmallUNet for 50 logged epochs.','Generate the saved 1,000-image collection for each condition, analyse every output, and compare requested labels with heuristic assignments.'],
 interpretation='The dominant observed failure is asymmetric agreement. There are 675 cracked requests assigned non-cracked and 156 non-cracked requests assigned cracked. Precision, recall and F1 derived from this matrix describe correspondence to requested conditions under this heuristic. Requested labels are generation instructions, so these statistics are not supervised classifier performance on human-labeled crack images. A balanced output count does not remove detector bias or make the image collection representative of deployment prevalence.',
 assumptions=[('Binary labels','Crack status is sufficient as an initial conditioning variable.','No crack path, width, severity or cause is specified.'),('Balanced sampling','Rare cracked images need greater exposure.','Sampling with replacement may repeat the same examples; it does not create independent data.'),('Generated class targets','The requested condition is the intended output attribute.','A request is not evidence that the generated image contains the attribute.'),('Auxiliary classifier','A preliminary classifier or visual count can validate generated crack presence.','Not established; current primary results use the documented image heuristic.')],
 errors=[('Low historical class-specific FID interpreted as proven crack control','Feature similarity does not validate the crack attribute; later 32.50% cracked-request agreement exposes this distinction.'),('Missing standalone technical report','This report provides full methods, sample denominators, confusion counts and protocol-qualified interpretation.'),('Accuracy terminology without ground truth','Use requested-label agreement throughout; retain original JSON field names only as artifact identifiers.')],
 conclusion='Binary conditioning adds an explicit generation instruction, but the saved heuristic evaluation shows substantial failure on cracked requests. The experiment is a useful control baseline, not a validated crack classifier or a demonstration of reliable structural damage synthesis.'),
3:dict(
 abstract='Experiment 3 combines a 128-pixel conditional DDPM with a cracked-image subset labeled RC_10519, producing a historical training manifest of 61,001 images. Fifty logged epochs end at noise MSE 0.014058. The later saved evaluation contains 400 images, evenly divided between requested classes, with 64.25% overall heuristic agreement and 51.00% agreement for cracked requests. Resolution, training composition and sampling protocol change together, so the result is not an isolated estimate of a resolution benefit. The earlier closed-experiment statement that no metrics existed is superseded by this later descriptive evaluation.',
 aim='What output behavior is observed after increasing resolution and adding the RC cracked subset to binary conditional training?',
 hypothesis='Greater spatial sampling and additional cracked examples may retain finer visible detail. This historical experiment changes both factors together and cannot identify their separate causal effects.',
 procedure=['Combine 50,482 SDNET training records with 10,519 local RC images assigned to the cracked class. Preserve source paths and annotation availability.','Resize RGB images to 128 by 128, use the binary embedding and inverse-frequency sampler, and train for 50 logged epochs.','Preserve the earlier stopped-generation record as historical context; use the later completed collection of 200 non-cracked and 200 cracked requests for this report.','Analyse all 400 outputs using the 128-pixel heuristic, and calculate the separately scoped 160-image feature diagnostic from the saved evaluation protocol.'],
 interpretation='The saved matrix contains 155 agreements among 200 non-cracked requests and 102 among 200 cracked requests. Although these rates differ from Experiment 2, the detector window, thresholds, span criteria, image resolution, data composition and generated sample budget differ. The results support a descriptive observation, not a controlled claim that doubling resolution improved crack fidelity. Adding a named source also does not establish source independence: later content and perceptual audits link apparent aliases across collections.',
 assumptions=[('Augmentation source','All local RC images are treated as cracked.','All local JSON files contain crack-labeled rectangles, but independent image-level correctness remains unverified.'),('Higher resolution','128-pixel sampling can retain more visible detail.','Pixel count is not a measurement of physical crack width or improved continuity.'),('Dataset independence','A different source name supplies independent examples.','Contradicted as a general rule by later cross-source alias candidates; v5 grouping addresses the declared similarity screen.'),('Historical status','The stopped early run describes the final experiment state.','Superseded by the later 400-image analysis; earlier status records remain archived.')],
 errors=[('No metrics claimed after a later evaluation existed','Report the 400-image analysis and its dates/status separately from the earlier closure.'),('RC described as independently new data without qualification','Matching publisher attribution is available; payload identity and acquisition independence are not fully verified.'),('Improvement attributed solely to resolution','Resolution, composition and detector protocol are confounded.')],
 conclusion='Experiment 3 provides a higher-resolution, augmented-data conditional checkpoint and a later 400-image descriptive evaluation. It motivates native-resolution morphology work, but it does not isolate the effect of resolution or establish independent external-dataset generalization.'),
4:dict(
 abstract='A 13,820,003-parameter attention U-Net is trained at 256-pixel resolution with five morphology-derived class labels. The historical manifest contains 13,139 patches from SDNET2018 and DeepCrack. Fifty logged epochs end at noise MSE 0.014690. The 80-image heuristic evaluation reports zero clean-surface agreement for non-cracked requests and zero orientation alignment for longitudinal, transverse and diagonal requests. Its map predicate accepts all 16 map requests. These results do not support the earlier statement that fragmentation was resolved, and their interpretation depends strongly on detector behavior.',
 aim='Can native 256-pixel synthesis with attention and morphology labels produce outputs that satisfy class-specific image-geometry predicates?',
 hypothesis='Higher-resolution images and nonlocal feature interactions may help represent extended structures. The hypothesis must be tested with validated topology measurements and controlled ablations; architecture choice alone is not evidence of continuity.',
 procedure=['Prepare native 256-pixel patches using source crack masks and image-label curation. The actual manifest contains DeepCrack and SDNET records; CrackForest is not present in this Exp4 manifest.','Assign five image-geometry classes and balance their exposure during training. Use horizontal and vertical flips in the historical trainer.','Train the attention U-Net for 50 logged epochs. The current historical trainer sets class-drop probability 0.10; the earlier 0.15 report statement is inconsistent with that source.','Sample the saved 16-image collection per class with the historical 50-step DDIM/CFG path and evaluate the class-specific morphology rules.'],
 interpretation='The recorded linear continuity ratios are 0.0676, 0.0553 and 0.0744, with approximately 79, 84 and 70 significant components for the three linear classes. Calling this resolved fragmentation contradicts the recorded proxy measurements. The map rule accepts a dispersed set of components; a 100% acceptance rate can therefore be compatible with over-segmented texture. Detector calibration and a prospective paired architecture comparison are needed to distinguish generator limitations from evaluator failure.',
 assumptions=[('Morphology labels','Orientation and component structure summarize the intended classes.','These are image-geometry labels, not independently verified structural failure causes.'),('Native patches','256-pixel processing preserves source detail.','Resolution alone does not preserve topology, and crops may remove structural context.'),('Flip invariance','Horizontal/vertical flips preserve Exp4 morphology labels.','Suitable only for geometry categories; they do not preserve a known beam tension-zone frame.'),('Attention','Long-range feature interactions should improve continuity.','Design rationale, not demonstrated causal benefit in this historical comparison.'),('Map detection','Many components with a dispersed mask indicate map cracking.','Texture fragmentation can also satisfy the rule.')],
 errors=[('Fragmentation labeled resolved despite low continuity fractions','Replace the claim with the observed measurements and evaluator uncertainty.'),('CrackForest listed in the Exp4 training set','The manifest source counts contain SDNET2018 and DeepCrack only.'),('Class dropout reported as 15%','The source trainer uses 10%; historical CLI/runtime provenance remains incomplete.'),('DDIM described as guaranteeing invertibility','Finite clipped sampling does not establish exact inverse reconstruction or cross-device bit identity.')],
 conclusion='Experiment 4 implements native-resolution attention and five-class geometry conditioning. Its saved evaluation does not demonstrate resolved fragmentation or reliable linear-class control. It establishes an architectural and taxonomy step whose benefits require a new, matched and independently evaluated comparison.'),
5:dict(
 abstract='Experiment 5 expands the label space to seven mechanism-inspired morphology categories and documents transfer from the Exp4 attention backbone. The historical training set has 15,440 patches and the phase completes 25 logged epochs with final noise MSE 0.012985. On 112 saved generated images, class-specific heuristic acceptance ranges from 50.00% for flexure-shear to 100.00% for shrinkage-map. These predicates depend on the requested class and are not a seven-class independent classifier. Local source and annotation audits further qualify the dataset: the prospective Exp5/6 view excludes 481 flagged RC records.',
 aim='Can a class-conditioned generator express geometry patterns inspired by named failure mechanisms while retaining a compact attention backbone?',
 hypothesis='Orientation, curvature, component multiplicity and apparent width taper can provide useful controllable visual proxies. Their correspondence to real loading, reinforcement, corrosion or shrinkage causes requires source context and independent expert labels.',
 procedure=['Combine DeepCrack training and test originals, CrackForest masks, RC box-guided fissure extraction and SDNET non-cracked patches into the historical 15,440-patch manifest.','Map mask geometry to seven proxy classes, canonically orient selected patches and prohibit vertical flipping in the historical mechanism trainer.','Initialize shared weights from Exp4 and construct the expanded class embedding table using the documented projection recipe; fine-tune for 25 logged epochs.','Generate 16 saved examples per class, apply the requested-class predicate, and compute the separate historical feature-distance diagnostic against selected training references.'],
 interpretation='Sixteen examples per class give a coarse 6.25 percentage-point resolution and cannot establish robust class-level performance. The torsion and map acceptance rules do not require the same crack-presence gate as the other rules. Accordingly, torsion acceptance (15/16) exceeds its presence count (13/16), and map acceptance (16/16) exceeds its presence count (13/16). These are different predicates, not contradictory estimates of one accuracy quantity. The geometry-to-mechanism mapping is a modeling assumption rather than a recovered physical diagnosis.',
 assumptions=[('A1 Canonical frame','The lower image edge represents a tension-side reference after curation.','A synthetic image convention; cropped field images lack independently verified member/load frames.'),('A2 Augmentation','Horizontal flipping preserves the convention; vertical flipping does not.','Implemented augmentation choice, not proof of physical orientation.'),('A3 Curvature','Piecewise mask orientation differentiates straight and curved patterns.','Sensitive to segmentation, component selection and crop extent.'),('A4 Taper','EDT widths summarize apparent lower/upper width.','Pixel-scale proxy; no calibrated opening displacement or strain measurement.'),('A5 Multiplicity','Parallel diagonal components approximate torsion-like appearance.','A single face does not identify a member-wide torsional mechanism.'),('A6 Alignment','Horizontal splitting approximates reinforcement-parallel corrosion appearance.','Actual bar layout, corrosion and causation are unobserved.'),('A7 Dispersion','Multi-component masks approximate shrinkage-map appearance.','Texture, other damage processes and detector fragmentation are alternative explanations.')],
 errors=[('Mechanism acceptance described as validated physical accuracy','Report requested-class predicate acceptance and identify the absence of expert mechanism labels.'),('Presence and acceptance percentages combined as one quantity','Retain separate denominators and predicate definitions, especially for torsion and map classes.'),('Training references described as independent validation','The historical fidelity evaluator selects reference patches from the training manifest.'),('RC box geometry assumed reliable throughout','The later qualified view removes 481 affected base records; historical models are not retroactively corrected.')],
 conclusion='Experiment 5 implements a seven-category, mechanism-inspired visual taxonomy and transfer-learning stage. It demonstrates computable geometry predicates and controllable inputs, while physical mechanism validity and independent generalization remain unproven.'),
6:dict(
 abstract='Experiment 6 adds two spatial conditioning channels to a transferred 256-pixel U-Net and completes 15 logged fine-tuning epochs. The five-channel checkpoint contains 13,821,043 parameters, 720 more than Exp5. The corrected image-only heuristic assigns 1,230 of 1,400 saved generated images to their requested class (87.86%); torsion agreement is 78/200 (39.00%). The earlier 99.29% figure used requested-class-dependent assignments and is superseded. A separate 126-image paired-field pilot supports sensitivity to spatial inputs, but neither that pilot nor geometric correspondence establishes physically valid fracture simulation.',
 aim='How closely do generated image structures correspond to supplied spatial templates, and how does the output change when those templates are zeroed or shuffled?',
 hypothesis='A spatially conditioned denoiser can learn correspondence between image-derived training fields and generated appearance. Success must be separated into image-only class agreement, template correspondence, image realism and independent physical validity.',
 procedure=['Use the historical seven-class patch set. Normalize chromatic appearance and extract a local-darkness fissure mask from each training image.','Construct a smoothed distance-ridge channel and an EDT-based aperture channel from that same image. Concatenate them with noisy RGB, producing five input channels.','Copy the shared Exp5 backbone, retain the first three input-convolution channels and initialize the added spatial-channel weights to zero; jointly drop class and spatial information in 15% of training samples.','Fine-tune for 15 logged epochs. At inference, use a class-specific procedural trajectory/template generator and 50-step guided DDIM sampling.','Pair saved generated images with commanded fields by sample identifier, run corrected image-only heuristic assignment on all 1,400 images, and retain target-dependent geometric predicates as separate diagnostics.','Run the separately identified three-seed, correct/zero/shuffled field pilot with matched initial-noise states; evaluate it with both fixed and exploratory calibrated detectors.'],
 interpretation='The most consequential corrected failure is torsion-to-map confusion: 122 of 200 torsion requests are assigned shrinkage-map by the image-only heuristic. The legacy target-dependent torsion predicate still accepts all 200 outputs, which shows why predicate acceptance cannot substitute for independent assignment. High trajectory recall also does not by itself imply precise tracing: broad or noisy dark masks can cover a commanded corridor. Balanced precision/recall/F1 is implemented for the prospective pipeline, but the historical main evaluation should not be relabeled as having completed that new protocol.',
 assumptions=[('C1 Spatial fields','Image-derived ridge and EDT fields encode visible crack geometry.','Self-supervised image proxies, not measured stress, strain or calibrated crack opening.'),('C2 Command generator','Procedural class templates approximate mechanism-inspired crack shapes.','The implementation samples shapes, angles, widths and warped cell boundaries; it is not a solved boundary-value fracture model.'),('C3 Input-channel surgery','Zero-initialized extra channels preserve the initial RGB convolution contribution.','Architectural initialization property; no guarantee of convergence, compliance or realism after training.'),('C4 Chromatic transform','Strong desaturation and channel offsets reduce unwanted tints.','Changes image distribution; clipping/quantization prevent a guarantee of perfect luminance preservation.'),('C5 Joint CFG','Class and spatial fields are dropped together during training.','Zero-field plus present-class ablations can be outside the historical conditioning distribution.'),('Train/inference correspondence','Image-extracted fields and procedural inference fields have compatible distributions.','Unverified domain-transfer assumption requiring controlled validation.'),('Pixel geometry','Distances and widths in pixels are meaningful image measurements.','No millimetre scale, material properties, equilibrium residual or structural safety assessment is established.')],
 errors=[('99.29% labeled independent seven-class accuracy','Superseded by the corrected morphology-only matrix: 1,230/1,400 agreements.'),('Procedural generator described as guaranteed physically valid','The code does not solve a calibrated fracture boundary-value problem or validate equilibrium/material failure.'),('High recall treated as sufficient template fidelity','Report false positives, precision and missing detections alongside recall; distances are conditional when detections are absent.'),('Zero-field pilot described as a trained class-only baseline','It is an input ablation on a historical checkpoint; a separately trained matched class-only control remains pending.'),('New split and quality audits presented as validation of old models','The historical weights and transfer ancestors remain exposed to earlier data.')],
 conclusion='Experiment 6 provides the strongest direct evidence of spatial input sensitivity in the project, together with a corrected 1,400-image morphology evaluation. The defensible claim is controllable image-template correspondence under the studied heuristics. Physical fracture validity, expert mechanism correctness, clean generalization and downstream inspection utility remain open research questions.')
}


def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return f'{100*(c-h):.2f}–{100*(c+h):.2f}%'


def result_tables(n,data):
    if n==1:
        p=data['dataset_prior_comparison'];s=data['surface_and_morphology_error_statistics']
        return table(['Measurement','Saved result','Interpretation'],[
            ['Generated heuristic crack detections','771 / 1,000 (77.10%)','Detector prevalence; Wilson 95% interval 74.39–79.60%'],
            ['Historical training label prior',pct(p['real_sdnet2018_cracked_rate']),'Different label source; not a detector-calibrated reference'],
            ['Mean continuity proxy',s['mean_continuity_ratio'],'Largest-component statistic; depends on mask extraction'],
            ['Generated / reference luminance mean',f"{s['generated_luma_mean']} / {s['real_reference_luma_mean']}",'Intensity units, not physical surface reflectance'],
            ['Generated / reference contrast summary',f"{s['generated_contrast_mean']} / {s['real_reference_contrast_std']}",'Image intensity standard deviation'],
            ['Luminance deviation MAE / RMSE',f"{s['luma_error_mae']} / {s['luma_error_rmse']}",'Deviation from a reference summary; not a paired reconstruction']])
    cm=data.get('confusion_matrix')
    if cm:
        matrix=cm['matrix_counts'];order=cm['class_order'];total=sum(map(sum,matrix));diagonal=sum(matrix[i][i] for i in range(len(order)))
        rows=[]
        for i,name in enumerate(order):
            k=matrix[i][i];count=sum(matrix[i]);pred=sum(row[i] for row in matrix)
            precision=k/pred if pred else 0;recall=k/count;f1=2*precision*recall/(precision+recall) if precision+recall else 0
            rows.append([name,f'{k} / {count}',pct(recall),wilson(k,count),pct(precision),f'{f1:.4f}'])
        text=f'Overall requested-label agreement: **{diagonal:,} / {total:,} ({100*diagonal/total:.2f}%)**. The following intervals are calculated from the saved counts; precision and F1 also use these counts rather than rounded JSON rates.\n\n'
        text+=table(['Requested class','Diagonal / requested','Agreement','Wilson 95% interval','Proxy precision','Proxy F1'],rows)
        text+='\n\nConfusion counts, with requested classes in rows and heuristic-assigned classes in columns:\n\n'
        text+=table(['Requested / assigned']+order,[[name]+matrix[i] for i,name in enumerate(order)])
        if n==6:
            rows=[]
            for name,r in data['classes'].items():
                if name!='non_cracked':rows.append([name,pct(r['mechanism_adherence_rate']),pct(r['spatial_trajectory_recall']),r['trajectory_mae_px'],r['trajectory_hausdorff95_px'],r['taper_error_mae']])
            text+='\n\nThe next table preserves the **separate target-dependent predicate and command-correspondence diagnostics**. Its acceptance column is not the diagonal of the confusion matrix.\n\n'
            text+=table(['Requested class','Target predicate acceptance','Trajectory recall','Legacy trajectory MAE px','Pooled distance 95th percentile px','Taper error ratio units'],rows)
        return text
    classes=data['classes'] if n==5 else {k:v for k,v in data.items() if isinstance(v,dict) and 'sample_count' in v}
    rows=[]
    for name,r in classes.items():
        count=r.get('samples',r.get('sample_count'))
        rate=next(r[k] for k in ('mechanism_adherence_rate','clean_surface_rate','orientation_alignment_accuracy','mesh_detection_rate') if k in r)
        k=round(rate*count)
        rows.append([name,count,f'{k} / {count}',pct(rate),wilson(k,count),r.get('mean_continuity_ratio',r.get('mean_c_ratio','Not reported'))])
    return table(['Requested class','Images','Accepted count','Predicate acceptance','Wilson 95% interval','Continuity proxy'],rows)+'\n\nCounts are arithmetic reconstructions from the saved rates and denominators. These class-specific predicates are not an independently assigned multiclass confusion matrix.'


def theory(n):
    text=r'''For normalized image $x_0$, the implemented forward process uses $T=1000$ and a linear variance schedule from $10^{-4}$ to $0.02$. With $\alpha_t=1-\beta_t$ and $\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s$:

$$x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I).$$

The denoiser minimizes noise-prediction mean squared error (the expression suppresses constant normalization by the number of image elements):

$$\mathcal{L}=\mathbb{E}_{x_0,t,\epsilon}\left[\lVert\epsilon-\epsilon_\theta(x_t,t,c)\rVert_2^2\right].$$

The condition $c$ is omitted for the unconditional model. This objective is implemented in the diffusion and training modules and follows the DDPM formulation [R1]. Training MSE is an optimization diagnostic, not a perceptual, topology or physical-validity score.
'''
    if n>=4:
        text+=r'''
The historical fast sampler uses a 50-step deterministic DDIM-style update [R2]. At the reported guidance scale $w=2.5$, it combines conditional and null predictions as

$$\hat{\epsilon}=\epsilon_{\mathrm{null}}+w\left(\epsilon_{\mathrm{cond}}-\epsilon_{\mathrm{null}}\right).$$

Classifier-free guidance [R3] changes the sampling trade-off. Finite steps, clipping, image transforms and device-dependent arithmetic preclude claiming exact invertibility or universal bitwise reproducibility. The historical sampler's implementation should be retained when reproducing historical artifacts; the prospective sampler is a separately documented protocol.
'''
    if n==6:
        text+=r'''
Spatial conditioning concatenates noisy RGB with two fields: $u_t=[x_t,S_0,S_1]$. Training fields come from the input image itself. Local-darkness thresholding uses a 21-pixel window and threshold 15.5, with a 2 by 2 opening and a lower-threshold fallback. Distance ridges are smoothed with Gaussian $\sigma=1.1$; aperture is $S_1=\operatorname{clip}(\operatorname{EDT}(M)/6,0,1)$. Fields are then mapped to $[-1,1]$ for network input.

Inference fields instead come from procedural templates: perturbed near-vertical paths, smoothly blended vertical/diagonal paths, single or multiple diagonals, horizontal paths, and warped nearest-seed cell boundaries. Their widths and angles are sampled in pixel units. The implementation does not consume measured loads, calibrated material parameters or structural boundary conditions; its function name does not establish a validated mechanics solver.

Chromatic normalization strongly desaturates each pixel toward $L=0.299R+0.587G+0.114B$, using $R'=L+0.08(R-L)+1.5$, $G'=L+0.08(G-L)+0.5$, and $B'=L+0.08(B-L)-1.5$, followed by clipping and integer conversion. This transform is a declared image prior, not recovered camera calibration or guaranteed exact luminance preservation.
'''
    if n>=5:
        text+='\nHistorical curation and evaluation use different rule sets. Curation first tests torsion-like multiplicity (at least two elongated diagonal components, each spanning at least 38 pixels, elongation at least 2.2, within 24 degrees of parallel, with component concentration 0.25–0.85), then map-like dispersion (at least four significant components and concentration below 0.48). Significant components require at least 30 pixels and span 25 pixels. Other classes require a dominant span of at least 55 pixels and concentration at least 0.52. The curved branch requires piecewise angular change at least 16 degrees, span at least 65 pixels and a vertical/diagonal half-patch transition. Near-vertical and horizontal branches use 21-degree tolerance and curvature below 20 degrees; horizontal and remaining diagonal branches require span at least 60 pixels. Branch order matters. These numerical cutoffs operationalize appearance labels, not independently identified physical causes.\n'
    return text


def protocol(n):
    if n<=3:
        size=64 if n<3 else 128
        return f'''The primary morphology source is `src/{EVALUATORS[n-1]}`. At {size} pixels it applies local-background subtraction with window {11 if n<3 else 17}, darkness threshold {13.5 if n<3 else 15.0}, minimum component area {6 if n<3 else 12} pixels and nominal span threshold {11 if n<3 else 22} pixels. The 128-pixel path additionally applies a 2 by 2 opening. Component counts, image-axis orientation and local contrast are detector outputs, not physical measurements.

The later fidelity diagnostic uses torchvision Inception-v3 with its classifier removed, 299-pixel resizing and ImageNet normalization. It selects training-manifest references: 160 real and 160 generated images for the stored score. KID uses 100 subsets of up to 128 feature vectors and a cubic polynomial kernel. This custom feature/preprocessing protocol is not interchangeable with a standard torch-fidelity run or the earlier torchmetrics feature-64 scores. The KID standard deviation describes subset variability, not a confidence interval over independent training runs.
'''
    if n==4:
        return '''The morphology evaluator subtracts a 15-pixel local grayscale mean and thresholds at 16 intensity units. Components, their bounding-box diagonal, principal orientation and component concentration determine the predicates. The five groups contain 16 saved images each. A map-rule success can be triggered by dispersed components rather than a verified crack network. The separate FID/KID artifact uses 80 generated and 80 reference images; no independently held-out evaluation lineage is established for that historical number.'''
    if n==5:
        return '''The evaluator uses local-darkness mask extraction (23-pixel window, threshold 24.5 and 2 by 2 opening), then component geometry. Flexural acceptance requires detected presence and a near-vertical principal orientation; flexure-shear requires presence and piecewise angular change of at least 14 degrees; web-shear and splitting use diagonal/horizontal orientation predicates. Torsion accepts at least two diagonal or significant components, and map acceptance uses at least three components and concentration below 0.65, without the same presence gate. These predicates explicitly depend on the requested class. Historical torch-fidelity FID/KID selects up to 16 training-reference patches per class and uses eight KID subsets of size 12. It is a training-reference diagnostic, not an unseen-data validation.'''
    return r'''The corrected evaluator predicts each class from image morphology alone, independent of the requested class and commanded field. Its fixed local-darkness mask uses window 23, threshold 24.5 and 2 by 2 opening. A second, requested-class-dependent acceptance predicate remains a separate descriptive quantity.

Command core pixels are $S_0>0.22$. Historical trajectory recall measures the fraction of command-core pixels covered after four dilation iterations of the detected mask. Distance statistics pool both command-to-detection and detection-to-command nearest-pixel distances; the stored 95th percentile is that pooled percentile, not a demonstrated physical error bound. The legacy evaluator substitutes 50 pixels for missing-detection MAE/RMSE and 80 pixels for its distance percentile. Such penalties must be identified rather than treated as measured distances. The later paired-field audit instead reports missing counts explicitly and leaves conditional distance undefined when no detection exists.

No new FID/KID score was requested in the corrected 1,400-image run (`fidelity_status: not_requested`). Earlier fidelity claims from different report versions must not be substituted into that run. New research fidelity code requires explicit reference selection and shared preprocessing, verifies manifest and pixel overlap, and refuses scientific scoring of smoke outputs.'''


def validation(n):
    current=read_json(ROOT/'datasets/current_research_split.json')
    counts=current['datasets'][DATASETS[n-1]]
    text='The prospective **research_v5** manifests contain the following base partition counts. They were constructed after the historical training and do not retroactively make its checkpoints unseen-data models.\n\n'
    text+=table(['Partition','Base records'],[[s,f'{counts[s]:,}'] for s in ('train','validation','test')])
    text+='\n\nThe shared source audit covers 67,264 originals, with 65,866 distinct decoded RGB images and 1,398 extra exact decoded copies. The v4 perceptual screen found 386 cross-split candidates. V5 conservatively links all 1,111 candidate pairs, including within-split pairs, before assignment; the same pHash ≤3 and dHash ≤6 search then finds zero cross-split candidates. This limited, uncalibrated screen can miss crops, rotations, flips and larger edits. Source names and zero screened matches do not prove site or structure independence.\n'
    if n>=3:
        text+='\nRC attribution identifies a matching Mendeley record [R9] and its requested paper [R10]. Local ZIP sizes/CRCs match extracted files, but no publisher digest established local archive identity. All 10,519 local JSON files contain crack rectangles; these boxes do not validate image-level labels, pixel masks or structural mechanisms. The geometry audit flags 1,145 source annotations, including 1,122 out-of-bounds cases and 150 image/annotation dimension mismatches; categories overlap.\n'
    if n>=5:
        q=current['quality_views'][DATASETS[n-1]]['split_counts']
        text+='\nThe prospective `strict_rc_geometry_v1` view excludes 481 affected Exp5/6 records (371 train, 67 validation, 43 test), retains all seven proxy classes and preserves v5 component assignments. Exclusion is conservative; it does not prove every excluded label wrong or every retained label correct.\n\n'
        text+=table(['Partition','Qualified records'],[[s,f'{q[s]:,}'] for s in ('train','validation','test')])
    text+='\n\nScratch training, EMA selection on fixed-noise validation, provenance-checked generation and geometry/fidelity preflight are implemented for the new lineage. Their bounded smoke runs are infrastructure checks. Full clean multi-seed training, independently labeled generated images, external validation and downstream detector utility have not been established by the historical artifacts. A prepared campaign is not a completed experiment.\n'
    fresh=ROOT/'experiments/fresh_install_cpu_verification.json'
    if fresh.exists():
        f=read_json(fresh)
        if f.get('fresh_install_verified') and len(f.get('inference',[]))==6:
            text+='\nComputational sharing was tested in a fresh Windows/Python 3.10 CPU environment: the public weights were downloaded and verified and one image per experiment was generated without raw datasets. This confirms that inference can execute under that environment; it does not reproduce model training or validate image quality.\n'
    if n==6:
        d=read_json(ROOT/'experiments/exp6_level_c_spatial_physics/detector_calibration_v1/summary.json')
        text+='\n### Detector validation and exploratory calibration\n\nThe fixed detector was compared with pixel masks on 51 DeepCrack originals (102 nonoverlapping native tiles). Its pixel precision/recall/IoU/Dice were 33.69% / 58.81% / 27.26% / 42.84%. A 24-setting validation-only search on 46 originals selected window 31, threshold 30 and opening 2. On the archived 51-original test selection, precision/recall/IoU/Dice became 42.36% / 61.25% / 33.41% / 50.08%. The original-image cluster-bootstrap paired IoU gain was 6.15 percentage points (95% interval 5.04–7.31). These selections belong to archived research_v1, and test results had previously been inspected, so this is exploratory detector evidence, not a new untouched test of the generator or generated-image mechanism labels.\n\n'
        a=read_json(ROOT/'experiments/exp6_level_c_spatial_physics/ablations/paired-seven-classes-three-seeds-v2/audited_summary.json')
        rows=[]
        for detector,conditions in a['summary'].items():
            for mode,r in conditions.items():rows.append([detector,mode,r['n'],pct(r['mean_recall']),r['no_detection'],'Undefined' if r['conditional_mae_px'] is None else f"{r['conditional_mae_px']:.3f}"])
        text+='### Paired field sensitivity pilot\n\nThe pilot contains 126 images: seven classes, three seeds, two commands per class/seed and three field conditions. Cracked-image summaries use 36 images per condition. Initial random states were matched and non-cracked controls were byte-identical. Outputs are evaluated against the original correct command.\n\n'
        text+=table(['Detector','Input field','Cracked images','Trajectory recall','Missing detections','Conditional MAE px'],rows)
        text+='\n\nThis supports input-field sensitivity for the historical checkpoint. The zero-field/present-class combination may be out of distribution because historical dropout removed both together. It is not a separately trained matched class-only baseline and cannot establish physical causation. Distances must be read alongside missing-detection counts.\n'
    return text


REFERENCES='''[R1] Ho, Jain and Abbeel (2020). Denoising Diffusion Probabilistic Models. https://arxiv.org/abs/2006.11239

[R2] Song, Meng and Ermon (2020; ICLR 2021). Denoising Diffusion Implicit Models. https://arxiv.org/abs/2010.02502

[R3] Ho and Salimans (2022). Classifier-Free Diffusion Guidance. https://arxiv.org/abs/2207.12598

[R4] Heusel et al. (2017). FID proposal in GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. https://arxiv.org/abs/1706.08500

[R5] Bińkowski et al. (2018). Demystifying MMD GANs; KID proposal. https://arxiv.org/abs/1801.01401

[R6] SDNET2018 dataset and acquisition documentation. https://digitalcommons.usu.edu/all_datasets/48/

[R7] DeepCrack source repository, masks and dataset splits. https://github.com/yhlleo/DeepCrack

[R8] CrackForest source repository and citation. https://github.com/cuilimeng/CrackForest-dataset

[R9] Wang (2026). Reinforced concrete structure crack detection dataset 10519, Mendeley Data v1. DOI 10.17632/3ghxck2wd5.1. https://data.mendeley.com/datasets/3ghxck2wd5/1

[R10] Wang et al. (2024). Automatic detection of building surface cracks using UAV and deep learning-combined approach. Structural Concrete 25(4), 2302–2322. https://doi.org/10.1002/suco.202300937

[R11] Zhang et al. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric; LPIPS. https://arxiv.org/abs/1801.03924
'''


def build():
    import torch
    manifest=read_json(ROOT/'models/checkpoint_manifest.json')
    spec=importlib.util.spec_from_file_location('report_web_examples',ROOT/'website/build_experiment_pages.py')
    examples=importlib.util.module_from_spec(spec);spec.loader.exec_module(examples)
    summary=[]
    for n,name in enumerate(NAMES,1):
        folder=ROOT/'experiments'/name;archive=preserve(folder,n);narrative=NARRATIVE[n]
        source=folder/('evaluation_corrected/EVALUATION_METRICS.json' if n==6 else 'EVALUATION_METRICS.json')
        data=read_json(source);log=training(folder);csv_path,rows=dataset_rows(n)
        checkpoint=ROOT/manifest['checkpoints'][n-1]['path']
        if sha(checkpoint)!=manifest['checkpoints'][n-1]['sha256']:raise ValueError('Checkpoint manifest mismatch')
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        parameters=sum(t.numel() for t in state['model'].values())
        if parameters!=PARAMETERS[n-1]:raise ValueError('Unexpected checkpoint architecture size')
        optimizer=[{k:v for k,v in g.items() if k!='params'} for g in state.get('optimizer',{}).get('param_groups',[])]
        checkpoint_meta=dict(path=str(checkpoint.relative_to(ROOT)),sha256=sha(checkpoint),parameters=parameters,epoch_field=state.get('epoch'),loss=state.get('loss',state.get('mean_loss')),optimizer_groups=optimizer)
        del state
        inputs=[source,log['path'],csv_path,folder/f'plots/exp{n}_overview.png',ROOT/'src'/TRAINERS[n-1],ROOT/'src'/EVALUATORS[n-1],ROOT/'src/diffusion.py',ROOT/'models/checkpoint_manifest.json',ROOT/'datasets/current_research_split.json',ROOT/'datasets/source_content_v1/summary.json',ROOT/'datasets/near_duplicate_screen_v1/research_v5_summary.json']
        if (ROOT/'experiments/fresh_install_cpu_verification.json').exists():inputs+=[ROOT/'experiments/fresh_install_cpu_verification.json']
        if n>=3:inputs+=[ROOT/'datasets/source_provenance/rc10519_local_inventory.json',ROOT/'datasets/source_provenance/rc10519_publisher_record.json']
        if n>=5:inputs+=[ROOT/'datasets/exp5_exp6_level_bc_mechanisms_256/research_v5/strict_rc_geometry_v1/audit.json']
        if n==6:inputs+=[ROOT/'src/physics_trajectory_solver.py',folder/'detector_validation/summary.json',folder/'detector_calibration_v1/summary.json',folder/'ablations/paired-seven-classes-three-seeds-v2/audited_summary.json',folder/'report_archive/20261122-original/original_stored_metrics.json']
        sample_log=folder/('evaluation_corrected/PER_SAMPLE_ERROR_LOG.csv' if n==6 else 'PER_SAMPLE_ERROR_LOG.csv')
        if sample_log.exists():inputs+=[sample_log]
        inputs+=[ROOT/'website'/path for _,path in examples.result_images[n]]
        evidence=dict(experiment=n,revision=REVISION,report=f'EXPERIMENT_{n}_FULL_REPORT.md',checkpoint=checkpoint_meta,
                      input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},archived_sources=archive['entries'],
                      input_lf_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in inputs if p.suffix in ('.json','.csv','.py','.md')},
                      hash_policy='Raw hashes attest observed bytes; LF-normalized hashes additionally identify text content across Git line-ending conversions. Archived originals retain exact bytes.',
                      training=dict(unique_epochs=len(log['rows']),raw_numeric_rows=log['raw_numeric_rows'],repeated_epochs=log['repeated_epochs'],ignored_rows=log['ignored_rows'],final=log['final'],minimum=log['minimum']),
                      limitations=['Retrospective artifact analysis, not new clean training','Requested labels and morphology predicates are not independent engineering ground truth'])
        evidence_path=folder/'report_evidence.json';evidence_path.write_text(json.dumps(evidence,indent=2),encoding='utf-8')
        resolution=64 if n<3 else 128 if n==3 else 256
        classes=1 if n==1 else 2 if n<4 else 5 if n==4 else 7
        source_counts=Counter(r.get('source','SDNET2018') for r in rows)
        class_counts=Counter(r.get('class_name',r.get('label','unknown')) for r in rows)
        lr=optimizer[0]['lr'] if optimizer else 2e-4
        tech=[['Image shape',f'{resolution} by {resolution} RGB'],['Denoiser','SmallUNet' if n<4 else 'AttentionUNet256'],['Checkpoint tensor parameters',f'{parameters:,}'],['Image / conditioning channels','3 RGB; no class input' if n==1 else '3 RGB + class embedding' if n<6 else '3 RGB + 2 spatial fields + class embedding'],['Requested categories',classes],['Completed logged epochs',len(log['rows'])],['Final logged noise MSE',f"{log['final']['loss']:.9f}"],['Minimum logged training MSE',f"{log['minimum']['loss']:.9f} at local epoch {log['minimum']['epoch']}"],['Learning rate',f'{lr:g}'+(' (checkpoint optimizer)' if optimizer else ' (trainer source default; optimizer not saved)')],['Batch size provenance',f"Trainer default {[64,32,8,6,6,6][n-1]}; historical command-line arguments not fully captured"],['Software','Python, PyTorch, torchvision, NumPy, Pillow, SciPy; Matplotlib and metric dependencies for evaluation']]
        tech.extend([['Optimizer','AdamW; beta values 0.9 and 0.999; weight decay 0.01 (saved optimizer where available, otherwise source defaults)'],['Gradient norm clipping','1.0 in trainer source'],['Precision policy','Current historical trainer enables CUDA autocast and GradScaler; exact historical execution precision is not independently logged'],['Class dropout','Not used in the binary/unconditional source trainers' if n<4 else '0.10 in Exp4 source' if n==4 else '0.15 in Exp5 source' if n==5 else '0.15 joint class/spatial dropout'],['Sampling','1,000-step ancestral DDPM for the shared pretrained reproduction path' if n<4 else 'Historical 50-step DDIM-style CFG, reported guidance 2.5'],['Checkpoint selection','Released phase-final checkpoint; no independently held-out selection record'],['Augmentation','Image normalization; external data addition in Exp3, not a verified geometric augmentation benefit' if n<4 else 'Horizontal and vertical flips in Exp4 source' if n==4 else 'Horizontal flips; vertical flips prohibited in the historical Exp5/6 source']])
        text=f'# Experiment {n} Research Report\n\n## {TITLES[n-1]}\n\nRevision: {REVISION}. Study type: retrospective computational experiment. Historical checkpoint and saved evaluation; prospective validation is identified separately.\n\n## Abstract\n\n{narrative["abstract"]}\n\n## 1. Aim and research question\n\n{narrative["aim"]}\n\n{narrative["hypothesis"]}\n\n## 2. Technical stack and model specification\n\n'+table(['Item','Specification and provenance'],tech)
        text+='\n\nEpoch numbering in checkpoints is zero-based for Experiments 1–3 and one-based for the later phases; the training log is used to establish the completed phase length. A low training loss is not evidence of held-out generalization. Exact historical CUDA/PyTorch versions, CLI arguments and original peak resource measurements are not fully captured, so later environment checks are not substituted for historical runtime provenance.\n' if n<=3 else '\n\nThe parameter count is verified from the saved state dictionary. Attention is implemented through scaled-dot-product attention; the selected backend depends on runtime and hardware, so calling the implementation FlashAttention does not prove that a particular kernel executed. Historical environment/CLI provenance is incomplete. Phase-local epochs and training loss are not a controlled comparison with other experiments.\n'
        text+='\n## 3. Dataset composition and provenance\n\n'+table(['Historical source','Training-manifest records'],[[k,f'{v:,}'] for k,v in source_counts.items()])+'\n\n'+table(['Historical label','Records'],[[k,f'{v:,}'] for k,v in class_counts.items()])
        text+=f'\n\nTotal historical training records: **{len(rows):,}**. Source: {link(csv_path)}. These are processed patches, not counts of independent structures or photographs. Labels for the morphology/mechanism-inspired experiments are derived by image-geometry rules. The historical train/reference design and transfer ancestry do not establish independent holdout performance. Source image identities, alias grouping and current qualified manifests are discussed in Section 10.\n'
        text+='\nDataset references: SDNET2018 [R6]'+('; DeepCrack [R7]' if n>=4 else '')+('; CrackForest [R8]' if n>=5 else '')+('; matching RC collection record [R9]' if n>=3 else '')+'.\n'
        text+='\n## 4. Mathematical and implementation basis\n\n'+theory(n)
        text+='\n## 5. Experimental procedure\n\n'+'\n\n'.join(f'{i}. {p}' for i,p in enumerate(narrative['procedure'],1))
        text+='\n\nThese steps reconstruct the procedure supported by saved artifacts and current source. Defaults are distinguished from observed checkpoint/log values; a missing historical execution configuration is not silently reconstructed as a verified run command.\n'
        text+='\n## 6. Evaluation protocol and statistical interpretation\n\n'+protocol(n)
        text+='\nWilson 95% intervals summarize binomial counting uncertainty conditional on the saved generator and heuristic. They do not include training-seed variability, detector bias, source clustering or uncertainty about real mechanisms. Precision, recall and F1 computed against requested conditions quantify heuristic correspondence, not human-labeled classification accuracy. Pixel distances are in image coordinates; no physical millimetre scale is available.\n'
        text+=r'''
For $k$ accepted outputs among $n$ requests, $\hat p=k/n$ and $z=1.95996$. The Wilson interval is

$$\frac{\hat p+z^2/(2n)\ \pm\ z\sqrt{\hat p(1-\hat p)/n+z^2/(4n^2)}}{1+z^2/n}.$$
'''
        if n in (2,3,6):
            text+=r'''
For saved count matrix $A$, $A_{ij}$ counts request $i$ assigned heuristic class $j$. Row agreement is $A_{ii}/\sum_j A_{ij}$; proxy precision is $A_{ii}/\sum_j A_{ji}$; $F_1=2PR/(P+R)$, with zero used for undefined zero-denominator cases. These definitions explain the table without changing the meaning of requested labels into ground truth.
'''
        text+='\n## 7. Results\n\n'+result_tables(n,data)
        fidelity=data.get('fidelity') if n!=4 else read_json(folder/'FID_KID_METRICS.json')
        if fidelity:
            text+='\n\n### Protocol-qualified feature diagnostic\n\n'+table(['Saved metric','Value'],[[k,v] for k,v in fidelity.items()])+'\n\nFID and KID summarize feature distributions [R4, R5]; they do not certify crack presence or physical validity. Reference exposure, sample count, feature extraction and image preprocessing are part of the metric definition. Do not rank this value against experiments using another implementation or a different reference distribution.\n'
        elif n==6:text+='\n\n### Fidelity status\n\nThe corrected primary run did not compute FID/KID. No fidelity number is inferred from another report version.\n'
        if n==6:
            legacy=read_json(folder/'report_archive/20261122-original/original_stored_metrics.json').get('fidelity',{})
            if legacy:
                text+='\n\n### Separately stored legacy fidelity record\n\nThe pre-revision metric artifact also stores the following fidelity values. They are retained as a separate historical diagnostic, not as newly computed results of the corrected classification run.\n\n'+table(['Legacy metric','Stored value'],list(legacy.items()))
                text+='\n\nThe legacy evaluator uses torchvision Inception features after 299-pixel resizing and ImageNet normalization, with a chromatic transform applied to real references but not to generated images. That asymmetric preprocessing, training-reference exposure and the absence of a complete frozen reference manifest limit interpretation. The previous comparison with an N=112 score also changes sample budget; a lower FID at N=1,400 does not isolate an improvement in model quality. The prospective fidelity implementation instead applies the declared transform consistently to both sides. The exact earlier artifact is retained at '+link(folder/'report_archive/20261122-original/original_stored_metrics.json')+'.\n'
        if n<=2:
            text+='\n\nEarlier report values belong to `src/evaluate_metrics.py`, which uses torchmetrics feature dimension 64 after resizing inputs to 64 pixels and index-paired AlexNet LPIPS [R11]. The later primary error-analysis score uses a different feature/preprocessing/reference path. Index-paired LPIPS on unpaired generated and real images is a distance diagnostic, not reconstruction fidelity. Earlier low FID values must not be combined with the later protocol as one trend. The pre-revision record is retained in the report archive.\n'
            earlier=[['Unconditional',5.604571,3.419779,1.028021,0.548679]] if n==1 else [['Non-cracked request',0.724640,0.756606,0.389974,0.416435],['Cracked request',0.701002,0.556873,0.202802,0.425783]]
            text+='\n\nEarlier report values, quoted as recorded rather than recomputed:\n\n'+table(['Generated group','Feature-64 FID','Feature-64 KID mean','KID subset SD','Index-paired LPIPS'],earlier)+'\n\nThese values belong to the earlier technical record; original sample identities and execution configuration are incompletely captured. They are not substituted for the later primary diagnostic.\n'
        text+=f'\n\n### Saved plots and generated examples\n\n![Experiment {n} training and saved evaluation overview](plots/exp{n}_overview.png)\n\nFigure interpretation: phase-local training loss, detector-defined rates and saved sample counts. This figure is a descriptive summary, not a controlled leaderboard or an independently labeled accuracy assessment.\n'
        if n==1:text+='\nThe raw training log contains an additional numeric epoch-1 record and an embedded header. The current report uses the last numeric record for each epoch; the source log and all overwritten report versions are preserved.\n'
        text+='\n### Archived illustrative outputs\n\nThe following saved illustrations retain their available filenames and request labels. Their membership in the quantified collection is not assumed without an original sampling manifest. They are not a blinded quality study, and their labels are not independent human judgments. The quantitative tables retain the complete stated evaluation denominators.\n'
        for label,path in examples.result_images[n]:
            text+=f'\n![Archived illustrative {label}](../../website/{path})\n\n{label}. Source illustration: `{Path(path).name}`.\n'
        text+='\n## 8. Observations and discussion\n\n'+narrative['interpretation']
        text+='\n\nThis experiment changes several design factors relative to other phases. Resolution, training images, label taxonomy, transfer ancestry, color processing, sampler and evaluator may differ. A before/after visual comparison is therefore hypothesis-generating; no causal attribution to a single architectural or conditioning change is made.\n'
        text+='\n## 9. Assumptions and their consequences\n\n'+table(['Assumption','Implementation or rationale','Consequence and validation status'],narrative['assumptions'])
        text+='\n\nAll image-derived labels assume that visible dark geometry is informative about the intended visual attribute. Real masks, bounding boxes and class labels answer different annotation questions. Additional annotations or a mechanics-inspired name do not independently verify a failure mechanism.\n'
        if n>=5:text+='\nMaterial and member context is an additional unverified assumption: the multi-source collection includes road-crack imagery [R8], and the available patch metadata do not establish concrete material, reinforcement layout or member/load history for every record. The report therefore treats the labels as appearance proxies rather than material-specific diagnoses.\n'
        text+='\n## 10. Completed audits and prospective validation\n\n'+validation(n)
        text+='\n## 11. Sources of error and errors identified\n\n'+table(['Identified issue','Correction or interpretation'],narrative['errors'])
        text+='\n\nFurther threats include segmentation sensitivity to texture and illumination; source overlap and patch clustering; heuristic circularity between curation and evaluation; small or unequal sample budgets; image-derived versus procedurally generated conditioning; reference preprocessing differences; and incomplete historical execution metadata. No structural safety, real crack width, downstream detection benefit, state-of-the-art performance or physical-cause accuracy is claimed from these measurements.\n'
        text+='\n## 12. Conclusion\n\n'+narrative['conclusion']
        category='all' if n==1 else 'cracked' if n<4 else 'all'
        text+=f'\n\n## 13. Reproducibility and artifact record\n\nThe released historical weights are used for inference; these commands do not start training. Run from the repository root with the documented dependencies and a fresh output directory.\n\n```powershell\n.\\.venv\\Scripts\\python.exe pipeline/download_models.py\n.\\.venv\\Scripts\\python.exe pipeline/checkpoints.py verify\n.\\.venv\\Scripts\\python.exe pipeline/generate_cracks.py --experiment {n} --crack-type {category} --count 1 --seed 2026 --output-dir experiments/{name}/report_replication_seed2026\n```\n\nA fixed seed requires matching checkpoints, sampling settings, device and software for comparable execution; it does not guarantee cross-device pixel identity or exact replication of a historical example grid. Research training uses a separate scratch lineage and must not overwrite these final weights.\n\n'
        text+=table(['Evidence','Artifact'],[['Primary metric record',link(source)],['Training log',link(log['path'])],['Checkpoint SHA256',checkpoint_meta['sha256']],['Input hashes and revision archive',link(evidence_path)],['Training source',link(ROOT/'src'/TRAINERS[n-1])],['Evaluation source',link(ROOT/'src'/EVALUATORS[n-1])]])
        text+='\n\nCode is available in the [project repository](https://github.com/AlgoRubic/Crack-Image-Generation). The six historical checkpoints are distributed in the [versioned model release](https://github.com/AlgoRubic/Crack-Image-Generation/releases/tag/historical-models-v1), with a checksum manifest. Raw datasets are not included in that inference bundle. Dataset references identify their publishers, while local source inventories record observed archive hashes and outstanding identity/annotation limitations.\n'
        text+='\n\nThe machine-readable evidence file records input SHA256 values, checkpoint metadata, log handling and exact archived pre-revision sources. The historical reports remain recoverable and are superseded where their claims conflict with the corrected tables above. Original sampling files do not provide a complete checkpoint-hash chain for every historical evaluation image; association with the released phase checkpoint follows the available procedural record. No new generation, training or rerun of image-evaluation algorithms is implied by this revision. Counts, Wilson intervals and derived matrix statistics are calculated from saved artifacts.\n'
        selected={1,4,5,6}
        if n>=4:selected.update((2,3,7))
        if n>=3:selected.update((9,10))
        if n>=5:selected.add(8)
        if n<=2:selected.add(11)
        refs=[r for r in REFERENCES.strip().split('\n\n') if int(re.match(r'\[R(\d+)\]',r)[1]) in selected]
        refs=[re.sub(r'(https://\S+)$',r'[Source](\1)',r) for r in refs]
        text+='\n## 14. References\n\n'+'\n\n'.join(refs)+'\n'
        target=folder/f'EXPERIMENT_{n}_FULL_REPORT.md';target.write_text(text,encoding='utf-8')
        if n in (5,6):
            prefix='A' if n==5 else 'C'
            note=f'# Experiment {n} Assumptions and Implementation Record\n\nRevision: {REVISION}. This current ledger supersedes unsupported interpretations in the exact archived pre-revision ledger. The full research report provides the methods, results, uncertainty and source evidence.\n\n'
            for i,(title,rationale,status) in enumerate(narrative['assumptions'],1):
                note+=f'## {title}\n\nImplementation or rationale: {rationale}\n\nInterpretation and validation status: {status}\n\n'
            note+='## Evaluation rules and observed consequences\n\n'+protocol(n)+'\n\n## Completed evidence and remaining validation\n\n'+validation(n)+'\n\n## Errors identified in earlier interpretations\n\n'+table(['Earlier issue','Current interpretation'],narrative['errors'])
            note+=f'\n\n## Evidence and version history\n\nRead [the full research report](EXPERIMENT_{n}_FULL_REPORT.md), [input signatures](report_evidence.json), and [archive index](report_archive/20261122-original/archive_index.json). Original statements are retained verbatim in the archive, including the full prior engineering rationale; they are historical assertions, not validated physical laws inferred from these images.\n'
            (folder/f'EXPERIMENT_{n}_DOCUMENTATION_AND_ASSUMPTIONS.md').write_text(note,encoding='utf-8')
        if n==4:
            (folder/'CRACK_CONTINUITY_ANALYSIS_REPORT.md').write_text('# Experiment 4 Continuity Analysis\n\nRevision: '+REVISION+'\n\nThe saved linear-class continuity ratios are 0.0676 (longitudinal), 0.0553 (transverse) and 0.0744 (diagonal); significant component means are 78.81, 84.06 and 70.06. These values do not establish resolved fragmentation. The older Exp3-upscaled comparison used another image/detector protocol and is not a matched architecture ablation.\n\n'+narrative['interpretation']+'\n\nRead [the complete revised report](EXPERIMENT_4_FULL_REPORT.md) for methods, assumptions, source errors, figures and prospective validation. The complete original continuity report is preserved under `report_archive/20261122-original/`.\n',encoding='utf-8')
        if n in (2,3):
            status=f'# Experiment {n} Current Research Record\n\nRevision: {REVISION}. Historical training is complete and a later saved descriptive evaluation exists: '+('2,000 generated images (1,000 per class).' if n==2 else '400 generated images (200 per class). The earlier closed-run statement that no final metrics existed is superseded by this later evaluation.')+f'\n\nRead [the complete research report](EXPERIMENT_{n}_FULL_REPORT.md). These are heuristic requested-label agreement and protocol-specific feature diagnostics, not independent mechanism accuracy or clean holdout validation. Earlier status and PDF/web technical records are preserved in `report_archive/20261122-original/`.\n'
            (folder/'STATUS.md').write_text(status,encoding='utf-8')
        summary.append(dict(experiment=n,report=str(target.relative_to(ROOT)),sha256=sha(target),words=len(text.split()),input_files=len(inputs),archive_files=len(archive['entries']),corrections=narrative['errors']))
        print(f'Wrote Experiment {n}: {len(text.split())} words',flush=True)
    (ROOT/'experiments/report_revision_audit.json').write_text(json.dumps(dict(revision=REVISION,reports=summary,scope='Retrospective research documentation; scientific validation gaps remain explicit'),indent=2),encoding='utf-8')


if __name__=='__main__':build()
