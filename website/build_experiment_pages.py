from pathlib import Path
import html
import re

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent / 'concrete_ddpm'

def markdown_to_html(text: str) -> str:
    lines = text.splitlines()
    out, in_ul, in_ol, in_code, in_table = [], False, False, False, False
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        if line.startswith("````"):
            i += 1
            continue
        if line.startswith("```"):
            if in_code:
                out.append("</code></pre>"); in_code = False
            else:
                out.append("<pre><code>"); in_code = True
            i += 1
            continue
        if in_code:
            out.append(html.escape(line)); i += 1; continue
        if in_table and not (line.startswith("|") and line.count("|") >= 2):
            out.append("</tbody></table>"); in_table = False
        if line.startswith("|") and line.count("|") >= 2:
            cells = split_table_cells(line)
            if not in_table and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{2,}", lines[i + 1]):
                out.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells) + "</tr></thead><tbody>")
                in_table = True; i += 2; continue
            if in_table:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
                i += 1; continue
        list_line = line.lstrip()
        if list_line.startswith("- ") or list_line.startswith("* "):
            if in_ol: out.append("</ol>"); in_ol = False
            if not in_ul: out.append("<ul>"); in_ul = True
            out.append(f"<li>{inline(list_line[2:])}</li>"); i += 1; continue
        if in_ul:
            out.append("</ul>"); in_ul = False
        om = re.match(r"^\d+[.]\s+(.*)$", list_line)
        if om:
            if not in_ol: out.append("<ol>"); in_ol = True
            out.append(f"<li>{inline(om.group(1))}</li>"); i += 1; continue
        if in_ol:
            out.append("</ol>"); in_ol = False
        if line in {"---", "***", "___"}:
            out.append("<hr>"); i += 1; continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level = len(m.group(1)); out.append(f"<h{level+1}>{inline(m.group(2))}</h{level+1}>"); i += 1; continue
        if not line:
            i += 1; continue
        if line.startswith("<!--") and line.endswith("-->"):
            i += 1; continue
        if line.startswith(">"):
            out.append(f"<blockquote>{inline(line[1:].strip())}</blockquote>"); i += 1; continue
        out.append(f"<p>{inline(line)}</p>")
        i += 1
    if in_table: out.append("</tbody></table>")
    if in_ul: out.append("</ul>")
    if in_ol: out.append("</ol>")
    if in_code: out.append("</code></pre>")
    return "\n".join(out)

def inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", lambda m: f'<img src="{image_src(m.group(2))}" alt="{m.group(1)}">', s)
    def render_link(match):
        url=html.unescape(match.group(2))
        if url.startswith(('https://','http://')):
            return f'<a href="{html.escape(url,quote=True)}">{match.group(1)}</a>'
        return f'<span class="artifact-link">{match.group(1)}</span>'
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", render_link, s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    return s

def split_table_cells(line: str):
    body = line.strip().strip("|")
    cells, current, in_math = [], [], False
    for ch in body:
        if ch == "$":
            in_math = not in_math
        if ch == "|" and not in_math:
            cells.append("".join(current).strip()); current = []
        else:
            current.append(ch)
    cells.append("".join(current).strip())
    return cells

def image_src(url: str) -> str:
    name = Path(url).name
    if (ROOT/'assets/research'/name).is_file():return 'assets/research/'+name
    aliases = {
        "exp5_non_cracked_256_grid.png": "non_cracked_256_grid.png",
        "exp5_flexural_256_grid.png": "flexural_256_grid.png",
        "exp5_flexure_shear_256_grid.png": "flexure_shear_256_grid.png",
        "exp5_web_shear_256_grid.png": "web_shear_256_grid.png",
        "exp5_torsion_256_grid.png": "torsion_256_grid.png",
        "exp5_corrosion_splitting_256_grid.png": "corrosion_splitting_256_grid.png",
        "exp5_shrinkage_map_256_grid.png": "shrinkage_map_256_grid.png",
    }
    return "assets/images/" + aliases.get(name, name)

def pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
        return "\n\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    except Exception as exc:
        return f"Report text extraction was unavailable: {exc}"

common = {
1: dict(title="Unconditional DDPM", kicker="EXPERIMENT 01 · BASELINE", status="COMPLETE", status_class="complete", subtitle="Learning the visual grammar of concrete without a crack condition.", image="assets/images/exp1_seed42.png", report="assets/reports/experiment_1_full_report.md", report_kind="md", facts=[("Resolution","64 × 64 RGB"),("Dataset","SDNET2018"),("Train images","50,482"),("Final loss","0.015996")], aim="Establish an unconditional baseline that can synthesize concrete-surface patches from random noise.", stack="PyTorch · DDPM · compact U-Net · AdamW · CUDA · RTX 4050", procedure="Index SDNET2018, preserve surface-category metadata, preprocess 64×64 RGB patches, train for 50 epochs with a 1,000-step diffusion schedule, sample at multiple seeds, and compare generated images with real validation images.", observations="The model learned concrete-like colour, aggregate texture and surface roughness. Crack visibility was inconsistent because the network was never told to produce a crack.", results="FID 5.604571 · KID 3.419779 ± 1.028021 · paired LPIPS 0.548679. These are general distribution diagnostics, not crack-correctness metrics.", assumptions="The SDNET2018 validation split is representative of the concrete-surface distribution; patch-level appearance is a useful baseline for later crack conditioning; labels can remain metadata because the model is intentionally unconditional.", errors="No controllable crack output; thin crack detail is limited by 64×64 resolution; the manual crack audit remains incomplete; FID/KID/LPIPS cannot establish engineering validity.", conclusion="The baseline learned surface appearance but failed the control requirement. A condition is necessary for crack-specific generation."),
2: dict(title="Crack-Status Conditional DDPM", kicker="EXPERIMENT 02 · BINARY CONDITIONING", status="COMPLETE", status_class="complete", subtitle="Turning cracked versus non-cracked into an explicit generation control.", image="assets/images/exp2_cracked.png", report="assets/reports/experiment_2_formula_report.pdf", report_kind="pdf", facts=[("Condition","0 non-cracked · 1 cracked"),("Resolution","64 × 64 RGB"),("Eval sets","1,000 per class"),("Best FID","0.701002")], aim="Train a DDPM that receives a binary crack-status label and generates separate cracked and non-cracked image sets.", stack="PyTorch · conditional U-Net · class embedding · balanced sampler · AdamW · CUDA", procedure="Reuse the processed SDNET2018 split, add a learned two-class embedding to the time embedding, use a balanced sampler, train for 50 epochs, generate 1,000 images for each condition, and evaluate each condition against the real validation protocol.", observations="Conditioning improved distribution similarity substantially under the chosen protocol. Visual inspection found crack-like patterns, but the crack-presence rate was not established with a validated evaluator.", results="Cracked: FID 0.701002 · KID 0.556873 ± 0.202802 · LPIPS 0.425783. Non-cracked: FID 0.724640 · KID 0.756606 ± 0.389974 · LPIPS 0.416435.", assumptions="The binary label is reliable enough to teach a first control signal; a balanced sampler prevents non-cracked samples from dominating; equal-size generated sets make the class comparison fair.", errors="The auxiliary crack classifier reached only 65.79% validation accuracy; metrics describe general similarity rather than crack correctness; a preliminary 5/16 visual count is not a final prevalence estimate.", conclusion="Binary conditioning is a meaningful improvement over the baseline, but it does not fully solve crack correctness or morphology control."),
3: dict(title="128×128 Conditional DDPM with Cracked-Image Augmentation", kicker="EXPERIMENT 03 · RESOLUTION + NEW DATA", status="CLOSED · EVALUATION NOT PERFORMED", status_class="closed", subtitle="Testing whether higher resolution and additional cracked data preserve finer fracture detail.", image="assets/images/exp3_cracked.png", report="assets/reports/experiment_3_formula_report.pdf", report_kind="pdf", facts=[("Resolution","128 × 128 RGB"),("Training set","61,001 images"),("External source","RC_10519 cracked subset"),("Final loss","0.0140576")], aim="Train a higher-resolution conditional DDPM using SDNET2018 plus an external cracked subset, while keeping class control between cracked and non-cracked outputs.", stack="PyTorch · conditional SmallUNet · 1,000 diffusion steps · AdamW · FP32/CUDA · RTX 4050", procedure="Prepare SDNET2018 and RC_10519, exclude the SDNET-derived Kaggle set to reduce leakage risk, resize to 128×128, train the conditional model for 50 epochs, generate reference grids, and reserve a real-only validation protocol for formal evaluation.", observations="Training completed and reference grids were generated. The large evaluation run was stopped after 32 cracked samples, so no final quality metric exists for this experiment.", results="Final training loss: 0.014057591692068844. FID, KID, LPIPS and crack-presence rate: not measured. This is an explicit closed-experiment state, not a zero result.", assumptions="RC_10519 provides genuinely additional cracked examples; cracked-only external augmentation improves crack detail without invalidating the real-only validation set; 128×128 is a useful compromise for the available 6 GB GPU.", errors="Evaluation was not completed; 32 partial images are insufficient for a final metric; external-domain shift remains possible; the model may still produce fragmented cracks despite higher resolution.", conclusion="The training milestone was achieved, but the quality claim remains open. This experiment must not be described as proving improved crack generation."),
4: dict(title="Morphological Crack-Type Conditioning", kicker="EXPERIMENT 04 · NATIVE 256×256", status="COMPLETE", status_class="complete", subtitle="Moving from crack presence to crack shape with attention and morphology labels.", image="assets/images/exp4_longitudinal_256_grid.png", report="assets/reports/experiment_4_full_report.md", report_kind="md", facts=[("Resolution","256 × 256 native"),("Classes","5 morphology states"),("Parameters","13.82M"),("Fidelity","FID 167.6 · KID 0.04526")], aim="Generate non-cracked, longitudinal, transverse, diagonal and alligator-map patterns at native 256×256 resolution while improving continuity over the upscaled 128×128 baseline.", stack="AttentionUNet256 · FlashAttention · DDIM 50-step sampling · CFG w=2.5 · FP16 AMP · AdamW", procedure="Curate 13,139 native patches from DeepCrack, CrackForest and SDNET2018; train a 5-class conditional AttentionUNet for 50 epochs; use self-attention at 32×32 and 16×16 stages; sample with deterministic DDIM and classifier-free guidance; evaluate fidelity and continuity.", observations="The native-resolution model produced visually continuous cracks across the morphology grids and resolved the fragmented-speck failure mode seen in the earlier upscaled baseline. Automatic orientation alignment was not successful in the recorded evaluation.", results="FID 167.6 · KID 0.04526 ± 0.01413. Crack presence was 1.0 for the four cracked classes, but orientation alignment accuracy was 0.0. The non-cracked set had a 1.0 false-positive crack rate in the recorded 16-sample diagnostic.", assumptions="Native 256×256 preserves hairline topology; attention provides global coordinate awareness; morphology labels can be inferred from image/mask structure; CFG improves class adherence without overwhelming texture diversity.", errors="High FID indicates a distribution mismatch under the selected protocol; orientation alignment failed; non-cracked false positives were severe; continuity metrics depend on thresholding and connected-component choices; the report’s physical interpretation should not be treated as a structural safety claim.", conclusion="Native resolution and morphology conditioning improved visible continuity, but controllability remains uneven: continuity improved while orientation and non-cracked rejection still require work."),
5: dict(title="Structural Failure Mechanism Conditioned DDPM", kicker="EXPERIMENT 05 · LEVEL B EXECUTED", status="COMPLETE", status_class="complete", subtitle="Seven physics-grounded failure mechanisms synthesized at native 256×256 resolution.", image="assets/images/flexure_shear_256_grid.png", report="assets/reports/experiment_5_full_report.md", report_kind="md", facts=[("Canvas","256 × 256 native"),("States","7 mechanism classes"),("Dataset","15,440 patches"),("Fidelity","KID 0.03931 ± 0.01168")], aim="Advance from visual crack morphology to controllable synthesis of seven structural failure mechanisms using physics-guided labels and native-resolution diffusion.", stack="AttentionUNet256 · 13.82M parameters · FlashAttention · 50-step DDIM · CFG w=2.5 · FP16 AMP · AdamW", procedure="Curate native patches from DeepCrack, CrackForest, RC_10519 and SDNET2018; classify them with orientation, curvature, aperture-taper and echelon descriptors; canonically align tension-face geometry; transfer the Experiment 4 backbone with semantic class-embedding projection; fine-tune for 25 epochs; sample 112 images and evaluate mechanism adherence and fidelity.", observations="The model generated mechanism-specific signatures: flexure-shear showed the highest mean curvature (21.4°), torsion achieved 93.8% parallel-echelon adherence, shrinkage-map achieved 100% multi-branch adherence, and non-cracked samples achieved a 93.8% clean-surface rate.", results="Final mean MSE 0.012985 (best 0.012607 at epoch 19) · FID 170.582 · KID 0.03931 ± 0.01168. The KID is the best recorded across the five experiments, while FID remains comparable to Experiment 4.", assumptions="Public datasets lack load-cell and strain-gauge metadata, so A1–A7 map observable morphology to structural mechanisms; the bottom edge represents the tension face; no vertical flip preserves gravity orientation; geometric proxies approximate, but do not prove, the underlying stress state.", errors="Strict evaluation exposed weaker corrosion-splitting presence/adherence (62.5%) and flexure-shear adherence (50.0%); high FID indicates residual distribution mismatch; patch-level images cannot represent full-member behavior; synthetic morphology must not be treated as structural safety evidence.", conclusion="Experiment 5 successfully extends the project to seven physics-grounded mechanisms. Transfer learning, canonical orientation and multi-source curation improved convergence and KID, but direct mechanics labels and full-member validation remain important next steps."),
6: dict(title="Physics-Guided Spatial Control DDPM", kicker="EXPERIMENT 06 · LEVEL C EXECUTED", status="COMPLETE", status_class="complete", subtitle="Adding explicit spatial trajectory and aperture-field conditioning to mechanism-aware diffusion.", image="assets/images/exp6_flexure_shear_256_grid.png", report="assets/reports/experiment_6_full_report.md", report_kind="md", facts=[("Canvas","256 × 256 native"),("Control field","2 spatial physics channels"),("Training","15 transfer epochs · 90 cumulative"),("Adherence","100% across 6 cracked classes")], aim="Test whether explicit spatial physics fields can improve trajectory compliance, crack-width taper and mechanism adherence beyond scalar mechanism conditioning.", stack="PhysicsControlUNet256 · 5-channel input · zero-initialized channel surgery · analytical trajectory solver · Dual-CFG · 50-step DDIM · FP16 AMP", procedure="Solve analytical stress trajectories and aperture fields for seven mechanism classes; expand the Experiment 5 input from three to five channels with zero-initialized physics channels; fine-tune for 15 epochs; sample 16 images per class; evaluate spatial trajectory recall, mechanism adherence, aperture taper and chromatic deviation.", observations="The spatially conditioned model achieved perfect measured mechanism adherence across all six cracked classes and a 100% clean-surface rate for non-cracked samples. It also reduced the camera colour-cast issue while preserving native-resolution outputs.", results="Best MSE 0.011051 at epoch 6; final MSE 0.011304 at epoch 15; FID 332.847; KID 0.17539 ± 0.02070. Structural adherence was 100% across cracked classes, with spatial trajectory recall from 93.49% to 100% and flexural taper ratio 1.50.", assumptions="A two-channel field can represent both crack centerline and local aperture; analytical plane-stress and strain-compatible proxies are useful conditioning signals; zero-initialized surgery preserves the Experiment 5 representation at transfer start; chromatic normalization can reduce camera cast without removing structural texture.", errors="FID and KID worsened relative to Experiment 5 despite stronger physical adherence; torsion continuity remained lower at 0.421; synthetic patch-level adherence is not proof of full-member structural validity; the analytical solver depends on simplified geometry, material assumptions and proxy stress states.", conclusion="Experiment 6 demonstrates that explicit spatial physics control can make the requested mechanism and trajectory measurable and highly adherent. The trade-off is distribution fidelity: stronger control still needs broader validation and more realistic structural data."),
}

result_images = {
    1: [("Generated concrete surface", "assets/images/exp1_seed42.png")],
    2: [("Cracked condition", "assets/images/exp2_cracked.png"), ("Non-cracked condition", "assets/images/exp2_noncracked.png")],
    3: [("Cracked samples", "assets/images/exp3_cracked.png"), ("Non-cracked samples", "assets/images/exp3_noncracked.png")],
    4: [("Non-cracked", "assets/images/exp4_non_cracked_256_grid.png"), ("Longitudinal", "assets/images/exp4_longitudinal_256_grid.png"), ("Transverse", "assets/images/exp4_transverse_256_grid.png"), ("Diagonal", "assets/images/exp4_diagonal_256_grid.png"), ("Alligator map", "assets/images/exp4_alligator_map_256_grid.png")],
    5: [("Non-cracked", "assets/images/non_cracked_256_grid.png"), ("Flexural", "assets/images/flexural_256_grid.png"), ("Flexure-shear", "assets/images/flexure_shear_256_grid.png"), ("Web shear", "assets/images/web_shear_256_grid.png"), ("Torsion", "assets/images/torsion_256_grid.png"), ("Corrosion splitting", "assets/images/corrosion_splitting_256_grid.png"), ("Shrinkage / map", "assets/images/shrinkage_map_256_grid.png")],
    6: [("Non-cracked", "assets/images/exp6_non_cracked_256_grid.png"), ("Flexural", "assets/images/exp6_flexural_256_grid.png"), ("Flexure-shear", "assets/images/exp6_flexure_shear_256_grid.png"), ("Web shear", "assets/images/exp6_web_shear_256_grid.png"), ("Torsion", "assets/images/exp6_torsion_256_grid.png"), ("Corrosion splitting", "assets/images/exp6_corrosion_splitting_256_grid.png"), ("Shrinkage / map", "assets/images/exp6_shrinkage_map_256_grid.png")],
}

def page(n, d):
    report_path = PROJECT / ("experiments/unconditional_ddpm/EXPERIMENT_1_FULL_REPORT.md" if n == 1 else d["report"] .replace("assets/reports/", "experiments/"))
    if d['report_kind'] == 'archive':
        body = d['archive_body']
    elif d["report_kind"] == "md":
        if n == 2: report_path = PROJECT / 'experiments/conditional_ddpm/EXPERIMENT_2_FULL_REPORT.md'
        if n == 3: report_path = PROJECT / 'experiments/conditional_ddpm_128_augmented/EXPERIMENT_3_FULL_REPORT.md'
        if n == 4: report_path = PROJECT / "experiments/crack_type_conditional/EXPERIMENT_4_FULL_REPORT.md"
        if n == 5: report_path = PROJECT / "experiments/structural_mechanism_ddpm/EXPERIMENT_5_FULL_REPORT.md"
        if n == 6: report_path = PROJECT / "experiments/exp6_level_c_spatial_physics/EXPERIMENT_6_FULL_REPORT.md"
        body = markdown_to_html(report_path.read_text(encoding="utf-8"))
    else:
        report_path = PROJECT / ("experiments/conditional_ddpm/reports/experiment_2_conditional_ddpm_formula_report.pdf" if n == 2 else "experiments/conditional_ddpm_128_augmented/reports/experiment_3_conditional_128_augmented_report.pdf")
        body = "<p>" + "</p><p>".join(html.escape(pdf_text(report_path)).split("\n\n")) + "</p>"
    facts = "".join(f'<div><span>{html.escape(k)}</span><b>{html.escape(v)}</b></div>' for k,v in d["facts"])
    gallery = "".join(f'<figure><img src="{src}" alt="{html.escape(label)} result image"><figcaption>{html.escape(label)}</figcaption></figure>' for label, src in result_images[n])
    previous = f'<a href="experiment-{n-1}.html">← Experiment {n-1:02d}</a>' if n > 1 else '<a href="index.html">← Archive home</a>'
    nxt = f'<a href="experiment-{n+1}.html">Experiment {n+1:02d} →</a>' if n < 6 else '<a href="index.html#archive">Back to archive →</a>'
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Experiment {n:02d} · {html.escape(d['title'])}</title><link rel="stylesheet" href="experiment.css"><script>window.MathJax={{tex:{{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']]}},options:{{skipHtmlTags:['script','noscript','style','textarea','pre','code']}}}};</script><script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script></head><body><header class="detail-header"><a href="index.html" class="brand">▧ CRACK / SYNTHESIS</a><a href="index.html#experiments" class="back">All experiments ↗</a></header><main><section class="detail-hero"><div><p class="eyebrow">{d['kicker']}</p><span class="detail-status {d['status_class']}">{d['status']}</span><h1>{html.escape(d['title'])}</h1><p class="lede">{html.escape(d['subtitle'])}</p></div></section><section class="facts">{facts}</section><section class="report-grid"><article class="report-main"><section><p class="eyebrow">AIM</p><h2>What this experiment was trying to learn</h2><p>{html.escape(d['aim'])}</p></section><section><p class="eyebrow">TECH STACK</p><h2>Implementation choices</h2><p>{html.escape(d['stack'])}</p></section><section><p class="eyebrow">PROCEDURE</p><h2>How it was run</h2><p>{html.escape(d['procedure'])}</p></section><section class="result-box"><p class="eyebrow">RESULT</p><h2>What the evidence says</h2><p>{html.escape(d['results'])}</p></section><section><p class="eyebrow">OBSERVATIONS</p><h2>What appeared in the outputs</h2><p>{html.escape(d['observations'])}</p></section><section><p class="eyebrow">ASSUMPTIONS</p><h2>Decisions behind the result</h2><p>{html.escape(d['assumptions'])}</p></section><section><p class="eyebrow">ERRORS & SOURCES OF ERROR</p><h2>What could mislead us</h2><p>{html.escape(d['errors'])}</p></section><section><p class="eyebrow">CONCLUSION</p><h2>Research takeaway</h2><p>{html.escape(d['conclusion'])}</p></section><section class="result-gallery"><p class="eyebrow">RESULT IMAGES</p><h2>Generated outputs</h2><div class="gallery-grid">{gallery}</div></section><section class="full-report"><p class="eyebrow">FULL REPORT CONTENT</p><h2>Technical record</h2><details open><summary>Expand / collapse the original report</summary><div class="report-content">{body}</div></details></section></article><aside class="detail-aside"><div class="aside-card"><p class="eyebrow">EVIDENCE STATUS</p><strong>{d['status']}</strong><p>This page distinguishes measured outputs from interpretation and planned work.</p></div><div class="aside-card"><p class="eyebrow">NAVIGATE</p><div class="pager">{previous}{nxt}</div></div></aside></section></main><footer class="detail-footer"><span>CRACK / SYNTHESIS · Experiment {n:02d}</span><a href="index.html">Return to research archive</a></footer></body></html>'''

if __name__ == '__main__':
    for n, data in common.items():
        (ROOT / f"experiment-{n}.html").write_text(page(n, data), encoding="utf-8")
    print("Generated experiment pages")
