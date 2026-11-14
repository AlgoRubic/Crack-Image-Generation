from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(r"D:\Downloads\concrete_ddpm")
EXP = ROOT / "experiments" / "conditional_ddpm"
OUT = EXP / "reports"
OUT.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUT / "experiment_2_conditional_ddpm_formula_report.pdf"
EQUATION_DIR = OUT / "equations"
EQUATION_DIR.mkdir(parents=True, exist_ok=True)
EQUATION_COUNTER = 0


def styles():
    base = getSampleStyleSheet()
    base["Normal"].fontName = "Helvetica"
    base["Normal"].fontSize = 9.4
    base["Normal"].leading = 13
    base["Normal"].alignment = TA_JUSTIFY
    base["Title"].fontName = "Helvetica-Bold"
    base["Title"].fontSize = 20
    base["Title"].leading = 24
    base["Title"].alignment = TA_CENTER
    base["Heading1"].fontName = "Helvetica-Bold"
    base["Heading1"].fontSize = 14
    base["Heading1"].leading = 18
    base["Heading1"].spaceBefore = 12
    base["Heading1"].spaceAfter = 6
    base["Heading2"].fontName = "Helvetica-Bold"
    base["Heading2"].fontSize = 11.5
    base["Heading2"].leading = 15
    base["Heading2"].spaceBefore = 8
    base["Heading2"].spaceAfter = 4
    base.add(
        ParagraphStyle(
            name="Formula",
            parent=base["Normal"],
            fontName="Courier",
            fontSize=8.2,
            leading=11,
            leftIndent=18,
            rightIndent=12,
            backColor=colors.HexColor("#F4F6F8"),
            borderColor=colors.HexColor("#D9DEE5"),
            borderWidth=0.35,
            borderPadding=5,
            spaceBefore=4,
            spaceAfter=6,
        )
    )
    base.add(
        ParagraphStyle(
            name="Caption",
            parent=base["Normal"],
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#505860"),
        )
    )
    return base


S = styles()


def p(text: str, style: str = "Normal"):
    return Paragraph(text, S[style])


def formula(text: str):
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(escaped.replace("\n", "<br/>"), S["Formula"])


def math_block(lines: list[str], width: float | None = None):
    global EQUATION_COUNTER
    EQUATION_COUNTER += 1
    path = EQUATION_DIR / f"equation_{EQUATION_COUNTER:03d}.png"
    fig_height = max(0.55, 0.34 * len(lines) + 0.16)
    fig = plt.figure(figsize=(7.6, fig_height), dpi=220)
    fig.patch.set_facecolor("#F4F6F8")
    y = 0.88
    step = 0.78 / max(1, len(lines))
    for line in lines:
        fig.text(0.03, y, f"${line}$", fontsize=11, color="#111827", va="center")
        y -= step
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08, facecolor=fig.get_facecolor())
    plt.close(fig)
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    if width is None:
        max_len = max(len(line) for line in lines)
        width = min(5.75 * inch, max(2.4 * inch, max_len * 0.052 * inch))
    img.drawWidth = width
    img.drawHeight = width * ratio
    img.hAlign = "CENTER"
    return img


def bullets(items: list[str]):
    return ListFlowable(
        [ListItem(p(item), bulletText="-", leftIndent=12) for item in items],
        bulletType="bullet",
        leftIndent=16,
    )


def table(rows, widths=None):
    wrapped = []
    for row_index, row in enumerate(rows):
        wrapped_row = []
        for cell in row:
            if isinstance(cell, str):
                wrapped_row.append(Paragraph(cell, S["Heading2" if row_index == 0 else "Normal"]))
            else:
                wrapped_row.append(cell)
        wrapped.append(wrapped_row)
    t = Table(wrapped, colWidths=widths, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1F2933")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.4),
                ("LEADING", (0, 0), (-1, -1), 10.5),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D0DA")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFBFC")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(0.65 * inch, 0.42 * inch, "Experiment 2 - Conditional DDPM Formula Report")
    canvas.drawRightString(A4[0] - 0.65 * inch, 0.42 * inch, f"Page {doc.page}")
    canvas.restoreState()


def add_reference_grid(story, path: Path, width=3.05 * inch):
    if path.exists():
        img = Image(str(path), width=width, height=width)
        img.hAlign = "CENTER"
        story.append(img)
        story.append(p(path.name, "Caption"))
        story.append(Spacer(1, 7))


def build():
    story = []
    story.append(p("Experiment 2 Formula Report", "Title"))
    story.append(p("Crack-Status Conditional DDPM for Concrete Images", "Title"))
    story.append(Spacer(1, 10))
    story.append(
        p(
            "This report documents the second concrete crack image generation experiment. Experiment 2 extends the "
            "unconditional DDPM baseline by adding a binary crack-status condition so the same model can be sampled as "
            "either cracked or non-cracked."
        )
    )
    story.append(Spacer(1, 10))
    story.append(
        table(
            [
                ["Item", "Value"],
                ["Dataset", "SDNET2018 processed dataset from Experiment 1"],
                ["Task", "Class-conditional concrete image generation"],
                ["Conditions", "class 0 = non_cracked, class 1 = cracked"],
                ["Resolution", "64 x 64 RGB"],
                ["Model", "SmallUNet DDPM with learned two-class embedding"],
                ["Epochs", "50"],
                ["Diffusion steps", "1000"],
                ["Final checkpoint", r"experiments\conditional_ddpm\checkpoints\conditional_epoch_050.pt"],
                ["Final loss", "approximately 0.015247"],
            ],
            [1.55 * inch, 4.85 * inch],
        )
    )
    story.append(PageBreak())

    story.append(p("1. Aim and Motivation", "Heading1"))
    story.append(
        p(
            "Experiment 1 could generate concrete-like images, but it could not be instructed to generate cracked or "
            "non-cracked samples. Experiment 2 addresses that limitation by adding a crack-status condition to the DDPM."
        )
    )
    story.append(
        bullets(
            [
                "Train a DDPM that receives an explicit binary crack-status condition.",
                "Generate separate cracked-condition and non-cracked-condition image sets.",
                "Compare conditional generation against the unconditional baseline.",
                "Measure general distribution similarity while clearly noting that crack correctness still needs visual or classifier-based evaluation.",
            ]
        )
    )

    story.append(p("2. Dataset Structure and Preprocessing", "Heading1"))
    story.append(p("2.1 Dataset Used", "Heading2"))
    story.append(
        p(
            "Experiment 2 reused the same processed SDNET2018 dataset as Experiment 1. Images were already indexed, "
            "split, converted to RGB, resized to 64 x 64, saved as PNG files, and normalized to [-1,1] during loading."
        )
    )
    story.append(
        table(
            [
                ["Quantity", "Value"],
                ["Total indexed images", "56,092"],
                ["Training images", "50,482"],
                ["Validation images", "5,610"],
                ["Training cracked", "7,636"],
                ["Training non-cracked", "42,846"],
                ["Validation cracked", "848"],
                ["Validation non-cracked", "4,762"],
            ],
            [2.05 * inch, 4.35 * inch],
        )
    )
    story.append(p("2.2 Raw Dataset Meaning", "Heading2"))
    story.append(
        table(
            [
                ["Raw folder", "Surface", "Class folder", "Condition label", "Class id"],
                ["D/CD", "bridge_deck", "CD", "cracked", "1"],
                ["D/UD", "bridge_deck", "UD", "non_cracked", "0"],
                ["P/CP", "pavement", "CP", "cracked", "1"],
                ["P/UP", "pavement", "UP", "non_cracked", "0"],
                ["W/CW", "wall", "CW", "cracked", "1"],
                ["W/UW", "wall", "UW", "non_cracked", "0"],
            ],
            [0.9 * inch, 1.2 * inch, 1.0 * inch, 1.35 * inch, 0.8 * inch],
        )
    )
    story.append(
        p(
            "Cracked images contain visible crack lines, fissures, or crack-like separations. Non-cracked images contain "
            "concrete texture without a visible crack, though they can still include roughness, aggregate, stains, pores, "
            "and illumination variation. The semantic difference between these classes is the reason conditional generation "
            "is needed."
        )
    )
    story.append(p("2.3 Preprocessing Pipeline", "Heading2"))
    story.append(
        formula(
            "raw image -> RGB conversion -> 64 x 64 LANCZOS resize -> PNG\n"
            "uint8 pixel p in [0,255]\n"
            "ToTensor: p -> p / 255 in [0,1]\n"
            "Normalize(mean=0.5, std=0.5): x_0 = 2 * (p / 255) - 1 in [-1,1]"
        )
    )

    story.append(PageBreak())
    story.append(p("3. Conditional DDPM Theory", "Heading1"))
    story.append(p("3.1 Unconditional vs Conditional Modeling", "Heading2"))
    story.append(
        p(
            "The unconditional model estimates the image distribution without a control label. The conditional model "
            "estimates image generation under a class condition y."
        )
    )
    story.append(math_block([r"\mathrm{Unconditional:}\quad p_\theta(x_0)", r"\mathrm{Conditional:}\quad p_\theta(x_0\mid y)"]))
    story.append(
        p(
            "For this experiment, y is binary: y=0 means non-cracked and y=1 means cracked. The model learns a class "
            "embedding and adds it to the timestep embedding before the U-Net residual blocks."
        )
    )
    story.append(math_block([r"y\in\{0,1\}", r"e_t=\mathrm{TimeEmbed}(t)", r"e_y=\mathrm{ClassEmbed}(y)", r"e=e_t+e_y"]))

    story.append(p("3.2 Forward Diffusion", "Heading2"))
    story.append(
        p(
            "The forward noising process is the same as Experiment 1. The condition is not used to add noise; it is used "
            "by the denoising model when predicting the noise."
        )
    )
    story.append(math_block([r"q(x_t\mid x_0)=\mathcal{N}\!\left(x_t;\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I\right)"]))
    story.append(math_block([r"x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I)"]))

    story.append(p("3.3 Conditional Noise Prediction Objective", "Heading2"))
    story.append(
        p(
            "The training target remains Gaussian noise prediction. The difference is that the U-Net now receives both "
            "the timestep and the class condition."
        )
    )
    story.append(math_block([r"\hat{\epsilon}=\epsilon_\theta(x_t,t,y)"]))
    story.append(math_block([r"L_{\mathrm{cond}}(\theta)=\mathbb{E}_{x_0,\epsilon,t,y}\left[\left\|\epsilon-\epsilon_\theta(x_t,t,y)\right\|_2^2\right]"]))

    story.append(p("3.4 Conditional Sampling", "Heading2"))
    story.append(
        p(
            "During sampling, the desired class label is fixed for the whole reverse process. Setting y=1 asks the model "
            "for cracked-condition samples; setting y=0 asks for non-cracked-condition samples."
        )
    )
    story.append(
        math_block(
            [
                r"x_T\sim\mathcal{N}(0,I)",
                r"x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t,y)\right)+\sigma_t z",
            ]
        )
    )

    story.append(PageBreak())
    story.append(p("4. Experimental Procedure", "Heading1"))
    story.append(
        table(
            [
                ["Order", "Step", "Output"],
                ["1", "Reuse Experiment 1 processed SDNET2018 data", r"data\processed\train.csv and validation.csv"],
                ["2", "Load labels and assign class ids", "non_cracked=0, cracked=1"],
                ["3", "Create weighted random sampler", "balanced cracked/non-cracked training batches"],
                ["4", "Train SmallUNet with num_classes=2 for 50 epochs", r"checkpoints\conditional_epoch_050.pt"],
                ["5", "Generate cracked sample grids", r"outputs\samples\cracked_epoch_050_seed_*.png"],
                ["6", "Generate non-cracked sample grids", r"outputs\samples\non_cracked_epoch_050_seed_*.png"],
                ["7", "Generate 1,000 cracked-condition eval images", r"outputs\evaluation\conditional_cracked"],
                ["8", "Generate 1,000 non-cracked-condition eval images", r"outputs\evaluation\conditional_non_cracked"],
                ["9", "Evaluate both generated sets", "FID, KID, paired LPIPS"],
            ],
            [0.5 * inch, 2.85 * inch, 3.05 * inch],
        )
    )
    story.append(p("4.1 Balanced Sampler", "Heading2"))
    story.append(
        p(
            "The raw SDNET2018 training split is heavily imbalanced: non-cracked images are much more common than cracked "
            "images. The conditional training script uses a WeightedRandomSampler so cracked images are not overwhelmed."
        )
    )
    story.append(math_block([r"w_i=\frac{1}{N_{y_i}}", r"P(i)\propto w_i"]))
    story.append(
        p(
            "This does not change the stored dataset; it changes the probability that each training example is sampled "
            "during an epoch. The goal is to give the model enough cracked examples while still learning non-cracked texture."
        )
    )
    story.append(p("4.2 Training Configuration", "Heading2"))
    story.append(
        table(
            [
                ["Item", "Value"],
                ["Training script", r"src\train_conditional.py"],
                ["Batch size", "32"],
                ["Learning rate", "2e-4"],
                ["Epochs", "50"],
                ["Optimizer", "AdamW"],
                ["Mixed precision", "CUDA AMP enabled when CUDA is available"],
                ["Gradient clipping", "clip_grad_norm = 1.0"],
                ["Model parameters", "3,258,627"],
                ["Additional parameters vs Experiment 1", "512"],
            ],
            [2.2 * inch, 4.2 * inch],
        )
    )

    story.append(PageBreak())
    story.append(p("5. Evaluation Metrics", "Heading1"))
    story.append(
        p(
            "Experiment 2 uses the same general metrics as Experiment 1. These metrics evaluate distributional or "
            "perceptual similarity, not guaranteed crack correctness."
        )
    )
    story.append(p("5.1 FID", "Heading2"))
    story.append(math_block([r"\mathrm{FID}=\left\|\mu_r-\mu_g\right\|_2^2+\mathrm{Tr}\!\left(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\right)"]))
    story.append(p("5.2 KID", "Heading2"))
    story.append(
        math_block(
            [
                r"\mathrm{MMD}^2(X,Y)=\frac{1}{m(m-1)}\sum_{i\ne j}k(x_i,x_j)",
                r"+\frac{1}{n(n-1)}\sum_{i\ne j}k(y_i,y_j)-\frac{2}{mn}\sum_i\sum_j k(x_i,y_j)",
            ]
        )
    )
    story.append(math_block([r"\mathrm{KID}=\mathrm{MMD}^2(F_{\mathrm{real}},F_{\mathrm{generated}})"]))
    story.append(p("5.3 LPIPS", "Heading2"))
    story.append(math_block([r"\mathrm{LPIPS}(x,y)=\sum_l\frac{1}{H_lW_l}\sum_{h,w}\left\|w_l\odot(\phi_l(x)_{h,w}-\phi_l(y)_{h,w})\right\|_2^2"]))

    story.append(p("6. Results", "Heading1"))
    story.append(
        table(
            [
                ["Condition", "FID", "KID mean", "KID std", "Paired LPIPS"],
                ["cracked", "0.701002", "0.556873", "0.202802", "0.425783"],
                ["non_cracked", "0.724640", "0.756606", "0.389974", "0.416435"],
            ],
            [1.35 * inch, 1.0 * inch, 1.1 * inch, 1.0 * inch, 1.3 * inch],
        )
    )
    story.append(
        p(
            "Both conditional sets scored much lower FID and KID than the unconditional baseline under the same evaluation "
            "procedure. This indicates stronger general distribution matching. It does not by itself prove reliable crack "
            "generation, because crack presence still requires visual annotation or a validated crack classifier."
        )
    )
    story.append(
        table(
            [
                ["Experiment", "FID", "KID mean", "Paired LPIPS"],
                ["Experiment 1 unconditional", "5.604571", "3.419779", "0.548679"],
                ["Experiment 2 cracked condition", "0.701002", "0.556873", "0.425783"],
                ["Experiment 2 non-cracked condition", "0.724640", "0.756606", "0.416435"],
            ],
            [2.35 * inch, 1.0 * inch, 1.2 * inch, 1.25 * inch],
        )
    )

    story.append(PageBreak())
    story.append(p("7. Reference Generated Images", "Heading1"))
    story.append(p("7.1 Cracked Condition Reference Grids", "Heading2"))
    add_reference_grid(story, EXP / "outputs" / "samples" / "cracked_epoch_050_seed_42.png")
    add_reference_grid(story, EXP / "outputs" / "samples" / "cracked_epoch_050_seed_123.png")
    story.append(p("7.2 Non-Cracked Condition Reference Grids", "Heading2"))
    add_reference_grid(story, EXP / "outputs" / "samples" / "non_cracked_epoch_050_seed_42.png")
    add_reference_grid(story, EXP / "outputs" / "samples" / "non_cracked_epoch_050_seed_123.png")

    story.append(PageBreak())
    story.append(p("8. Observations and Limitations", "Heading1"))
    story.append(
        bullets(
            [
                "The conditional model achieved substantially better general metrics than the unconditional baseline.",
                "The learned two-class embedding allows separate cracked and non-cracked sampling commands.",
                "A preliminary visual check found about 5 visibly cracked samples out of 16 in one cracked-condition grid, but this is not a final crack-presence metric.",
                "The first auxiliary crack classifier reached only 65.79 percent validation accuracy, so automatic crack-presence evaluation is not yet reliable.",
                "FID, KID, and LPIPS must not be interpreted as proof that the cracked condition reliably generates real cracks.",
            ]
        )
    )
    story.append(p("9. Conclusion", "Heading1"))
    story.append(
        p(
            "Experiment 2 successfully extends the DDPM baseline from unconditional concrete texture generation to "
            "binary crack-status conditional generation. It preserves the same 64 x 64 SDNET2018 setup but adds a learned "
            "class embedding, balanced class sampling, and separate cracked/non-cracked generation paths. The formal metrics "
            "improve strongly relative to Experiment 1, but a reliable crack-presence result remains pending."
        )
    )
    story.append(p("10. Recommended Next Steps", "Heading1"))
    story.append(
        bullets(
            [
                "Create a manual visual audit CSV for representative cracked-condition and non-cracked-condition samples.",
                "Report visible crack rate, texture quality, and artifact rate separately for each condition.",
                "Train or validate a stronger crack classifier before using automatic crack-presence rates.",
                "Compare Experiment 2 against the 128 x 128 augmented Experiment 3 only after Experiment 3 evaluation is complete.",
            ]
        )
    )

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=0.62 * inch,
        leftMargin=0.62 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.65 * inch,
        title="Experiment 2 Conditional DDPM Formula Report",
        author="Codex",
    )
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(PDF_PATH)


if __name__ == "__main__":
    build()
