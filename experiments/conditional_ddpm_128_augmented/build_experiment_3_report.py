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
from reportlab.platypus import Image, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(r"D:\Downloads\concrete_ddpm")
EXP = ROOT / "experiments" / "conditional_ddpm_128_augmented"
OUT = EXP / "reports"
OUT.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUT / "experiment_3_conditional_128_augmented_report.pdf"
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
    base.add(ParagraphStyle(name="Formula", parent=base["Normal"], fontName="Courier", fontSize=8.2, leading=11, leftIndent=18, rightIndent=12, backColor=colors.HexColor("#F4F6F8"), borderColor=colors.HexColor("#D9DEE5"), borderWidth=0.35, borderPadding=5, spaceBefore=4, spaceAfter=6))
    base.add(ParagraphStyle(name="Caption", parent=base["Normal"], fontSize=8, leading=10, alignment=TA_CENTER, textColor=colors.HexColor("#505860")))
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
    return ListFlowable([ListItem(p(item), bulletText="-", leftIndent=12) for item in items], bulletType="bullet", leftIndent=16)


def table(rows, widths=None):
    wrapped = []
    for row_index, row in enumerate(rows):
        wrapped.append([Paragraph(cell, S["Heading2" if row_index == 0 else "Normal"]) if isinstance(cell, str) else cell for cell in row])
    t = Table(wrapped, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
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
    ]))
    return t


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(0.65 * inch, 0.42 * inch, "Experiment 3 - 128x128 Conditional Augmented DDPM Report")
    canvas.drawRightString(A4[0] - 0.65 * inch, 0.42 * inch, f"Page {doc.page}")
    canvas.restoreState()


def add_image(story, path: Path, caption: str, width=4.25 * inch):
    if path.exists():
        img = Image(str(path), width=width, height=width)
        img.hAlign = "CENTER"
        story.append(img)
        story.append(p(caption, "Caption"))
        story.append(Spacer(1, 8))


def build():
    story = []
    story.append(p("Experiment 3 Report", "Title"))
    story.append(p("128x128 Conditional DDPM with Additional Cracked Data", "Title"))
    story.append(Spacer(1, 10))
    story.append(p("This report documents Experiment 3. It increases image resolution from 64 x 64 to 128 x 128 and augments SDNET2018 with an external cracked reinforced-concrete image dataset. Formal FID/KID/LPIPS evaluation is pending and is intentionally not reported here."))
    story.append(Spacer(1, 10))
    story.append(table([
        ["Item", "Value"],
        ["Dataset", "SDNET2018 + external RC_10519 cracked images"],
        ["Task", "128x128 binary conditional concrete image generation"],
        ["Conditions", "class 0 = non_cracked, class 1 = cracked"],
        ["Resolution", "128 x 128 RGB"],
        ["Model", "Conditional SmallUNet/DDPM"],
        ["Epochs", "50"],
        ["Diffusion steps", "1000"],
        ["Final checkpoint", r"experiments\conditional_ddpm_128_augmented\checkpoints\conditional_128_augmented_epoch_050.pt"],
        ["Final training loss", "0.014057591692068844"],
    ], [1.65 * inch, 4.75 * inch]))
    story.append(PageBreak())

    story.append(p("1. Aim and Motivation", "Heading1"))
    story.append(p("Experiment 2 introduced crack-status conditioning, but it still generated 64 x 64 images from SDNET2018 only. Experiment 3 was designed to preserve more crack detail by moving to 128 x 128 resolution and adding an independent cracked-concrete dataset."))
    story.append(bullets([
        "Increase output resolution from 64 x 64 to 128 x 128.",
        "Keep binary class conditioning for cracked and non-cracked generation.",
        "Add 10,519 external cracked reinforced-concrete images.",
        "Improve visible crack detail compared with lower-resolution experiments.",
        "Avoid using the Kaggle 800-image SDNET-derived crack dataset to reduce leakage risk.",
    ]))

    story.append(p("2. Dataset Structure", "Heading1"))
    story.append(p("2.1 Dataset Sources", "Heading2"))
    story.append(table([
        ["Source", "Images used", "Label policy", "Role"],
        ["SDNET2018", "50,482 training images", "cracked and non_cracked labels from SDNET2018 folder structure", "Base concrete dataset"],
        ["RC_10519", "10,519 images", "all used as cracked", "Additional independent cracked-concrete data"],
    ], [1.25 * inch, 1.45 * inch, 2.0 * inch, 1.7 * inch]))
    story.append(p("2.2 Combined Training Composition", "Heading2"))
    story.append(table([
        ["Component", "Count"],
        ["SDNET2018 non-cracked", "42,846"],
        ["SDNET2018 cracked", "7,636"],
        ["External cracked RC_10519", "10,519"],
        ["Total", "61,001"],
    ], [2.6 * inch, 1.5 * inch]))
    story.append(p("2.3 External Dataset Record", "Heading2"))
    story.append(table([
        ["Item", "Value"],
        ["Downloaded archive", r"D:\Downloads\Reinforced concrete structure crack detection data.zip"],
        ["Preserved raw path", r"data\raw\external_cracks\Reinforced concrete structure crack detection data"],
        ["Image folder", "images, 10,519 JPG files"],
        ["Annotation folder", "label_json, 5,512 LabelMe JSON files"],
        ["Annotation note", "Some images do not have JSON annotations, so annotation_available was preserved as metadata."],
    ], [1.65 * inch, 4.75 * inch]))
    story.append(p("2.4 Dataset Decision", "Heading2"))
    story.append(p("The Kaggle 800-image crack segmentation dataset was not used because its data card states that it was selected from SDNET2018. Adding it to SDNET2018 could introduce duplication or train/test leakage. The external RC_10519 dataset was used instead as the additional cracked source."))

    story.append(PageBreak())
    story.append(p("3. Preprocessing Procedure", "Heading1"))
    story.append(p("The preprocessing script src/prepare_experiment3_128.py created a new 128 x 128 processed training set under data/processed_128. It reused the SDNET2018 training split and appended external cracked images."))
    story.append(table([
        ["Step", "What was done", "Why it was necessary"],
        ["Source integration", "SDNET2018 train.csv records and external JPG files were combined.", "Creates one training CSV for the augmented experiment."],
        ["Metadata preservation", "source, original_path, surface, label, annotation_available, width, and height were written.", "Keeps provenance and annotation availability traceable."],
        ["RGB conversion", "Every image was converted to RGB.", "Ensures consistent 3-channel model input."],
        ["Resize", "Every image was resized to 128 x 128 using LANCZOS.", "Allows 128 x 128 training while preserving more detail than 64 x 64."],
        ["Stable naming", "Processed filenames used SHA1 hashes based on source and original path.", "Avoids collisions between SDNET2018 and external filenames."],
        ["Normalization during loading", "ToTensor maps [0,255] to [0,1], then Normalize(0.5,0.5) maps to [-1,1].", "Matches diffusion training scale."],
    ], [1.25 * inch, 2.45 * inch, 2.7 * inch]))
    story.append(formula("raw image -> RGB -> resize_128x128_LANCZOS -> PNG\nuint8 pixel p in [0,255]\nToTensor: p -> p / 255 in [0,1]\nNormalize(mean=0.5, std=0.5): x_0 = 2 * (p / 255) - 1 in [-1,1]"))

    story.append(p("4. Conditional DDPM Theory", "Heading1"))
    story.append(p("The model keeps the binary conditioning mechanism from Experiment 2. A class embedding is added to the timestep embedding so the denoising network receives both diffusion time and desired crack status."))
    story.append(math_block([r"y\in\{0,1\},\quad 0=\mathrm{non\_cracked},\quad 1=\mathrm{cracked}", r"e=e_t+e_y"]))
    story.append(math_block([r"q(x_t\mid x_0)=\mathcal{N}\!\left(x_t;\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I\right)"]))
    story.append(math_block([r"L_{\mathrm{cond}}(\theta)=\mathbb{E}_{x_0,\epsilon,t,y}\left[\left\|\epsilon-\epsilon_\theta(x_t,t,y)\right\|_2^2\right]"]))
    story.append(math_block([r"x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t,y)\right)+\sigma_t z"]))

    story.append(PageBreak())
    story.append(p("5. Experimental Procedure", "Heading1"))
    story.append(table([
        ["Order", "Step", "Output"],
        ["1", "Prepare 128 x 128 augmented training data", r"data\processed_128\train.csv"],
        ["2", "Preserve SDNET2018 and RC_10519 source metadata", "source and original_path fields"],
        ["3", "Assign labels", "SDNET2018 labels plus RC_10519 as cracked"],
        ["4", "Use weighted random sampler", "balanced cracked/non-cracked training batches"],
        ["5", "Train conditional SmallUNet/DDPM for 50 epochs", r"checkpoints\conditional_128_augmented_epoch_050.pt"],
        ["6", "Generate 16-image reference grids", r"outputs\reference_128"],
        ["7", "Stop incomplete large cracked generation at 32 images", r"outputs\evaluation\conditional_128_augmented_cracked"],
        ["8", "Defer full FID/KID/LPIPS evaluation", "pending complete generated sets"],
    ], [0.55 * inch, 2.8 * inch, 3.05 * inch]))
    story.append(p("5.1 Training Configuration", "Heading2"))
    story.append(table([
        ["Item", "Value"],
        ["Training script", r"src\train_conditional_128_augmented.py"],
        ["Batch size", "16 recorded in STATUS.md"],
        ["Optimizer", "AdamW, learning rate 2e-4"],
        ["Epochs", "50"],
        ["Diffusion steps", "1000"],
        ["Mixed precision", "CUDA AMP enabled when CUDA is available"],
        ["Gradient clipping", "clip_grad_norm = 1.0"],
        ["GPU", "NVIDIA GeForce RTX 4050 Laptop GPU"],
        ["Final training loss", "0.014057591692068844"],
    ], [2.0 * inch, 4.4 * inch]))

    story.append(p("6. Training Result", "Heading1"))
    story.append(p("Training completed successfully for 50 epochs. The final checkpoint exists and should be treated as the completed Experiment 3 model. Formal image-distribution metrics have not yet been computed because the large generation run was intentionally stopped early."))
    story.append(table([
        ["Artifact", "Path / value"],
        ["Final checkpoint", r"checkpoints\conditional_128_augmented\conditional_128_augmented_epoch_050.pt"],
        ["Training log", r"outputs\logs\conditional_128_augmented_training_loss.csv"],
        ["Final loss", "0.014057591692068844"],
        ["Partial generated cracked images", "32 images only, not sufficient for final evaluation"],
    ], [2.2 * inch, 4.2 * inch]))

    story.append(PageBreak())
    story.append(p("7. Reference Generated Images", "Heading1"))
    story.append(p("The following reference grids contain 16 generated images per class from epoch 50. These are visual references only. They are not a substitute for large-scale FID/KID/LPIPS evaluation."))
    add_image(story, EXP / "outputs" / "reference_128" / "cracked_epoch_050.png", "Cracked-condition 128 x 128 reference grid, 16 generated images.")
    add_image(story, EXP / "outputs" / "reference_128" / "non_cracked_epoch_050.png", "Non-cracked-condition 128 x 128 reference grid, 16 generated images.")

    story.append(PageBreak())
    story.append(p("8. Evaluation Status", "Heading1"))
    story.append(p("Formal evaluation metrics are intentionally omitted from this report because the full generated evaluation sets do not exist yet. The large cracked generation run was stopped after 32 images, and those partial images must not be used for final FID/KID/LPIPS reporting."))
    story.append(bullets([
        "No final FID is reported.",
        "No final KID is reported.",
        "No final LPIPS is reported.",
        "No final crack-presence rate is reported.",
        "A full evaluation requires complete generated sets, ideally 1,000 images per class, followed by the same metric procedure used in Experiments 1 and 2.",
    ]))
    story.append(p("9. Observations and Limitations", "Heading1"))
    story.append(bullets([
        "The cracked reference grid shows visible crack-like patterns in several samples.",
        "The non-cracked reference grid mostly shows texture-only concrete surfaces.",
        "The 128 x 128 resolution preserves more visual detail than 64 x 64 outputs.",
        "External cracked-only augmentation may bias the cracked condition and must be evaluated carefully.",
        "External dataset overlap and annotation inconsistency remain risks.",
        "The model should not be claimed superior until complete evaluation is performed.",
    ]))
    story.append(p("10. Conclusion", "Heading1"))
    story.append(p("Experiment 3 successfully completed dataset preparation and training for a 128 x 128 conditional DDPM using SDNET2018 plus 10,519 external cracked reinforced-concrete images. It is the strongest visual-resolution experiment so far, but its formal quality metrics remain pending. The correct current conclusion is that the model is trained and visually inspectable, not fully evaluated."))
    story.append(p("11. Recommended Next Steps", "Heading1"))
    story.append(bullets([
        "Generate complete 1,000-image cracked and non-cracked evaluation sets.",
        "Run FID, KID, and LPIPS against the same real validation comparison protocol.",
        "Create manual visual audit sheets for 128 x 128 cracked and non-cracked outputs.",
        "Compare visual crack clarity against Experiment 2 after full evaluation is complete.",
    ]))

    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4, rightMargin=0.62 * inch, leftMargin=0.62 * inch, topMargin=0.62 * inch, bottomMargin=0.65 * inch, title="Experiment 3 128x128 Conditional Augmented DDPM Report", author="Codex")
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(PDF_PATH)


if __name__ == "__main__":
    build()
