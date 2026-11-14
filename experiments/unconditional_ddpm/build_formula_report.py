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
EXP = ROOT / "experiments" / "unconditional_ddpm"
OUT = EXP / "reports"
OUT.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUT / "experiment_1_unconditional_ddpm_formula_report.pdf"
EQUATION_DIR = OUT / "equations"
EQUATION_DIR.mkdir(parents=True, exist_ok=True)
EQUATION_COUNTER = 0


def styles():
    base = getSampleStyleSheet()
    base["Normal"].fontName = "Helvetica"
    base["Normal"].fontSize = 9.5
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
            fontSize=8.3,
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
    height_per_line = 0.34
    fig_height = max(0.55, height_per_line * len(lines) + 0.16)
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
                style = "Heading2" if row_index == 0 else "Normal"
                wrapped_row.append(Paragraph(cell, S[style]))
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
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
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
    canvas.drawString(0.65 * inch, 0.42 * inch, "Experiment 1 - Unconditional DDPM Formula Report")
    canvas.drawRightString(A4[0] - 0.65 * inch, 0.42 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build():
    story = []
    story.append(p("Experiment 1 Formula Report", "Title"))
    story.append(p("Unconditional DDPM for Concrete Surface Image Generation", "Title"))
    story.append(Spacer(1, 10))
    story.append(
        p(
            "This report documents the complete first experiment in the concrete crack image generation project. "
            "It includes the aim, dataset, preprocessing, DDPM theory, training equations, sampling equations, "
            "evaluation metric formulas, observations, results, limitations, and recommended next steps."
        )
    )
    story.append(Spacer(1, 10))
    story.append(
        table(
            [
                ["Item", "Value"],
                ["Dataset", "SDNET2018"],
                ["Task", "Unconditional concrete surface image generation"],
                ["Resolution", "64 x 64 RGB"],
                ["Model", "SmallUNet unconditional DDPM"],
                ["Epochs", "50"],
                ["Diffusion steps", "1000"],
                ["Final checkpoint", r"experiments\unconditional_ddpm\checkpoints\ddpm_epoch_050.pt"],
                ["Final logged loss", "0.01599615436508108"],
            ],
            [1.55 * inch, 4.85 * inch],
        )
    )
    story.append(PageBreak())

    story.append(p("1. Aim and Motivation", "Heading1"))
    story.append(
        p(
            "The aim was to train a diffusion model that generates new concrete surface patches from random noise. "
            "The model was intentionally unconditional: it received no cracked/non-cracked label during generation. "
            "This makes the experiment a baseline for later controlled models."
        )
    )
    story.append(
        bullets(
            [
                "Establish a baseline for concrete image synthesis.",
                "Check whether a compact DDPM can learn SDNET2018 concrete texture at 64 x 64 resolution.",
                "Produce a quantitative reference point using FID, KID, and LPIPS.",
                "Identify why unconditional generation is limited for crack-specific image synthesis.",
            ]
        )
    )

    story.append(p("2. Dataset and Preprocessing", "Heading1"))
    story.append(p("2.1 Dataset Used", "Heading2"))
    story.append(
        p(
            "Experiment 1 used SDNET2018, a concrete surface image dataset containing patch-level images from "
            "bridge decks, pavement, and walls. The dataset contains cracked and non-cracked examples, but those "
            "labels were used only as metadata in this unconditional experiment."
        )
    )
    story.append(
        table(
            [
                ["Quantity", "Value"],
                ["Total indexed images", "56,092"],
                ["Training images", "50,482"],
                ["Validation images", "5,610"],
                ["Surface categories", "bridge_deck, pavement, wall"],
                ["Processed training path", r"data\processed\train"],
                ["Processed validation path", r"data\processed\validation"],
                ["Metadata files", r"data\processed\train.csv, data\processed\validation.csv"],
            ],
            [2.1 * inch, 4.3 * inch],
        )
    )
    story.append(p("2.2 Dataset Folder Structure", "Heading2"))
    story.append(
        p(
            "SDNET2018 is organized by concrete surface type first and crack class second. The raw folder names encode "
            "both pieces of information. The experiment decoded these folder names into human-readable metadata before "
            "training."
        )
    )
    story.append(
        table(
            [
                ["Raw folder", "Decoded surface", "Class folder", "Decoded label", "Meaning"],
                ["D", "bridge_deck", "CD", "cracked", "Bridge deck image patch containing visible cracking"],
                ["D", "bridge_deck", "UD", "non_cracked", "Bridge deck image patch without visible cracking"],
                ["P", "pavement", "CP", "cracked", "Pavement image patch containing visible cracking"],
                ["P", "pavement", "UP", "non_cracked", "Pavement image patch without visible cracking"],
                ["W", "wall", "CW", "cracked", "Wall image patch containing visible cracking"],
                ["W", "wall", "UW", "non_cracked", "Wall image patch without visible cracking"],
            ],
            [0.75 * inch, 1.15 * inch, 0.85 * inch, 1.05 * inch, 2.6 * inch],
        )
    )
    story.append(
        p(
            "The complete raw dataset therefore contains six semantic groups: cracked bridge deck, non-cracked bridge "
            "deck, cracked pavement, non-cracked pavement, cracked wall, and non-cracked wall. This structure matters "
            "because concrete texture differs by surface type, and crack appearance differs from non-cracked surface texture."
        )
    )
    story.append(p("2.3 Cracked vs Non-Cracked Image Meaning", "Heading2"))
    story.append(
        table(
            [
                ["Class", "Visual content", "Role in this experiment"],
                ["Cracked", "A concrete patch where a visible crack line, crack branch, dark fissure, or separated surface trace is present. The crack may be thin, thick, straight, branched, dark, or partially blended with rough texture.", "Included in the training distribution, but not supplied as a generation condition."],
                ["Non-cracked", "A concrete patch without a visible crack. It may still contain aggregate, pores, stains, shadows, roughness, color variation, or surface marks that are not labeled as cracks.", "Included in the same unconditional training distribution as cracked images."],
            ],
            [1.15 * inch, 3.1 * inch, 2.15 * inch],
        )
    )
    story.append(
        p(
            "The difference between cracked and non-cracked images is therefore semantic, not merely numerical. Both "
            "classes can contain rough concrete texture, noise, aggregate, and illumination variation. The cracked class "
            "is distinguished by the presence of an actual visible crack pattern. This is why FID, KID, and LPIPS alone "
            "cannot prove that generated images contain correct cracks."
        )
    )
    story.append(p("2.4 Dataset Composition by Surface and Label", "Heading2"))
    story.append(
        table(
            [
                ["Surface", "Cracked total", "Non-cracked total", "Train cracked", "Train non-cracked", "Validation cracked", "Validation non-cracked"],
                ["bridge_deck", "2,025", "11,595", "1,823", "10,435", "202", "1,160"],
                ["pavement", "2,608", "21,726", "2,347", "19,553", "261", "2,173"],
                ["wall", "3,851", "14,287", "3,466", "12,858", "385", "1,429"],
                ["Total", "8,484", "47,608", "7,636", "42,846", "848", "4,762"],
            ],
            [1.05 * inch, 0.85 * inch, 1.0 * inch, 0.85 * inch, 1.0 * inch, 0.9 * inch, 1.0 * inch],
        )
    )
    story.append(
        p(
            "The dataset is imbalanced: non-cracked images are much more common than cracked images. For Experiment 1, "
            "this imbalance was not corrected with class conditioning or class balancing because the purpose was to learn "
            "the mixed unconditional image distribution. This also explains why an unconditional model may generate "
            "non-cracked-looking concrete more often than clearly cracked concrete."
        )
    )
    story.append(p("2.5 Preprocessing Steps and Necessity", "Heading2"))
    story.append(
        table(
            [
                ["Step", "What was done", "Why it was necessary"],
                ["Raw image discovery", "The script recursively scanned data\\raw\\SDNET2018\\images and accepted .jpg, .jpeg, and .png files.", "This created a complete, reproducible list of usable image files and excluded non-image files."],
                ["Folder-label decoding", "Surface labels were decoded from D, P, and W folders as bridge_deck, pavement, and wall. Crack labels were decoded from class folders beginning with C or U.", "SDNET2018 stores class information in folder names, so the labels had to be decoded before splitting and metadata creation."],
                ["Stratified train/validation split", "Records were grouped by (surface, crack label), shuffled with seed 42, and split with a 10 percent validation ratio.", "This prevented the validation set from being dominated by only one surface or crack class and made the split reproducible."],
                ["CSV split files", "The split files all.csv, train.csv, and validation.csv were written under data\\splits.", "CSV files make the dataset state auditable and allow the same split to be reused without repeating random sampling."],
                ["RGB conversion", "Every image was opened with PIL and converted to RGB.", "The U-Net expects consistent 3-channel tensors. RGB conversion avoids failures from grayscale, palette, or mode-inconsistent images."],
                ["Resize to 64 x 64", "Every image was resized to 64 x 64 using LANCZOS interpolation.", "Uniform size is required for batching. LANCZOS was chosen because it preserves fine detail better than nearest-neighbor resizing."],
                ["Stable processed filename", "Each processed filename was generated from the SHA1 hash of the original relative path, truncated to 12 hex characters, and saved as PNG.", "This avoided filename collisions while preserving deterministic mapping from original images to processed images."],
                ["Processed metadata", "Processed CSV rows stored path, original_path, surface, label, width, and height.", "This preserved traceability from processed files back to raw SDNET2018 images."],
                ["Training tensor normalization", "During loading, 8-bit image values were converted by ToTensor from [0,255] to [0,1], then normalized with mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5).", "This maps the original image intensity range to approximately [-1,1], which is standard and numerically stable for diffusion training."],
                ["Unconditional label handling", "Crack labels were retained in metadata but not passed to the DDPM as conditioning input.", "This preserved labels for later analysis while keeping Experiment 1 a true unconditional baseline."],
            ],
            [1.1 * inch, 2.3 * inch, 3.0 * inch],
        )
    )
    story.append(p("2.6 Preprocessing Pipeline Formula View", "Heading2"))
    story.append(
        p(
            "For each raw image, the preprocessing pipeline can be summarized as the following deterministic mapping. "
            "The image is decoded, converted to RGB, resized, saved as a processed PNG, and later normalized when loaded "
            "for training."
        )
    )
    story.append(
        formula(
            "raw_path -> decode folder labels -> split assignment\n"
            "image_raw -> RGB(image_raw) -> resize_64x64_LANCZOS -> image_png\n"
            "uint8 pixel p in [0,255]\n"
            "tensor = ToTensor(image_png) = p / 255 in [0,1]\n"
            "x_0 = (tensor - 0.5) / 0.5 = 2 * tensor - 1 in [-1,1]"
        )
    )
    story.append(
        p(
            "The resize step is useful computationally, but it is also a limitation: very thin cracks may become blurred "
            "or disappear at 64 x 64 resolution. This is one reason later experiments moved toward conditional and higher "
            "resolution generation."
        )
    )

    story.append(PageBreak())
    story.append(p("3. Experimental Procedure", "Heading1"))
    story.append(
        p(
            "This section describes the actual procedure followed in Experiment 1. The purpose of writing the procedure "
            "explicitly is to make the experiment reproducible and to separate practical execution from the mathematical "
            "theory of DDPMs."
        )
    )
    story.append(p("3.1 Raw Dataset Preparation", "Heading2"))
    story.append(
        bullets(
            [
                r"The SDNET2018 image dataset was placed under data\raw\SDNET2018\images.",
                "The dataset was not manually relabeled for this experiment; the original SDNET2018 folder structure was used.",
                "Surface type and crack status were decoded from folder names during indexing.",
                "The raw dataset was kept separate from processed images so the preprocessing pipeline remained reproducible.",
            ]
        )
    )
    story.append(p("3.2 Dataset Indexing and Splitting", "Heading2"))
    story.append(
        p(
            "The indexing stage used the shared source script src/index_dataset.py. It scanned all eligible image files, "
            "decoded labels, grouped records by surface and crack status, and created reproducible split CSV files."
        )
    )
    story.append(
        table(
            [
                ["Procedure item", "Value"],
                ["Indexing script", r"src\index_dataset.py"],
                ["Input directory", r"data\raw\SDNET2018\images"],
                ["Random seed", "42"],
                ["Validation ratio", "0.10"],
                ["Split method", "Stratified by surface and crack label"],
                ["Output split files", r"data\splits\all.csv, data\splits\train.csv, data\splits\validation.csv"],
            ],
            [2.1 * inch, 4.3 * inch],
        )
    )
    story.append(p("3.3 Image Preprocessing", "Heading2"))
    story.append(
        p(
            "The preprocessing stage used src/preprocess_images.py. Each raw image was converted to RGB, resized to "
            "64 x 64 using LANCZOS interpolation, saved as a PNG, and recorded in processed metadata CSV files."
        )
    )
    story.append(
        bullets(
            [
                "Training images were written to data\\processed\\train.",
                "Validation images were written to data\\processed\\validation.",
                "Processed metadata was written to data\\processed\\train.csv and data\\processed\\validation.csv.",
                "Each processed file used a deterministic SHA1-based filename derived from the original relative path.",
                "Original paths and labels were preserved in the processed CSV files for traceability.",
            ]
        )
    )
    story.append(p("3.4 Dataset Loading for Training", "Heading2"))
    story.append(
        p(
            "During training, src/dataset.py loaded processed PNG images using the processed CSV metadata. PIL opened each "
            "image, converted it to RGB again as a safety step, converted it to a tensor, and normalized it for DDPM input."
        )
    )
    story.append(
        formula(
            "uint8 pixel p in [0,255]\n"
            "ToTensor: p -> p / 255 in [0,1]\n"
            "Normalize(mean=0.5, std=0.5): x_0 = 2 * (p / 255) - 1 in [-1,1]"
        )
    )
    story.append(p("3.5 Model Training", "Heading2"))
    story.append(
        p(
            "The unconditional DDPM was trained using src/train.py. The compact U-Net received noisy images and diffusion "
            "timesteps, but no crack-status label. For each batch, the training script sampled random timesteps, added "
            "Gaussian noise according to the diffusion schedule, predicted the noise with the U-Net, and optimized mean "
            "squared error between actual and predicted noise."
        )
    )
    story.append(
        table(
            [
                ["Training item", "Value"],
                ["Training script", r"src\train.py"],
                ["Model", "SmallUNet unconditional DDPM"],
                ["Input images", r"data\processed\train"],
                ["Condition input", "None"],
                ["Diffusion steps", "1,000"],
                ["Epochs", "50"],
                ["Optimizer", "AdamW"],
                ["Training loss", "Mean squared error on predicted noise"],
                ["Final checkpoint", r"experiments\unconditional_ddpm\checkpoints\ddpm_epoch_050.pt"],
            ],
            [1.8 * inch, 4.6 * inch],
        )
    )
    story.append(p("3.6 Sample Generation", "Heading2"))
    story.append(
        p(
            "After training, the final checkpoint was used to generate visual reference grids. Different random seeds were "
            "used to check whether the model produced varied concrete-like images."
        )
    )
    story.append(
        table(
            [
                ["Generated reference", "Purpose"],
                [r"outputs\samples\samples_epoch_050_seed_42.png", "Reference grid from final checkpoint"],
                [r"outputs\samples\samples_epoch_050_seed_123.png", "Reference grid with different random seed"],
                [r"outputs\samples\samples_epoch_050_seed_999.png", "Reference grid with different random seed"],
            ],
            [3.4 * inch, 3.0 * inch],
        )
    )
    story.append(p("3.7 Evaluation Set Generation", "Heading2"))
    story.append(
        p(
            "A larger generated set of 1,000 images was created from the epoch 50 checkpoint for formal metric evaluation. "
            "These images were stored separately from the small reference grids."
        )
    )
    story.append(
        table(
            [
                ["Evaluation artifact", "Value"],
                ["Generated image count", "1,000"],
                ["Generated image directory", r"outputs\evaluation\unconditional_epoch_050"],
                ["Real comparison set", "1,000 validation images"],
                ["Evaluation purpose", "FID, KID, LPIPS, and lightweight diagnostics"],
            ],
            [2.2 * inch, 4.2 * inch],
        )
    )
    story.append(p("3.8 Quantitative Evaluation", "Heading2"))
    story.append(
        p(
            "The generated evaluation set was compared against real validation images using src/evaluate_metrics.py. "
            "The formal metrics measured distributional and perceptual similarity. A separate lightweight diagnostic "
            "run measured simple pixel statistics and rough diversity/nearest-real distances."
        )
    )
    story.append(
        bullets(
            [
                "FID measured deep-feature distribution distance between generated and real images.",
                "KID measured kernel MMD distance between generated and real feature sets.",
                "Paired LPIPS measured perceptual feature distance as a diagnostic.",
                "Lightweight diagnostics measured mean pixel value, pixel standard deviation, pairwise generated distance, and nearest-real pixel distance.",
            ]
        )
    )
    story.append(p("3.9 Visual Audit Preparation", "Heading2"))
    story.append(
        p(
            "Because the formal metrics do not directly measure crack visibility, a manual visual-audit starter grid and "
            "CSV were created for the first 64 generated evaluation images. This does not complete visual evaluation, but "
            "it prepares the experiment for human crack-presence and texture-quality annotation."
        )
    )
    story.append(
        table(
            [
                ["Manual audit artifact", "Purpose"],
                [r"outputs\audit\unconditional_epoch_050_audit_grid_first_64.png", "Visual review sheet for first 64 generated evaluation images"],
                [r"outputs\audit\unconditional_epoch_050_manual_audit_first_64.csv", "Blank annotation template for visible crack and texture quality"],
            ],
            [3.6 * inch, 2.8 * inch],
        )
    )
    story.append(p("3.10 Procedure Summary", "Heading2"))
    story.append(
        table(
            [
                ["Order", "Step", "Output"],
                ["1", "Index raw SDNET2018 images", r"data\splits\*.csv"],
                ["2", "Preprocess images to 64 x 64 RGB PNG", r"data\processed\train and data\processed\validation"],
                ["3", "Load tensors and normalize to [-1,1]", "Training-ready image tensors"],
                ["4", "Train unconditional DDPM for 50 epochs", r"checkpoints\ddpm_epoch_050.pt"],
                ["5", "Generate sample grids", r"outputs\samples\*.png"],
                ["6", "Generate 1,000-image evaluation set", r"outputs\evaluation\unconditional_epoch_050"],
                ["7", "Compute quantitative metrics", "FID, KID, LPIPS"],
                ["8", "Create visual audit starter files", r"outputs\audit\*.png and *.csv"],
            ],
            [0.55 * inch, 2.55 * inch, 3.3 * inch],
        )
    )

    story.append(PageBreak())
    story.append(p("4. DDPM Theory", "Heading1"))
    story.append(p("4.1 Data Distribution", "Heading2"))
    story.append(
        p(
            "Let x_0 be a real concrete image sampled from the data distribution q(x_0). "
            "The goal of generative modeling is to learn a model distribution p_theta(x_0) that produces samples "
            "that resemble q(x_0)."
        )
    )
    story.append(math_block([r"x_0 \sim q(x_0)", r"\hat{x}_0 \sim p_\theta(x_0)"]))

    story.append(p("4.2 Forward Diffusion Process", "Heading2"))
    story.append(
        p(
            "The forward process gradually adds Gaussian noise to the real image. At timestep t, the noisy image is x_t. "
            "A variance schedule beta_t controls how much noise is added at each step."
        )
    )
    story.append(math_block([r"q(x_t \mid x_{t-1})=\mathcal{N}\!\left(x_t;\sqrt{1-\beta_t}\,x_{t-1},\beta_t I\right)"]))
    story.append(
        p(
            "Define alpha_t = 1 - beta_t and alpha_bar_t as the cumulative product of alphas from step 1 to t. "
            "This allows direct sampling of x_t from x_0 without iterating through every intermediate step."
        )
    )
    story.append(math_block([r"\alpha_t=1-\beta_t", r"\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s"]))
    story.append(math_block([r"q(x_t \mid x_0)=\mathcal{N}\!\left(x_t;\sqrt{\bar{\alpha}_t}\,x_0,(1-\bar{\alpha}_t)I\right)"]))
    story.append(math_block([r"x_t=\sqrt{\bar{\alpha}_t}\,x_0+\sqrt{1-\bar{\alpha}_t}\,\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I)"]))

    story.append(p("4.3 Reverse Denoising Process", "Heading2"))
    story.append(
        p(
            "The reverse process learns to remove noise. Since the exact reverse distribution is unknown, a neural "
            "network parameterized by theta approximates it."
        )
    )
    story.append(math_block([r"p_\theta(x_{t-1}\mid x_t)=\mathcal{N}\!\left(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t)\right)"]))
    story.append(
        p(
            "In this experiment, the U-Net predicts the noise epsilon_theta(x_t, t). The predicted noise is then used "
            "to estimate the mean of the reverse distribution."
        )
    )
    story.append(
        math_block([r"\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(x_t,t)\right)"])
    )

    story.append(p("4.4 Training Objective", "Heading2"))
    story.append(
        p(
            "The DDPM training objective used here is mean squared error between the actual sampled Gaussian noise "
            "and the noise predicted by the U-Net."
        )
    )
    story.append(math_block([r"L_{\mathrm{simple}}(\theta)=\mathbb{E}_{x_0,\epsilon,t}\left[\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|_2^2\right]"]))
    story.append(
        p(
            "This objective is necessary because it teaches the network how to denoise images at arbitrary diffusion "
            "timesteps. A lower value means better noise prediction, but it is not by itself a complete image-quality metric."
        )
    )

    story.append(p("4.5 Sampling Equation", "Heading2"))
    story.append(
        p(
            "During generation, the model starts from pure Gaussian noise x_T and repeatedly applies the learned reverse "
            "step until x_0 is obtained."
        )
    )
    story.append(math_block([r"x_T\sim\mathcal{N}(0,I)"]))
    story.append(
        math_block(
            [
                r"x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(x_t,t)\right)+\sigma_t z",
                r"z\sim\mathcal{N}(0,I)\ \mathrm{for}\ t>1,\qquad z=0\ \mathrm{for}\ t=1",
            ]
        )
    )
    story.append(
        p(
            "This step is repeated from t = T down to t = 1. Because the model is unconditional, the prediction function "
            "epsilon_theta(x_t, t) receives no class label."
        )
    )

    story.append(PageBreak())
    story.append(p("5. Model and Training Configuration", "Heading1"))
    story.append(
        table(
            [
                ["Component", "Configuration"],
                ["Architecture", "Compact U-Net DDPM"],
                ["Parameters", "3,258,115"],
                ["Image size", "64 x 64 RGB"],
                ["Diffusion steps", "1,000"],
                ["Loss", "MSE between actual and predicted Gaussian noise"],
                ["Optimizer", "AdamW"],
                ["Training epochs", "50"],
                ["GPU", "NVIDIA GeForce RTX 4050 Laptop GPU"],
            ],
            [2.0 * inch, 4.4 * inch],
        )
    )
    story.append(
        p(
            "A compact model was appropriate because this was a baseline at low resolution. The objective was not to "
            "maximize model capacity, but to establish a stable first DDPM that could be trained and evaluated efficiently."
        )
    )

    story.append(p("6. Evaluation Metric Theory and Formulas", "Heading1"))
    story.append(p("6.1 FID", "Heading2"))
    story.append(
        p(
            "FID, or Frechet Inception Distance, compares the feature distribution of real images with the feature "
            "distribution of generated images. Images are passed through a pretrained feature extractor, and the feature "
            "vectors are modeled as multivariate Gaussians."
        )
    )
    story.append(math_block([r"f_r\sim\mathcal{N}(\mu_r,\Sigma_r)", r"f_g\sim\mathcal{N}(\mu_g,\Sigma_g)"]))
    story.append(math_block([r"\mathrm{FID}=\left\|\mu_r-\mu_g\right\|_2^2+\mathrm{Tr}\!\left(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\right)"]))
    story.append(
        p(
            "Lower FID is better. It measures general distribution similarity, not crack correctness. A generated image set "
            "can receive a good FID even if it does not contain clear cracks."
        )
    )

    story.append(p("6.2 KID", "Heading2"))
    story.append(
        p(
            "KID, or Kernel Inception Distance, is based on Maximum Mean Discrepancy between real and generated feature "
            "sets. It is often useful for smaller sample sizes because its estimator is less biased than FID."
        )
    )
    story.append(
        math_block(
            [
                r"\mathrm{MMD}^2(X,Y)=\frac{1}{m(m-1)}\sum_{i\ne j}k(x_i,x_j)",
                r"+\frac{1}{n(n-1)}\sum_{i\ne j}k(y_i,y_j)-\frac{2}{mn}\sum_i\sum_j k(x_i,y_j)",
            ]
        )
    )
    story.append(math_block([r"\mathrm{KID}=\mathrm{MMD}^2(F_{\mathrm{real}},F_{\mathrm{generated}})"]))
    story.append(
        p(
            "The kernel k is commonly a polynomial kernel. Lower KID is better. KID mean reports the average estimate, "
            "while KID standard deviation reports variability across repeated subset comparisons."
        )
    )

    story.append(p("6.3 LPIPS", "Heading2"))
    story.append(
        p(
            "LPIPS, or Learned Perceptual Image Patch Similarity, compares images using deep neural-network feature maps. "
            "It is designed to align better with human perceptual differences than raw pixel distance."
        )
    )
    story.append(
        math_block([r"\mathrm{LPIPS}(x,y)=\sum_l\frac{1}{H_lW_l}\sum_{h,w}\left\|w_l\odot\left(\phi_l(x)_{h,w}-\phi_l(y)_{h,w}\right)\right\|_2^2"])
    )
    story.append(
        p(
            "Here phi_l denotes a feature map at layer l, and w_l is a learned channel weighting. Lower LPIPS means two "
            "images are perceptually closer. In this unconditional experiment, paired LPIPS is only a diagnostic because "
            "each generated image is not meant to reconstruct a specific real image."
        )
    )

    story.append(p("6.4 Lightweight Diagnostic Proxies", "Heading2"))
    story.append(
        p(
            "Additional proxy measurements were computed to compare simple pixel statistics and rough diversity. These "
            "are sanity checks only and should not replace standard metrics."
        )
    )
    story.append(math_block([r"\mu_x=\frac{1}{NCHW}\sum_i\sum_c\sum_h\sum_w x_{i,c,h,w}"]))
    story.append(math_block([r"\sigma_x=\sqrt{\frac{1}{NCHW}\sum_i\sum_c\sum_h\sum_w(x_{i,c,h,w}-\mu_x)^2}"]))
    story.append(math_block([r"d_{\mathrm{pair}}=\frac{2}{M(M-1)}\sum_{i<j}\left\|g_i-g_j\right\|_2"]))
    story.append(math_block([r"d_{\mathrm{nearest}}=\frac{1}{M}\sum_i\min_j\left\|g_i-r_j\right\|_2"]))

    story.append(p("7. Results", "Heading1"))
    story.append(
        table(
            [
                ["Metric", "Value", "Interpretation"],
                ["Final training loss", "0.01599615436508108", "The model learned the denoising objective."],
                ["FID", "5.604571", "Generated and real image distributions are close in deep feature space."],
                ["KID mean", "3.419779", "Distributional distance estimate; lower is better."],
                ["KID std", "1.028021", "Variability of KID estimate across subsets."],
                ["Paired LPIPS mean", "0.548679", "Perceptual distance diagnostic, not a reconstruction guarantee."],
            ],
            [1.45 * inch, 1.25 * inch, 3.7 * inch],
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        table(
            [
                ["Lightweight diagnostic", "Generated", "Real / comparison", "Meaning"],
                ["Mean pixel value", "0.5682", "0.6330", "Generated images are darker on average."],
                ["Pixel standard deviation", "0.1513", "0.1117", "Generated images have higher contrast variation."],
                ["Mean pairwise generated distance", "20.5188", "-", "Rough diversity signal inside generated set."],
                ["Mean nearest-real pixel distance", "11.1758", "-", "Crude nearest-neighbor distance to real validation images."],
            ],
            [1.7 * inch, 1.0 * inch, 1.15 * inch, 2.55 * inch],
        )
    )

    story.append(p("8. Observations", "Heading1"))
    story.append(
        bullets(
            [
                "The generated samples learned concrete-like color, aggregate texture, and surface roughness.",
                "Crack visibility was inconsistent because the model was trained without crack-status conditioning.",
                "Some generated structures appear crack-like, but there is no reliable control over crack presence.",
                "The 64 x 64 resolution likely removes or blurs thin crack details.",
                "The formal metrics support general image similarity, not engineering validity.",
            ]
        )
    )

    story.append(PageBreak())
    story.append(p("9. Reference Generated Images", "Heading1"))
    story.append(
        p(
            "The following reference images are the generated sample grids from the final epoch. They are included as "
            "compact visual references only. The full 1,000 generated evaluation images are not embedded in this report."
        )
    )
    sample_paths = [
        EXP / "outputs" / "samples" / "samples_epoch_050_seed_42.png",
        EXP / "outputs" / "samples" / "samples_epoch_050_seed_123.png",
        EXP / "outputs" / "samples" / "samples_epoch_050_seed_999.png",
    ]
    for sample_path in sample_paths:
        if sample_path.exists():
            img = Image(str(sample_path), width=3.1 * inch, height=3.1 * inch)
            img.hAlign = "CENTER"
            story.append(img)
            story.append(p(sample_path.name, "Caption"))
            story.append(Spacer(1, 7))

    audit = EXP / "outputs" / "audit" / "unconditional_epoch_050_audit_grid_first_64.png"
    if audit.exists():
        story.append(PageBreak())
        story.append(p("10. Visual Audit Starter", "Heading1"))
        story.append(
            p(
                "A visual audit grid was created for the first 64 generated evaluation images. This grid is intended "
                "for manual annotation of visible crack presence and texture quality."
            )
        )
        img = Image(str(audit), width=5.6 * inch, height=5.6 * inch)
        img.hAlign = "CENTER"
        story.append(img)
        story.append(p("Manual audit grid for the first 64 unconditional generated samples.", "Caption"))

    story.append(PageBreak())
    story.append(p("11. Limitations", "Heading1"))
    story.append(
        bullets(
            [
                "The model is unconditional and cannot be instructed to generate cracked or non-cracked images.",
                "No completed human crack-presence audit exists yet.",
                "FID, KID, and LPIPS do not measure crack semantics or structural plausibility.",
                "The low resolution may suppress thin crack details.",
                "The generated images are patch-level samples, not full structural inspection scenes.",
                "The existing lightweight diagnostics are not replacements for formal evaluation or human review.",
            ]
        )
    )

    story.append(p("12. Conclusion", "Heading1"))
    story.append(
        p(
            "Experiment 1 successfully established an unconditional DDPM baseline for concrete surface image generation. "
            "The model learned the broad visual distribution of SDNET2018 concrete patches and achieved a strong general "
            "distribution match under FID, KID, and LPIPS-style evaluation. However, the experiment also demonstrates that "
            "unconditional generation is not sufficient for controlled crack synthesis. Crack visibility remains inconsistent "
            "because the model receives no crack-status condition."
        )
    )
    story.append(
        p(
            "Therefore, Experiment 1 should be treated as a concrete texture and image realism baseline. It is useful for "
            "comparison, but conditional or mask-conditioned diffusion is needed for controlled crack generation."
        )
    )

    story.append(p("13. Recommended Next Steps", "Heading1"))
    story.append(
        bullets(
            [
                r"Manually annotate outputs\audit\unconditional_epoch_050_manual_audit_first_64.csv.",
                "Compute visible crack rate, usable texture rate, and artifact rate from the manual audit.",
                "Use Experiment 1 as the baseline when comparing conditional DDPM experiments.",
                "Avoid claiming crack-generation reliability from FID, KID, or LPIPS alone.",
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
        title="Experiment 1 Formula Report",
        author="Codex",
    )
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(PDF_PATH)


if __name__ == "__main__":
    build()
