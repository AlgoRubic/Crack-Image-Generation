# Experiment 1 Full Report: Unconditional DDPM for Concrete Surface Image Generation

## 1. Experiment Identity

Experiment name: Unconditional DDPM for Concrete Surface Image Generation

Experiment number: Experiment 1

Experiment type: Baseline generative image modeling

Dataset: SDNET2018

Resolution: 64 x 64 RGB

Model: SmallUNet unconditional DDPM

Training duration: 50 epochs

Project root:

```text
D:\Downloads\concrete_ddpm
```

Experiment directory:

```text
D:\Downloads\concrete_ddpm\experiments\unconditional_ddpm
```

Final checkpoint:

```text
D:\Downloads\concrete_ddpm\experiments\unconditional_ddpm\checkpoints\ddpm_epoch_050.pt
```

Generated evaluation images:

```text
D:\Downloads\concrete_ddpm\experiments\unconditional_ddpm\outputs\evaluation\unconditional_epoch_050
```

## 2. Aim of the Experiment

The aim of Experiment 1 was to train a diffusion model that can generate new concrete surface image patches from random noise. The model was trained without any explicit crack-status condition. This means the model was not told whether an image was cracked or non-cracked during generation. Instead, it learned the overall visual distribution of concrete surface patches from the full SDNET2018 training dataset.

The main goals were:

1. To establish a baseline generative model for concrete surface images.
2. To test whether a compact DDPM can learn realistic concrete texture at 64 x 64 resolution.
3. To measure the general similarity between generated images and real validation images.
4. To identify the limitations of unconditional generation for concrete crack synthesis.
5. To create a reference point for later conditional experiments.

This experiment is important because every later improvement needs a baseline. Without an unconditional baseline, it would be difficult to know whether a conditional model, a higher-resolution model, or an augmented-data model is genuinely better.

## 3. Background and Theory

### 3.1 Generative Modeling

Generative modeling is a machine-learning approach where the model learns the probability distribution of a dataset and then produces new samples that resemble the original data. In this project, the samples are concrete surface image patches.

For a concrete crack project, generative modeling is useful because it may help produce additional image examples for inspection, data augmentation, model testing, and research into crack-image synthesis. However, simply generating concrete-like texture is not the same as generating meaningful crack images. This distinction is central to Experiment 1.

### 3.2 Diffusion Models

A diffusion model learns to generate images by reversing a gradual noising process.

The forward process starts with a real image and adds Gaussian noise over many steps until the image becomes nearly pure noise. The reverse process trains a neural network to remove noise step by step. During generation, the model starts from random noise and repeatedly denoises it until an image appears.

The main idea is:

```text
real image -> progressively noisier image -> pure noise
pure noise -> learned denoising steps -> generated image
```

The model does not generate the full image in one step. It learns a sequence of small denoising operations.

### 3.3 DDPM

DDPM stands for Denoising Diffusion Probabilistic Model. It is one of the classical diffusion-model formulations.

In this experiment, the DDPM used 1,000 diffusion steps. During training, a random timestep was selected, noise was added to an image according to that timestep, and the neural network was trained to predict the noise that had been added.

The training target was therefore not the clean image directly. The model learned to predict Gaussian noise. Once the model can predict noise well, it can be used in reverse to gradually remove noise from a random starting image.

### 3.4 Why Use a U-Net?

A U-Net is commonly used in image-to-image and diffusion tasks because it combines local detail with broader spatial context.

The encoder part reduces image resolution and learns higher-level structure. The decoder part reconstructs spatial detail. Skip connections help preserve low-level texture information. This is especially useful for concrete images because the model needs to learn fine surface roughness, aggregate patterns, stains, edges, and possible crack-like structures.

The SmallUNet used here is compact, with 3,258,115 parameters. This size was appropriate for the available GPU and for 64 x 64 image generation.

### 3.5 Unconditional Generation

An unconditional model generates images without receiving a class label, text prompt, mask, or other control signal. It learns the entire training distribution as one mixed distribution.

In this experiment, cracked and non-cracked images were both included in the training data, but their labels were not supplied to the model as generation conditions.

This matters because the model cannot be instructed to generate:

```text
cracked concrete
```

or:

```text
non-cracked concrete
```

It only learns:

```text
concrete surface image
```

As a result, the generated output may sometimes contain crack-like features, but there is no mechanism to control crack presence.

## 4. Necessity of Each Major Step

### 4.1 Dataset Indexing

The SDNET2018 dataset contains concrete images from multiple surface categories. Indexing was necessary to create a structured list of image paths and metadata.

Without indexing, the training code would not have a reliable way to know which files belong to the dataset, how many images are available, or how to separate training and validation data.

### 4.2 Train/Validation Splitting

The dataset was split into training and validation sets:

```text
Training images:   50,482
Validation images:  5,610
Total indexed:     56,092
```

The training set was used to optimize the DDPM. The validation set was held back for comparison during evaluation.

This step was necessary because evaluating generated images against the same images used for training would make the results less trustworthy. A separate validation set gives a better estimate of how well the generated image distribution matches unseen real data.

### 4.3 Image Preprocessing

Images were processed into a consistent format:

```text
64 x 64 RGB
```

This was necessary because neural networks require uniform tensor dimensions within a batch. The 64 x 64 resolution was chosen as a practical first baseline because it reduces GPU memory demand and training time.

The tradeoff is that thin cracks may become less visible after resizing. This limitation is important for crack-generation work.

### 4.4 Preserving Labels as Metadata

Labels such as cracked and non-cracked were preserved in the dataset metadata but were not supplied to the unconditional DDPM.

This was necessary because Experiment 1 was designed specifically as an unconditional baseline. The purpose was to test whether the model could learn the overall concrete image distribution without class control.

The consequence is that the model cannot be directly evaluated as a controlled crack generator.

### 4.5 Model Training

Training was necessary so the SmallUNet could learn to predict noise at different diffusion timesteps. The model was trained for 50 epochs using mean squared error between the actual Gaussian noise and the predicted noise.

The final logged training loss was:

```text
0.01599615436508108
```

This indicates that the model learned the denoising objective. However, training loss alone is not enough to judge generated image quality. A model can have a low denoising loss and still produce images with visual artifacts or weak crack detail.

### 4.6 Checkpointing

Checkpoints were saved during training, with the final checkpoint stored at:

```text
checkpoints\ddpm_epoch_050.pt
```

Checkpointing was necessary because training is computationally expensive. It allows the experiment to be resumed, sampled, evaluated, and compared without retraining from the beginning.

### 4.7 Sample Generation

Sample grids were generated after training:

```text
outputs\samples\samples_epoch_050_seed_42.png
outputs\samples\samples_epoch_050_seed_123.png
outputs\samples\samples_epoch_050_seed_999.png
```

This step was necessary for initial qualitative inspection. Different seeds produce different generated samples from the same trained model. Multiple seeds help reveal whether the model is diverse or whether it repeatedly produces similar images.

### 4.8 Formal Evaluation Set Generation

A larger generated evaluation set of 1,000 images was created:

```text
outputs\evaluation\unconditional_epoch_050
```

This step was necessary because formal metrics such as FID, KID, and LPIPS are not meaningful from a tiny sample grid. A larger sample set gives a more stable comparison with real validation images.

### 4.9 Quantitative Metric Evaluation

The generated set was compared with 1,000 real validation images using FID, KID, and paired LPIPS.

This was necessary to measure general image-distribution similarity. These metrics help answer whether generated images resemble real concrete images in a broad statistical sense.

However, these metrics do not directly answer whether cracks are present, realistic, or structurally meaningful.

### 4.10 Manual Audit Preparation

A manual audit grid and CSV template were created:

```text
outputs\audit\unconditional_epoch_050_audit_grid_first_64.png
outputs\audit\unconditional_epoch_050_manual_audit_first_64.csv
```

This step was necessary because no automatic metric in the current Experiment 1 pipeline reliably measures visible crack presence. Human visual review is currently the most trustworthy way to assess whether generated samples contain plausible cracks.

## 5. Dataset Description

The experiment used SDNET2018, a concrete crack image dataset containing images from three surface categories:

```text
bridge_deck
pavement
wall
```

The processed dataset paths were:

```text
data\processed\train\
data\processed\validation\
data\processed\train.csv
data\processed\validation.csv
```

The total indexed dataset contained 56,092 images:

```text
Training images:   50,482
Validation images:  5,610
```

The dataset contains both cracked and non-cracked concrete patches. For Experiment 1, these labels were not used as conditioning inputs.

## 6. Model Configuration

The model was a compact U-Net used inside a DDPM framework.

Configuration:

```text
Architecture: compact U-Net DDPM
Parameters: 3,258,115
Image size: 64 x 64 RGB
Diffusion steps: 1,000
Loss: mean squared error between true noise and predicted noise
Optimizer: AdamW
Training epochs: 50
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
```

The model learned to predict noise at different diffusion timesteps. During generation, the trained model was used to gradually denoise random Gaussian noise into concrete-like image patches.

## 7. Training Procedure

The training process followed the standard DDPM objective:

1. Load a real training image.
2. Randomly choose a diffusion timestep.
3. Add Gaussian noise to the image according to that timestep.
4. Ask the U-Net to predict the noise.
5. Compare predicted noise with actual noise using mean squared error.
6. Backpropagate the loss.
7. Update the model parameters using AdamW.
8. Repeat for 50 epochs.

The final logged loss was:

```text
0.01599615436508108
```

This shows that the model successfully optimized the denoising objective. The low loss supports the conclusion that the network learned meaningful denoising behavior, but it does not by itself prove that crack structures are realistic.

## 8. Generated Outputs

The model produced sample grids and a full evaluation set.

Sample grids:

```text
outputs\samples\samples_epoch_050_seed_42.png
outputs\samples\samples_epoch_050_seed_123.png
outputs\samples\samples_epoch_050_seed_999.png
```

Evaluation images:

```text
outputs\evaluation\unconditional_epoch_050
```

The evaluation folder contains 1,000 generated PNG images.

These outputs represent the learned unconditional distribution of concrete-like surface patches.

## 9. Evaluation Metrics

### 9.1 FID

FID means Fréchet Inception Distance. It compares the distribution of generated images with the distribution of real images using deep image features.

Experiment 1 result:

```text
FID: 5.604571
```

Lower FID is better. This result indicates that the generated images are close to the real validation images in general feature distribution.

However, FID does not measure crack correctness. A model can achieve a good FID by generating realistic concrete texture even if it rarely generates clear cracks.

### 9.2 KID

KID means Kernel Inception Distance. It is another distribution-comparison metric based on deep features.

Experiment 1 result:

```text
KID mean: 3.419779
KID std: 1.028021
```

The mean gives the average distance estimate. The standard deviation shows how much the estimate varies across repeated subsets.

Lower KID is better. As with FID, KID measures general similarity, not crack-specific correctness.

### 9.3 Paired LPIPS

LPIPS means Learned Perceptual Image Patch Similarity. It estimates perceptual difference using neural-network features.

Experiment 1 result:

```text
Paired LPIPS mean: 0.548679
```

Lower LPIPS means paired images are more perceptually similar. In this unconditional experiment, the pairing does not mean that each generated image is supposed to reconstruct a specific real image. Therefore, LPIPS should be interpreted as a perceptual-distance diagnostic rather than a reconstruction score.

### 9.4 Lightweight Diagnostic Proxies

Additional lightweight diagnostics were calculated on 2026-09-24 using 1,000 generated images and the first 1,000 validation images:

```text
Generated images: 1000
Real comparison images: 1000
Generated mean pixel value: 0.5682
Real mean pixel value: 0.6330
Generated pixel std: 0.1513
Real pixel std: 0.1117
Mean pairwise generated distance: 20.5188
Mean nearest-real pixel distance: 11.1758
```

These are not formal generative-model metrics. They are sanity checks.

The generated mean pixel value was lower than the real mean pixel value. This suggests that the generated images were darker on average than the real validation sample.

The generated pixel standard deviation was higher than the real pixel standard deviation. This suggests that the generated images had more contrast or variation than the real comparison set.

The mean pairwise generated distance gives a rough diversity signal among generated samples. The mean nearest-real pixel distance gives a crude indication of how close generated images are to real images in raw pixel space.

These diagnostics should not replace FID, KID, LPIPS, or human evaluation.

## 10. Observations

The generated images show that the model learned several important visual properties of concrete surfaces:

1. Concrete-like color distribution.
2. Aggregate texture.
3. Surface roughness.
4. Patch-level variation.
5. Some darker streaks or local structures that may appear crack-like.

The strongest observation is that the model successfully learned the general appearance of concrete patches.

The main weakness is crack control. Because the model is unconditional, it cannot be asked to generate cracked images specifically. Crack visibility is therefore inconsistent.

Another important observation is the limitation caused by 64 x 64 resolution. Thin cracks may be lost, blurred, or merged into general surface texture. This limits the usefulness of Experiment 1 for crack-specific generation.

## 11. Results Summary

Training result:

```text
Final logged training loss: 0.01599615436508108
```

Formal evaluation:

```text
FID: 5.604571
KID mean: 3.419779
KID std: 1.028021
Paired LPIPS mean: 0.548679
```

Lightweight diagnostics:

```text
Generated mean pixel value: 0.5682
Real mean pixel value: 0.6330
Generated pixel std: 0.1513
Real pixel std: 0.1117
Mean pairwise generated distance: 20.5188
Mean nearest-real pixel distance: 11.1758
```

Qualitative result:

```text
The model learned realistic concrete-like texture, but crack visibility was inconsistent.
```

## 12. Discussion

Experiment 1 achieved its primary purpose as a baseline. It demonstrated that a compact DDPM can learn the general distribution of SDNET2018 concrete image patches at 64 x 64 resolution.

The quantitative metrics indicate that the generated images are broadly similar to real validation images. The FID value is reasonably low, suggesting that the generated set is close to the real set in deep feature space. KID and LPIPS provide additional evidence of distributional and perceptual similarity.

However, the experiment also shows why unconditional generation is insufficient for controlled crack synthesis. Since the model receives no crack-status input, it has no way to separate cracked and non-cracked modes of the dataset. It may generate crack-like images occasionally, but it cannot be directed to do so.

This matters for practical use. In concrete inspection or crack-detection data augmentation, the important question is not only whether an image looks like concrete. The important question is whether the generated image contains a visible, plausible, useful crack. Experiment 1 does not yet answer that question.

The experiment also highlights the weakness of relying only on FID, KID, and LPIPS. These metrics are useful for general image quality but not enough for engineering validity. A realistic concrete texture without a crack could score well even if the goal is crack generation.

Therefore, the correct interpretation is:

```text
Experiment 1 is a successful unconditional concrete-texture baseline, not a controlled crack-generation model.
```

## 13. Limitations

The main limitations are:

1. No crack-status conditioning was used.
2. The model cannot be instructed to generate cracked or non-cracked concrete.
3. Crack-presence rate has not yet been measured reliably.
4. FID, KID, and LPIPS do not evaluate crack correctness.
5. 64 x 64 resolution may remove or blur thin crack details.
6. The model was trained on patch-level images, not full structural scenes.
7. Generated images may look realistic statistically while still being weak for engineering interpretation.
8. The manual audit CSV is currently only a blank template until reviewed by a human.

## 14. Necessity of Visual Evaluation

Visual evaluation is necessary because crack generation is a semantic and structural problem, not only a texture-similarity problem.

For this project, human review should measure:

```text
visible_crack: yes / no / uncertain
texture_quality: good / acceptable / poor
artifact_level: none / minor / severe
engineering_plausibility: plausible / questionable / implausible
```

The current audit files are:

```text
outputs\audit\unconditional_epoch_050_audit_grid_first_64.png
outputs\audit\unconditional_epoch_050_manual_audit_first_64.csv
```

The first visual audit should answer:

1. How many generated images contain visible crack-like structures?
2. How many generated images have acceptable concrete texture?
3. How many generated images contain obvious artifacts?
4. Are crack-like patterns clear cracks or just surface noise?

This visual audit would convert the current qualitative observation into a measurable result.

## 15. Conclusion

Experiment 1 successfully trained an unconditional DDPM baseline for concrete surface image generation using SDNET2018 at 64 x 64 resolution.

The model learned the general visual distribution of concrete patches, including surface color, roughness, and aggregate-like texture. The final training loss was low, and the generated images achieved a formal FID score of 5.604571 against real validation images.

The experiment confirms that diffusion modeling is suitable for concrete texture generation. However, it also confirms that unconditional generation is not enough for controlled crack synthesis. Since the model was not conditioned on crack status, crack visibility is inconsistent and cannot be controlled.

The most important conclusion is:

```text
Experiment 1 should be treated as a concrete-image baseline, not as the final crack-generation solution.
```

The next necessary step for Experiment 1 is a manual visual audit of generated samples. For controlled crack generation, conditional or mask-conditioned diffusion models are more appropriate.

## 16. Recommended Next Steps

Recommended Experiment 1 continuation:

1. Manually annotate the first 64 generated images using the existing audit CSV.
2. Expand the manual audit to 100 or 200 generated images if a more stable crack-rate estimate is needed.
3. Summarize visible crack rate, usable texture rate, and artifact rate.
4. Add the completed visual-audit result to `STATUS.md`.
5. Use the result as the unconditional baseline when comparing against conditional experiments.

Recommended project continuation:

1. Compare Experiment 1 against Experiment 2 for crack-control improvement.
2. Complete large-scale evaluation for Experiment 3.
3. Develop a stronger crack-presence evaluator before relying on automatic crack-rate metrics.
4. Consider mask-conditioned diffusion if the goal is location-controlled crack synthesis.

## 17. Important Files

Status file:

```text
D:\Downloads\concrete_ddpm\experiments\unconditional_ddpm\STATUS.md
```

Final checkpoint:

```text
D:\Downloads\concrete_ddpm\experiments\unconditional_ddpm\checkpoints\ddpm_epoch_050.pt
```

Training log:

```text
D:\Downloads\concrete_ddpm\experiments\unconditional_ddpm\logs\training_loss.csv
```

Sample grids:

```text
D:\Downloads\concrete_ddpm\experiments\unconditional_ddpm\outputs\samples\samples_epoch_050_seed_42.png
D:\Downloads\concrete_ddpm\experiments\unconditional_ddpm\outputs\samples\samples_epoch_050_seed_123.png
D:\Downloads\concrete_ddpm\experiments\unconditional_ddpm\outputs\samples\samples_epoch_050_seed_999.png
```

Evaluation images:

```text
D:\Downloads\concrete_ddpm\experiments\unconditional_ddpm\outputs\evaluation\unconditional_epoch_050
```

Lightweight diagnostics:

```text
D:\Downloads\concrete_ddpm\experiments\unconditional_ddpm\outputs\evaluation\unconditional_epoch_050_lightweight_diagnostics.txt
```

Manual audit files:

```text
D:\Downloads\concrete_ddpm\experiments\unconditional_ddpm\outputs\audit\unconditional_epoch_050_audit_grid_first_64.png
D:\Downloads\concrete_ddpm\experiments\unconditional_ddpm\outputs\audit\unconditional_epoch_050_manual_audit_first_64.csv
```

Shared source files:

```text
D:\Downloads\concrete_ddpm\src\model.py
D:\Downloads\concrete_ddpm\src\diffusion.py
D:\Downloads\concrete_ddpm\src\train.py
D:\Downloads\concrete_ddpm\src\sample.py
D:\Downloads\concrete_ddpm\src\generate_eval_set.py
D:\Downloads\concrete_ddpm\src\evaluate_metrics.py
```
