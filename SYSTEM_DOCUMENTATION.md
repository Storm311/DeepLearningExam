# System Documentation: TissueMNIST Denoising & Artifact Removal

## 1. Project Overview

This system is a Deep Learning-based image reconstruction pipeline designed to remove noise and artifacts from medical images. It repurposes the **TissueMNIST** dataset (originally for classification) into a **Denoising** task by simulating realistic medical imaging acquisition noise.

The goal is to map a noisy input image ($x_{noisy}$) to a clean target image ($x_{clean}$) using a Convolutional Neural Network (CNN), specifically comparing Autoencoder and U-Net architectures.

## 2. System Architecture

The project is structured modularly to separate configuration, data handling, modeling, and training.

### 2.1 Configuration (`config.py`)
**What:** A centralized class containing all hyperparameters (learning rate, batch size, noise type, model selection).
**Why:** Deep Learning experiments require frequent tuning. Hardcoding values in the training script leads to errors. A central config ensures reproducibility and easy experimentation.
**Key Variables:**
-   `NOISE_TYPE`: Selects between Gaussian, Salt & Pepper, Speckle, etc.
-   `NOISE_STD`: Controls the difficulty of the task.
-   `MODEL_ARCH`: Switches the backend between "Autoencoder" and "UNet".

### 2.2 Data Pipeline (`data_loader.py` & `image_processing.py`)
**Strategy:** Self-Supervised / Synthetic Generation.
Since we don't have "naturally noisy" versions of TissueMNIST, we generate them on the fly.

1.  **Loading**: We load clean 28x28 grayscale images from TissueMNIST.
2.  **Noise Injection (`image_processing.py`)**:
    -   **Gaussian Noise**: Simulates electronic noise in sensors. Added to the image tensor.
    -   **Salt & Pepper**: Simulates "dead pixels" or transmission errors. Randomly sets pixels to min/max values.
    -   **Speckle**: Simulates noise common in ultrasound (multiplicative noise).
3.  **Pairing (`data_loader.py`)**: The `Dataset` class returns a tuple `(noisy_image, clean_image)`. The model sees the noisy version, but the loss is calculated against the clean version.

### 2.3 Neural Network Models (`model.py`)

We implemented two architectures to compare performance:

#### A. Convolutional Autoencoder
**Structure:** Encoder $\rightarrow$ Bottleneck $\rightarrow$ Decoder.
-   **Encoder**: Series of Conv2d layers that downsample the image (28x28 $\rightarrow$ 14x14 $\rightarrow$ 7x7). This forces the model to learn the most important "features" of the tissue structure while discarding noise (which is high frequency and random).
-   **Bottleneck**: The compressed representation.
-   **Decoder**: Transpose Convolutions (Deconv) that upsample back to 28x28.
**Pros:** Good at removing heavy noise.
**Cons:** Can result in blurry output because fine details are lost in the bottleneck.

#### B. U-Net (Mini)
**Structure:** Similar to an Autoencoder but with **Skip Connections**.
-   **Skip Connections**: Links layers in the encoder directly to corresponding layers in the decoder (concatenation).
-   **Why:** Medical images rely on fine texture details. A standard autoencoder loses this in the bottleneck. Skip connections allow the network to "pass through" spatial information from the input to the output, making reconstruction much sharper.

### 2.4 Training Engine (`train.py`)

**Loss Function: Mean Squared Error (MSE)**
$$L = \frac{1}{N} \sum (y_{pred} - y_{true})^2$$
We use MSE because it directly penalizes pixel-level differences, encouraging the output to statistically match the clean image.

**Metrics:**
1.  **PSNR (Peak Signal-to-Noise Ratio)**: Measured in decibels (dB). Higher is better. It compares the maximum possible power of a signal to the power of corrupting noise.
2.  **SSIM (Structural Similarity Index)**: A value between 0 and 1. It measures perceptual similarity (luminance, contrast, structure) rather than just pixel difference. It correlates better with human visual perception than MSE.

### 2.5 Baselines (`image_processing.py`)
To prove the Deep Learning model is useful, we compare it against classical signal processing filters:
-   **Gaussian Blur**: Smooths out noise but also blurs edges.
-   **Median Filter**: Excellent for Salt & Pepper noise but can look "patchy".
-   **Non-local Means**: A sophisticated algorithm that averages similar patches within the image.

## 3. Workflow Summary

1.  **Initialization**: `main.py` reads `config.py`.
2.  **Data Loading**: Batches of (Noisy, Clean) images are generated.
3.  **Forward Pass**: Model predicts Denoised image from Noisy input.
4.  **Loss Calculation**: MSE between Denoised and Clean.
5.  **Backpropagation**: Gradients update model weights.
6.  **Evaluation**: Every epoch, PSNR/SSIM are calculated on the Validation set.
7.  **Result Generation**: After training, the system saves:
    -   `best_model.pth`: The weights of the epoch with lowest validation loss.
    -   `comparison_results.png`: A visual grid to qualitatively assess performance.

