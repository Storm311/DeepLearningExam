# TissueMNIST Denoising & Artifact Removal

This project implements a Deep Learning system for **Denoising and Artifact Removal** on the TissueMNIST medical dataset. Instead of classification, the goal is to recover clean, high-quality images from noisy simulations, which is a critical task in medical imaging acquisition.

## Features

-   **Task**: Image-to-Image Reconstruction (Denoising).
-   **Noise Simulation**: Realistic acquisition noise generation:
    -   Gaussian Noise
    -   Salt & Pepper (Impulse) Noise
    -   Speckle Noise
    -   Poisson Noise
-   **Models**: 
    -   **Denoising Autoencoder**: Convolutional Encoder-Decoder architecture.
    -   **U-Net (Mini)**: Architecture with skip connections for preserving high-frequency details.
-   **Baselines**: Comparison against classical methods (Gaussian Blur, Median Filter).
-   **Metrics**: Evaluation using **PSNR** (Peak Signal-to-Noise Ratio) and **SSIM** (Structural Similarity Index).
-   **Reproducibility**: Configuration logging and seed fixing.

## Project Structure

-   `config.py`: Controls noise types (`NOISE_TYPE`), intensity (`NOISE_STD`), model architecture, and training params.
-   `image_processing.py`: Contains the logic for injecting noise and applying classical denoising filters.
-   `data_loader.py`: Wraps TissueMNIST to provide `(noisy, clean)` image pairs.
-   `model.py`: PyTorch implementations of the Autoencoder and U-Net.
-   `train.py`: Training engine with MSE loss and PSNR/SSIM monitoring.
-   `main.py`: Entry point. Runs training, plots history, and saves visual comparisons.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1.  **Configure Experiment**:
    Open `config.py` to change the noise scenario:
    ```python
    self.NOISE_TYPE = "gaussian"  # or "salt_pepper", "speckle"
    self.NOISE_STD = 0.3          # Increase for harder task
    self.MODEL_ARCH = "UNet"      # Try "Autoencoder" vs "UNet"
    ```

2.  **Run Training**:
    ```bash
    python main.py
    ```

3.  **Outputs** (in `./outputs/`):
    -   `run_config.txt`: Exact parameters used for the run.
    -   `best_model.pth`: Saved model weights.
    -   `training_metrics.png`: Plot of Loss, PSNR, and SSIM over epochs.
    -   `comparison_results.png`: Grid showing **Original | Noisy | Classical Denoise | Deep Denoise**.

## Methodology

### Noise Simulation
We simulate the degradation of medical images (which often happens during low-dose CT or MRI acquisition) by injecting noise into the clean TissueMNIST samples.

### Models
-   **Autoencoder**: Compresses the noisy image into a latent space and reconstructs it, forcing the network to learn robust features and ignore noise.
-   **U-Net**: Uses skip connections to pass fine-grained spatial information from the encoder to the decoder, often resulting in sharper reconstructions than standard Autoencoders.

### Metrics
-   **MSE (Mean Squared Error)**: The optimization objective.
-   **PSNR**: Higher is better. Measures peak error.
-   **SSIM**: Higher is better (max 1.0). Measures perceptual similarity.

## Example Results

The system produces a side-by-side comparison of:
1.  **Ground Truth** (Clean Image)
2.  **Input** (Noisy Image)
3.  **Baseline** (Median Filter - good for salt & pepper, often blurry)
4.  **Deep Learning** (Restored Image)
