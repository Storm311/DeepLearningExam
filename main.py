import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from config import config
from image_processing import ImageProcessor
from data_loader import DataLoaderManager
from model import get_model
from train import Trainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def plot_history(history, output_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['val_loss'], label='Val')
    plt.title('Loss (MSE)')
    plt.legend()
    
    # PSNR
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['val_psnr'], label='Val PSNR')
    plt.title('PSNR (dB)')
    plt.legend()
    
    # SSIM
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['val_ssim'], label='Val SSIM')
    plt.title('SSIM')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))

def save_visual_comparison(model, test_loader, image_processor, device, output_dir):
    """
    Saves a grid showing: Clean | Noisy | Classical Denoise | Deep Denoise
    """
    model.eval()
    noisy, clean = next(iter(test_loader))
    
    # Take first 5 images
    n_samples = 5
    noisy = noisy[:n_samples].to(device)
    clean = clean[:n_samples].to(device)
    
    # Deep Learning Prediction
    with torch.no_grad():
        deep_denoised = model(noisy)
        
    # Classical Denoising (CPU)
    classical_denoised_list = []
    for i in range(n_samples):
        # Pass individual tensor
        res = image_processor.classical_denoise(noisy[i].cpu(), method="median") 
        classical_denoised_list.append(res)
    classical_denoised = torch.cat(classical_denoised_list, dim=0).to(device)
    
    # Prepare plot
    fig, axes = plt.subplots(n_samples, 4, figsize=(10, 2 * n_samples))
    
    cols = ['Original', 'Noisy', 'Classical (Median)', 'Deep Learning']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for i in range(n_samples):
        # Helper to show tensor
        def show(ax, t):
            img = t.squeeze().cpu().numpy()
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            
        show(axes[i, 0], clean[i])
        show(axes[i, 1], noisy[i])
        show(axes[i, 2], classical_denoised[i])
        show(axes[i, 3], deep_denoised[i])
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_results.png'))
    print(f"Comparison results saved to {os.path.join(output_dir, 'comparison_results.png')}")

def main():
    print(f"Initializing {config.PROJECT_NAME}...")
    
    # Save config state
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    config.save_config(os.path.join(config.OUTPUT_DIR, 'run_config.txt'))
    print(f"Configuration saved to {os.path.join(config.OUTPUT_DIR, 'run_config.txt')}")
    
    set_seed(config.SEED)
    
    # Setup
    image_processor = ImageProcessor(config)
    data_manager = DataLoaderManager(config, image_processor)
    train_loader, val_loader, test_loader = data_manager.get_dataloaders()
    
    print(f"Noise Type: {config.NOISE_TYPE} (std={config.NOISE_STD})")
    
    # Model
    model = get_model(config)
    print(f"Model: {config.MODEL_ARCH}")
    
    # Train
    trainer = Trainer(config, model, train_loader, val_loader, test_loader)
    history = trainer.train()
    
    # Visuals
    plot_history(history, config.OUTPUT_DIR)
    save_visual_comparison(model, test_loader, image_processor, config.DEVICE, config.OUTPUT_DIR)
    
    print("Process Completed Successfully!")

if __name__ == "__main__":
    main()
