import torch
import numpy as np
import cv2
from skimage.util import random_noise
from PIL import Image

class ImageProcessor:
    def __init__(self, config):
        self.config = config
    
    def add_noise(self, img_tensor):
        """
        Adds noise to a normalized tensor image.
        Input: Tensor [C, H, W] in range [-1, 1] (or [0, 1] depending on norm)
        Output: Noisy Tensor
        """
        # Convert tensor to numpy for skimage/noise processing
        # Assuming input is tensor, we move to CPU numpy
        img_np = img_tensor.cpu().numpy()
        
        if self.config.NOISE_TYPE == "gaussian":
            # random_noise expects [0, 1] usually, but works on floats. 
            # If our tensor is [-1, 1], we might want to shift it or just add noise directly.
            # Simple additive gaussian noise:
            noise = np.random.normal(self.config.NOISE_MEAN, self.config.NOISE_STD, img_np.shape)
            noisy_img = img_np + noise
            
        elif self.config.NOISE_TYPE == "salt_pepper":
            # Salt and pepper noise: randomly set pixels to min or max value
            noisy_img = img_np.copy()
            
            # Calculate number of pixels to corrupt
            num_salt = int(np.ceil(self.config.SP_AMOUNT * img_np.size * 0.5))
            num_pepper = int(np.ceil(self.config.SP_AMOUNT * img_np.size * 0.5))
            
            # Generate random flat indices and convert to multi-dimensional coordinates
            # This avoids issues with dimensions of size 1
            if num_salt > 0:
                flat_indices_salt = np.random.choice(img_np.size, size=num_salt, replace=False)
                coords_salt = np.unravel_index(flat_indices_salt, img_np.shape)
                noisy_img[coords_salt] = 1.0  # Max value (salt)
            
            if num_pepper > 0:
                flat_indices_pepper = np.random.choice(img_np.size, size=num_pepper, replace=False)
                coords_pepper = np.unravel_index(flat_indices_pepper, img_np.shape)
                min_val = -1.0 if img_np.min() < 0 else 0.0
                noisy_img[coords_pepper] = min_val  # Min value (pepper)
            
        elif self.config.NOISE_TYPE == "poisson":
            # Poisson depends on image intensity. 
            # Usually applied to raw counts. Simulating on normalized data is tricky.
            # We'll use skimage's implementation which handles [0, 1] or non-negative
            # Shift to [0, 1]
            img_01 = (img_np + 1) / 2
            noisy_01 = random_noise(img_01, mode='poisson')
            noisy_img = (noisy_01 * 2) - 1
            
        elif self.config.NOISE_TYPE == "speckle":
            # Noise = I + I * n
            noise = np.random.normal(0, self.config.NOISE_STD, img_np.shape)
            noisy_img = img_np + img_np * noise
            
        else:
            return img_tensor # No noise
            
        # Clip to valid range
        noisy_img = np.clip(noisy_img, -1.0, 1.0)
        
        return torch.from_numpy(noisy_img).float()

    def classical_denoise(self, noisy_tensor, method="gaussian"):
        """
        Applies classical denoising techniques.
        Input: Tensor [C, H, W]
        Output: Tensor [C, H, W]
        """
        img_np = noisy_tensor.squeeze().cpu().numpy() # (28, 28)
        
        # Convert to 0-255 uint8 for OpenCV tools
        img_uint8 = ((img_np + 1) / 2 * 255).astype(np.uint8)
        
        if method == "gaussian":
            # Gaussian Blur
            denoised = cv2.GaussianBlur(img_uint8, (3, 3), 0)
            
        elif method == "median":
            # Median Filter (good for salt and pepper)
            denoised = cv2.medianBlur(img_uint8, 3)
            
        elif method == "nlm":
            # Non-local Means
            denoised = cv2.fastNlMeansDenoising(img_uint8, None, h=10, templateWindowSize=7, searchWindowSize=21)
            
        else:
            return noisy_tensor
            
        # Convert back to Tensor [-1, 1]
        denoised_float = denoised.astype(np.float32) / 255.0 * 2 - 1
        return torch.from_numpy(denoised_float).unsqueeze(0)
