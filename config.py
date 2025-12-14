import torch
import os

class Config:
    def __init__(self):
        # ---------------------------------------------------------------------
        # System / General
        # ---------------------------------------------------------------------
        self.PROJECT_NAME = "TissueMNIST_Denoising_System"
        self.SEED = 42
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.NUM_WORKERS = 2
        self.OUTPUT_DIR = "./outputs/UNet_salt_pepper"
        
        # ---------------------------------------------------------------------
        # Data Configuration
        # ---------------------------------------------------------------------
        self.DATA_FLAG = "tissuemnist"
        self.DOWNLOAD_DATA = True
        self.IMAGE_SIZE = 28
        self.BATCH_SIZE = 128
        
        # ---------------------------------------------------------------------
        # Noise Simulation Configuration
        # ---------------------------------------------------------------------
        # Type of noise to add: "gaussian", "salt_pepper", "poisson", "speckle"
        self.NOISE_TYPE = "salt_pepper" 
        
        # Parameters for Gaussian Noise
        self.NOISE_MEAN = 0.0
        self.NOISE_STD = 0.2  # Sigma (intensity of noise)
        
        # Parameters for Salt & Pepper
        self.SP_AMOUNT = 0.05
        
        # ---------------------------------------------------------------------
        # Model Architecture (Denoising)
        # ---------------------------------------------------------------------
        # 'Autoencoder' or 'UNet' (simplified for 28x28)
        self.MODEL_ARCH = "UNet"
        self.INPUT_CHANNELS = 1
        self.HIDDEN_DIM = 64
        
        # ---------------------------------------------------------------------
        # Training Hyperparameters
        # ---------------------------------------------------------------------
        self.EPOCHS = 20
        self.LEARNING_RATE = 1e-3
        self.OPTIMIZER = "Adam"
        self.WEIGHT_DECAY = 1e-5
        
        # Scheduler
        self.USE_SCHEDULER = True
        self.SCHEDULER_STEP_SIZE = 10
        self.SCHEDULER_GAMMA = 0.1
        
        # ---------------------------------------------------------------------
        # Logging / Checkpointing
        # ---------------------------------------------------------------------
        self.LOG_INTERVAL = 10
        self.SAVE_MODEL = True
        self.SAVE_BEST_ONLY = True
        
    def save_config(self, path):
        """Saves the configuration to a text file for reproducibility."""
        with open(path, 'w') as f:
            f.write(str(self))
            
    def __str__(self):
        return "\n".join([f"{k}: {v}" for k, v in self.__dict__.items()])

config = Config()
