import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import time
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class Trainer:
    def __init__(self, config, model, train_loader, val_loader, test_loader):
        self.config = config
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Loss: MSE for reconstruction
        self.criterion = nn.MSELoss()
        
        if config.OPTIMIZER == "Adam":
            self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        else:
            self.optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.9)
            
        if config.USE_SCHEDULER:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=config.SCHEDULER_STEP_SIZE, gamma=config.SCHEDULER_GAMMA)
        else:
            self.scheduler = None
            
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': []}
        
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    def calculate_metrics(self, outputs, targets):
        """
        Calculate PSNR and SSIM for a batch.
        Inputs are Tensors [-1, 1].
        """
        outputs_np = outputs.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Denormalize to [0, 1] for metrics
        outputs_np = (outputs_np + 1) / 2.0
        targets_np = (targets_np + 1) / 2.0
        
        outputs_np = np.clip(outputs_np, 0, 1)
        targets_np = np.clip(targets_np, 0, 1)
        
        batch_psnr = 0.0
        batch_ssim = 0.0
        batch_size = outputs_np.shape[0]
        
        for i in range(batch_size):
            # Squeeze channel dim: (1, 28, 28) -> (28, 28)
            img_pred = outputs_np[i].squeeze()
            img_target = targets_np[i].squeeze()
            
            p = psnr(img_target, img_pred, data_range=1.0)
            s = ssim(img_target, img_pred, data_range=1.0)
            
            batch_psnr += p
            batch_ssim += s
            
        return batch_psnr / batch_size, batch_ssim / batch_size

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch}/{self.config.EPOCHS}")
        
        for i, (noisy, clean) in progress_bar:
            noisy = noisy.to(self.config.DEVICE)
            clean = clean.to(self.config.DEVICE)
            
            # Forward
            outputs = self.model(noisy)
            loss = self.criterion(outputs, clean)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': running_loss / (i + 1)})
            
        return running_loss / len(self.train_loader)

    def validate(self, loader):
        self.model.eval()
        running_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        
        with torch.no_grad():
            for noisy, clean in loader:
                noisy = noisy.to(self.config.DEVICE)
                clean = clean.to(self.config.DEVICE)
                
                outputs = self.model(noisy)
                loss = self.criterion(outputs, clean)
                
                running_loss += loss.item()
                
                # Calculate metrics
                p, s = self.calculate_metrics(outputs, clean)
                total_psnr += p
                total_ssim += s
                
        return (running_loss / len(loader), 
                total_psnr / len(loader), 
                total_ssim / len(loader))

    def train(self):
        print(f"Starting training on {self.config.DEVICE}...")
        start_time = time.time()
        
        for epoch in range(1, self.config.EPOCHS + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_psnr, val_ssim = self.validate(self.val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_psnr'].append(val_psnr)
            self.history['val_ssim'].append(val_ssim)
            
            if self.scheduler:
                self.scheduler.step()
                
            print(f"Epoch [{epoch}/{self.config.EPOCHS}] "
                  f"Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} PSNR: {val_psnr:.2f} SSIM: {val_ssim:.4f}")
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                if self.config.SAVE_MODEL:
                    torch.save(self.model.state_dict(), os.path.join(self.config.OUTPUT_DIR, 'best_model.pth'))
        
        total_time = time.time() - start_time
        print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
        
        # Test Evaluation
        print("Evaluating on Test Set...")
        if self.config.SAVE_MODEL:
             self.model.load_state_dict(torch.load(os.path.join(self.config.OUTPUT_DIR, 'best_model.pth')))
             
        test_loss, test_psnr, test_ssim = self.validate(self.test_loader)
        print(f"Test Loss: {test_loss:.6f} PSNR: {test_psnr:.2f} SSIM: {test_ssim:.4f}")
        
        return self.history
