import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, config):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.enc1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1) # 28 -> 14
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # 14 -> 7
        self.enc3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 7 -> 4 (padding makes it tricky, let's control size)
        
        # With standard padding=1, stride=2:
        # 28 -> 14
        # 14 -> 7
        # 7 -> 4
        
        # Decoder
        self.dec1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0) # 4 -> 7
        self.dec2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1) # 7 -> 14
        self.dec3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1) # 14 -> 28
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() # If output is [0,1], but we use [-1, 1], so Tanh
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        x = self.relu(self.enc1(x))
        x = self.relu(self.enc2(x))
        x = self.relu(self.enc3(x))
        
        # Decoder
        x = self.relu(self.dec1(x))
        x = self.relu(self.dec2(x))
        x = self.tanh(self.dec3(x)) # Output range [-1, 1]
        
        return x

class UNetSmall(nn.Module):
    """
    A simplified U-Net for 28x28 input.
    """
    def __init__(self, config):
        super(UNetSmall, self).__init__()
        
        # Contracting Path
        self.conv1 = self.double_conv(1, 32)
        self.pool1 = nn.MaxPool2d(2) # 14
        
        self.conv2 = self.double_conv(32, 64)
        self.pool2 = nn.MaxPool2d(2) # 7
        
        # Bridge
        self.conv3 = self.double_conv(64, 128)
        
        # Expansive Path
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2) 
        # Output of up4 will be 14x14 if input is 7x7. But 7x7 came from 14x14 pooling?
        # 7x7 is odd. MaxPool on 14 gives 7.
        # ConvTranspose on 7 with stride 2 gives 14. Matches.
        
        self.conv4 = self.double_conv(128, 64) # 64 from up + 64 from skip
        
        self.up5 = nn.ConvTranspose2d(64, 32, 2, stride=2) # 28
        self.conv5 = self.double_conv(64, 32)
        
        self.final = nn.Conv2d(32, 1, 1)
        self.tanh = nn.Tanh()

    def double_conv(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Down
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        
        # Bridge
        c3 = self.conv3(p2)
        
        # Up
        u4 = self.up4(c3)
        # Handle simple padding if shapes don't match perfectly (for odd sizes)
        # But here 28->14->7->14->28 is clean.
        x4 = torch.cat([u4, c2], dim=1)
        c4 = self.conv4(x4)
        
        u5 = self.up5(c4)
        x5 = torch.cat([u5, c1], dim=1)
        c5 = self.conv5(x5)
        
        out = self.final(c5)
        return self.tanh(out)

def get_model(config):
    if config.MODEL_ARCH == "Autoencoder":
        return Autoencoder(config)
    elif config.MODEL_ARCH == "UNet":
        return UNetSmall(config)
    else:
        raise ValueError(f"Unknown architecture: {config.MODEL_ARCH}")
