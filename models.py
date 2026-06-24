import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Standard YOLO-style Convolutional Block (Conv + BatchNorm + SiLU)"""
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class CSPBlock(nn.Module):
    """Simplified Cross Stage Partial Block for efficient gradient flow"""
    def __init__(self, in_c, out_c, num_blocks=1):
        super().__init__()
        mid_c = out_c // 2
        self.conv1 = ConvBlock(in_c, mid_c, 1, 1, 0)
        self.conv2 = ConvBlock(in_c, mid_c, 1, 1, 0)
        
        self.blocks = nn.Sequential(*[
            ConvBlock(mid_c, mid_c, 3, 1, 1) for _ in range(num_blocks)
        ])
        
        self.conv3 = ConvBlock(mid_c * 2, out_c, 1, 1, 0)

    def forward(self, x):
        y1 = self.blocks(self.conv1(x))
        y2 = self.conv2(x)
        return self.conv3(torch.cat([y1, y2], dim=1))

class TriModalYOLOSeg(nn.Module):
    """5-Channel YOLO-like Semantic Segmentation Network"""
    def __init__(self, in_channels=5, num_classes=32):
        super().__init__()
        
        # --- Encoder (Backbone) ---
        # Surgically adapted stem to accept 5 channels (RGB + Depth + Thermal)
        self.stem = ConvBlock(in_channels, 64, stride=2) 
        
        self.layer1 = nn.Sequential(ConvBlock(64, 128, stride=2), CSPBlock(128, 128, 1))
        self.layer2 = nn.Sequential(ConvBlock(128, 256, stride=2), CSPBlock(256, 256, 2))
        self.layer3 = nn.Sequential(ConvBlock(256, 512, stride=2), CSPBlock(512, 512, 2))
        
        # --- Decoder (Neck / Head) ---
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = CSPBlock(512, 256, 1) # Concat layer2 (256) + up1 (256) = 512
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = CSPBlock(256, 128, 1) # Concat layer1 (128) + up2 (128) = 256
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = CSPBlock(128, 64, 1)  # Concat stem (64) + up3 (64) = 128
        
        # Final upsample to match original (480, 640) resolution
        self.final_up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x_stem = self.stem(x)      # [B, 64, H/2, W/2]
        x1 = self.layer1(x_stem)   # [B, 128, H/4, W/4]
        x2 = self.layer2(x1)       # [B, 256, H/8, W/8]
        x3 = self.layer3(x2)       # [B, 512, H/16, W/16]
        
        # Decoder with Skip Connections
        d1 = self.up1(x3)
        d1 = self.dec1(torch.cat([d1, x2], dim=1))
        
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, x1], dim=1))
        
        d3 = self.up3(d2)
        d3 = self.dec3(torch.cat([d3, x_stem], dim=1))
        
        out = self.final_up(d3)
        out = self.head(out) # [B, num_classes, H, W]
        
        return out