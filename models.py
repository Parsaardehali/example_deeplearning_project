import torch
import torch.nn as nn
import torch.nn.functional as F


# Define a U-Net model
class MyUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MyUNet, self).__init__()
        
        # Encoder with batch normalization
        self.enc1 = self.conv_block(in_channels, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)
        self.enc4 = self.conv_block(64, 128)
        self.enc5 = self.conv_block(128, 256)
        
        # Decoder with skip connections
        self.dec1 = self.conv_block(256 + 128, 128)  # Added skip connections
        self.dec2 = self.conv_block(128 + 64, 64)
        self.dec3 = self.conv_block(64 + 32, 32)
        self.dec4 = self.conv_block(32 + 16, 16)
        self.final = nn.Conv2d(16, out_channels, 1)  # 1x1 conv for final output
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),  # Added batch normalization
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)  # Added dropout
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.maxpool(e1))
        e3 = self.enc3(self.maxpool(e2))
        e4 = self.enc4(self.maxpool(e3))
        e5 = self.enc5(self.maxpool(e4))
        
        # Decoder with skip connections
        d1 = self.dec1(torch.cat([self.upsample(e5), e4], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.upsample(d3), e1], dim=1))
        
        # Final 1x1 convolution
        output = self.final(d4)
        return torch.sigmoid(output)  # Added sigmoid activation

