import torch
import torch.nn as nn
import torchvision.models as models


class ConvBlock(nn.Module):
    """
    Two 3x3 convolutions each followed by BatchNorm and ReLU.
    Used as the basic building block in the UNet decoder.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNetWithResNet(nn.Module):
    """
    UNet-style encoder-decoder for binary change detection.

    Encoder: Pretrained ResNet34 (ImageNet weights), modified to accept
             4-channel input (3-channel EO + 1-channel SAR).
    Decoder: UNet decoder with skip connections from encoder stages.
    Output:  2-channel logits map (No-Change vs Change) at input resolution.

    4-channel adaptation:
        The pretrained RGB weights are kept for channels 1-3.
        Channel 4 (SAR) is initialised as the mean of the RGB weights,
        preserving pretrained feature representations.
    """

    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()

        # Load pretrained ResNet34
        resnet = models.resnet34(weights='IMAGENET1K_V1')

        # Adapt first conv layer for 4-channel input
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight          # RGB channels
            new_conv.weight[:, 3]  = old_conv.weight.mean(dim=1)  # SAR channel
        resnet.conv1 = new_conv

        # Encoder stages
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64ch
        self.pool  = resnet.maxpool
        self.enc2  = resnet.layer1   # 64ch
        self.enc3  = resnet.layer2   # 128ch
        self.enc4  = resnet.layer3   # 256ch
        self.enc5  = resnet.layer4   # 512ch

        # Decoder stages (each takes upsampled + skip connection features)
        self.dec4 = ConvBlock(512 + 256, 256)
        self.dec3 = ConvBlock(256 + 128, 128)
        self.dec2 = ConvBlock(128 + 64,  64)
        self.dec1 = ConvBlock(64  + 64,  64)

        # Final 1x1 conv to produce class logits
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

        # Bilinear upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder (progressively downsample)
        e1 = self.enc1(x)              # (B, 64,  H/2,  W/2)
        e2 = self.enc2(self.pool(e1))  # (B, 64,  H/4,  W/4)
        e3 = self.enc3(e2)             # (B, 128, H/8,  W/8)
        e4 = self.enc4(e3)             # (B, 256, H/16, W/16)
        e5 = self.enc5(e4)             # (B, 512, H/32, W/32)

        # Decoder with skip connections (progressively upsample)
        d4 = self.dec4(torch.cat([self.up(e5), e4], dim=1))  # (B, 256, H/16, W/16)
        d3 = self.dec3(torch.cat([self.up(d4), e3], dim=1))  # (B, 128, H/8,  W/8)
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))  # (B, 64,  H/4,  W/4)
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))  # (B, 64,  H/2,  W/2)

        # Final upsample to input resolution + class prediction
        out = self.up(self.final(d1))   # (B, num_classes, H, W)
        return out


class DiceLoss(nn.Module):
    """
    Dice Loss for the change class (label=1).
    Robust to class imbalance by focusing on minority class overlap.
    """
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.softmax(predictions, dim=1)
        pred = predictions[:, 1]
        targ = (targets == 1).float()
        intersection = (pred * targ).sum()
        dice = (2 * intersection + self.smooth) / (
            pred.sum() + targ.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined Dice Loss + Weighted Cross-Entropy Loss.

    Weighted CE penalises missing change pixels more heavily.
    Dice Loss directly optimises the change class overlap.
    Together they address the severe class imbalance (~95% no-change).
    """
    def __init__(self, class_weights, device):
        super().__init__()
        weight = torch.tensor(class_weights, dtype=torch.float32).to(device)
        self.ce   = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss()

    def forward(self, predictions, targets):
        return self.ce(predictions, targets) + self.dice(predictions, targets)
