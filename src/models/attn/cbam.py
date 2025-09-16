import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    CBAM channel attention:
      M_c(X) = σ( MLP(AvgPool(X)) + MLP(MaxPool(X)) ), shape (B,C,1,1)
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        avg = self.mlp(self.avg_pool(x))
        mx  = self.mlp(self.max_pool(x))
        w = torch.sigmoid(avg + mx)
        return x * w


class SpatialAttention(nn.Module):
    """
    CBAM spatial attention:
      M_s(X) = σ( Conv7x7([AvgPool_c(X); MaxPool_c(X)]) ), shape (B,1,H,W)
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)                     # (B,1,H,W)
        mx  = torch.amax(x, dim=1, keepdim=True)                     # (B,1,H,W)
        w = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))    # (B,1,H,W)
        return x * w


class CBAM(nn.Module):
    """
    Full CBAM = ChannelAttention -> SpatialAttention.
    Put it on a known channel count (e.g., 1280 for EfficientNet-B0 final feature map).
    """
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x