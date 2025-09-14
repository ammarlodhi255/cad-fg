import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# ========== Image-only / Late-fusion backbone ==========
class ResNet18Backbone(nn.Module):
    """
    Returns pooled features (512-d) and logits via a small head.
    Use out_dim=2 for binary classification.
    """
    def __init__(self, pretrained=True, out_dim=2):
        super().__init__()
        w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = resnet18(weights=w)
        in_feats = m.fc.in_features  # 512
        m.fc = nn.Identity()
        self.backbone = m
        self.head = nn.Linear(in_feats, out_dim)
        self.feat_dim = in_feats  # 512

    def forward_feats(self, x):
        # forward through convs → pool
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x  # 512

    def forward(self, x):
        feats = self.forward_feats(x)
        logits = self.head(feats)
        return feats, logits


# ========== Early fusion (metadata channels at input) ==========
class ResNet18EarlyFusionPatient(nn.Module):
    """
    Metadata vector (B, tab_dim) -> MLP -> c_pat channels,
    broadcast to HxW and concatenated with RGB before conv1.
    """
    def __init__(self, tab_dim, c_pat=8, pretrained=True, out_dim=2, pmap_hidden=64, pmap_drop=0.1):
        super().__init__()
        self.c_pat = c_pat
        self.tab_dim = tab_dim

        # metadata → channels
        self.pmap = nn.Sequential(
            nn.Linear(tab_dim, pmap_hidden), nn.ReLU(), nn.Dropout(pmap_drop),
            nn.Linear(pmap_hidden, c_pat)
        )

        w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = resnet18(weights=w)

        # patch first conv to accept 3 + c_pat channels
        old = m.conv1  # Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv = nn.Conv2d(
            in_channels=old.in_channels + c_pat,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight[:, :3, ...] = old.weight  # keep RGB pretrained weights
        nn.init.kaiming_normal_(new_conv.weight[:, 3:, ...], nonlinearity="relu")
        m.conv1 = new_conv

        in_feats = m.fc.in_features  # 512
        m.fc = nn.Identity()
        self.backbone = m
        self.head = nn.Linear(in_feats, out_dim)
        self.feat_dim = in_feats

    def forward(self, x_img, clin_vec):
        B, C, H, W = x_img.shape
        assert C == 3, f"expected 3-channel image, got {C}"
        assert clin_vec.dim() == 2 and clin_vec.size(0) == B and clin_vec.size(1) == self.tab_dim

        pm = self.pmap(clin_vec)                              # (B, c_pat)
        ch = pm.view(B, self.c_pat, 1, 1).expand(B, self.c_pat, H, W)  # (B,c_pat,H,W)
        x = torch.cat([x_img, ch], dim=1)                     # (B, 3+c_pat, H, W)

        # regular resnet18 forward to pooled features
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)
        logits = self.head(feats)
        return feats, logits