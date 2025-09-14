import torch, torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class PatientToChannels(nn.Module):
    def __init__(self, in_dim, c_pat=4, hidden=64, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, c_pat)
        )
    def forward(self, clin_vec):  # (B, D_tab)
        return self.net(clin_vec)  # (B, c_pat)

class ResNetEarlyFusionPatient(nn.Module):
    def __init__(self, tab_dim, c_pat=4, pretrained=True, out_dim=2):
        super().__init__()
        self.c_pat = c_pat
        self.pmap = PatientToChannels(in_dim=tab_dim, c_pat=c_pat)

        w = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = resnet50(weights=w)
        old = base.conv1
        base.conv1 = nn.Conv2d(3 + c_pat, old.out_channels,
                               kernel_size=old.kernel_size, stride=old.stride,
                               padding=old.padding, bias=False)
        with torch.no_grad():
            # keep original RGB weights
            base.conv1.weight[:, :3, ...] = old.weight
        # fresh init for patient channels
        nn.init.kaiming_normal_(base.conv1.weight[:, 3:3+c_pat, ...], nonlinearity="relu")

        base.fc = nn.Identity()
        self.backbone = base
        self.head = nn.Linear(2048, out_dim)

    def forward(self, x_img, clin_vec):
        B, _, H, W = x_img.shape
        ch = self.pmap(clin_vec).view(B, self.c_pat, 1, 1).expand(B, self.c_pat, H, W)
        feats = self.backbone(torch.cat([x_img, ch], dim=1))
        return feats, self.head(feats)