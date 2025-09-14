from torchvision.models import inception_v3
import torch.nn as nn
import torch

class InceptionV3Backbone(nn.Module):
    def __init__(self, pretrained=True, out_dim=2):
        super().__init__()
        base = inception_v3(weights="IMAGENET1K_V1" if pretrained else None)  # aux_logits=True default
        base.aux_logits = False
        base.AuxLogits = nn.Identity()
        base.transform_input = False      # <<< IMPORTANT: don't drop extra channels

        in_feats = base.fc.in_features
        self.feat_dim = in_feats
        base.fc = nn.Identity()
        self.backbone = base
        self.head = nn.Linear(in_feats, out_dim)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return feats, logits


class InceptionV3EarlyFusionPatient(nn.Module):
    def __init__(self, tab_dim, c_pat=8, pretrained=True, out_dim=2, pmap_hidden=64, pmap_drop=0.1):
        super().__init__()
        self.c_pat = c_pat
        self.tab_dim = tab_dim

        # metadata -> channel projection
        self.pmap = nn.Sequential(
            nn.Linear(tab_dim, pmap_hidden), nn.ReLU(), nn.Dropout(pmap_drop),
            nn.Linear(pmap_hidden, c_pat)
        )

        base = inception_v3(weights="IMAGENET1K_V1" if pretrained else None)  # aux_logits=True default
        base.aux_logits = False
        base.AuxLogits = nn.Identity()
        base.transform_input = False      # <<< IMPORTANT: preserve 3 + c_pat channels

        # patch first conv
        old = base.Conv2d_1a_3x3.conv
        new_in = old.in_channels + c_pat
        new_conv = nn.Conv2d(new_in, old.out_channels,
                             kernel_size=old.kernel_size,
                             stride=old.stride,
                             padding=old.padding,
                             bias=False)
        with torch.no_grad():
            new_conv.weight[:, :old.in_channels, ...] = old.weight
        nn.init.kaiming_normal_(new_conv.weight[:, old.in_channels:, ...], nonlinearity="relu")
        base.Conv2d_1a_3x3.conv = new_conv

        in_feats = base.fc.in_features
        self.feat_dim = in_feats
        base.fc = nn.Identity()

        self.backbone = base
        self.head = nn.Linear(in_feats, out_dim)

    def forward(self, x_img, clin_vec):
        B, C, H, W = x_img.shape
        assert C == 3, f"Expected 3-channel image, got {C}"
        assert clin_vec.dim() == 2 and clin_vec.size(0) == B and clin_vec.size(1) == self.tab_dim, \
            f"clin_vec shape {tuple(clin_vec.shape)} != (B,{self.tab_dim})"

        pm = self.pmap(clin_vec)  # (B, c_pat)
        ch = pm.view(B, self.c_pat, 1, 1).expand(B, self.c_pat, H, W)  # (B,c_pat,H,W)
        x = torch.cat([x_img, ch], dim=1)  # (B, 3+c_pat, H, W)

        # one-time debug
        if not hasattr(self, "_dbg_printed"):
            try:
                print(f"[early/inception] x_img={tuple(x_img.shape)} clin_vec={tuple(clin_vec.shape)} "
                    f"pmap={tuple(pm.shape)} ch={tuple(ch.shape)} cat={tuple(x.shape)} "
                    f"in_conv_in={self.backbone.Conv2d_1a_3x3.conv.in_channels}")
            except Exception as e:
                print("[early/inception] debug print failed:", e)
            self._dbg_printed = True

        feats = self.backbone(x)
        logits = self.head(feats)
        return feats, logits