import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights


class ConvNeXtBaseEarlyFusionFiLM(nn.Module):
    """
    Early-fusion FiLM for ConvNeXt-Base.
    - Apply FiLM modulation (per-channel gamma,beta) to the stem feature map
      using a projection from the clinical vector, then continue through the
      remaining ConvNeXt blocks.
    """
    def __init__(self, tab_dim: int, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        w = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        base = convnext_base(weights=w)
        base.classifier = nn.Identity()
        self.backbone = base
        stem_conv = self.backbone.features[0][0]
        assert isinstance(stem_conv, nn.Conv2d)
        c_stem = int(stem_conv.out_channels)
        self.film = nn.Linear(tab_dim, 2 * c_stem)
        # pooled feature dimension for base is 1024
        self.feat_dim = 1024
        self.head = nn.Linear(self.feat_dim, out_dim)

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        x = self.backbone.features[0](x_img)
        gamma, beta = self.film(clin_vec).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        x = x * (1 + gamma) + beta
        for i in range(1, len(self.backbone.features)):
            x = self.backbone.features[i](x)
        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)
        logits = self.head(feats)
        return feats, logits


class ConvNeXtBaseEarlyFusionPrompt(nn.Module):
    """
    Early-fusion Prompt Channels for ConvNeXt-Base (Concatenate K meta maps to input).
    """
    def __init__(self, tab_dim: int, K: int = 4, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        self.K = int(K)
        self.tab_dim = int(tab_dim)
        w = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        base = convnext_base(weights=w)
        base.classifier = nn.Identity()
        self.backbone = base
        stem_conv = self.backbone.features[0][0]
        assert isinstance(stem_conv, nn.Conv2d) and stem_conv.in_channels == 3
        new_stem = nn.Conv2d(
            in_channels=3 + self.K,
            out_channels=stem_conv.out_channels,
            kernel_size=stem_conv.kernel_size,
            stride=stem_conv.stride,
            padding=stem_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_stem.weight[:, :3, ...] = stem_conv.weight
        nn.init.kaiming_normal_(new_stem.weight[:, 3:, ...], nonlinearity="relu")
        self.backbone.features[0][0] = new_stem
        self.prompt_mlp = nn.Sequential(nn.Linear(self.tab_dim, self.K))
        self.feat_dim = 1024
        self.head = nn.Linear(self.feat_dim, out_dim)

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        B, C, H, W = x_img.shape
        prompts = self.prompt_mlp(clin_vec).view(B, self.K, 1, 1).expand(B, self.K, H, W)
        x = torch.cat([x_img, prompts], dim=1)
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)
        logits = self.head(feats)
        return feats, logits
