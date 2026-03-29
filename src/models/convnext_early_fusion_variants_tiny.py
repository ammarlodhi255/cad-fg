import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class ConvNeXtTinyEarlyFusionFiLM(nn.Module):
    """
    Early-fusion FiLM for ConvNeXt-Tiny.
    - Apply FiLM modulation (per-channel gamma,beta) to the stem feature map
      using a projection from the clinical vector, then continue through the
      remaining ConvNeXt blocks.
    """
    def __init__(self, tab_dim: int, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        w = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        base = convnext_tiny(weights=w)
        # keep classifier identity, we will add our own head
        base.classifier = nn.Identity()
        self.backbone = base
        # stem conv lives at features[0][0]
        stem_conv = self.backbone.features[0][0]
        assert isinstance(stem_conv, nn.Conv2d)
        c_stem = int(stem_conv.out_channels)
        # projection to FiLM parameters for stem channels
        self.film = nn.Linear(tab_dim, 2 * c_stem)
        # final pooled feature dimension for tiny is 768
        self.feat_dim = 768
        self.head = nn.Linear(self.feat_dim, out_dim)

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        # run stem
        x = self.backbone.features[0](x_img)
        gamma, beta = self.film(clin_vec).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        x = x * (1 + gamma) + beta
        # remaining stages
        for i in range(1, len(self.backbone.features)):
            x = self.backbone.features[i](x)
        x = self.backbone.avgpool(x)
        feats = torch.flatten(x, 1)
        logits = self.head(feats)
        return feats, logits


class ConvNeXtTinyEarlyFusionPrompt(nn.Module):
    """
    Early-fusion Prompt Channels for ConvNeXt-Tiny.
    - Generate K metadata-conditioned channel maps and concatenate to the RGB
      input, patching the stem conv to accept 3+K channels (akin to ViT prompts).
    """
    def __init__(self, tab_dim: int, K: int = 4, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        self.K = int(K)
        self.tab_dim = int(tab_dim)
        w = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        base = convnext_tiny(weights=w)
        base.classifier = nn.Identity()
        self.backbone = base
        # Patch stem conv to take 3+K channels
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
        # Metadata -> K prompt channels (broadcast spatially)
        self.prompt_mlp = nn.Sequential(nn.Linear(self.tab_dim, self.K))
        self.feat_dim = 768
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
