import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights


def _split_positional_embeddings(base, num_patches: int):
    """
    Returns (cls_pos, patch_pos_resized) where patch_pos_resized has length `num_patches`.
    TorchVision ViT stores a single pos_embedding of shape (1, 1+N_ref, D).
    We resize the patch positional part via bicubic interpolation when needed.
    """
    pos = base.encoder.pos_embedding  # (1, 1+N_ref, D)
    cls_pe, patch_pe = pos[:, :1, :], pos[:, 1:, :]
    if patch_pe.size(1) == num_patches:
        return cls_pe, patch_pe

    # Interpolate positional embeddings on the 2D patch grid
    d = patch_pe.shape[-1]
    n_ref = patch_pe.shape[1]
    h_ref = w_ref = int(math.sqrt(n_ref))
    h_new = w_new = int(math.sqrt(num_patches))
    if h_ref * w_ref != n_ref or h_new * w_new != num_patches:
        # Fallback: simple truncate/pad by repeating last if grid assumption fails
        if num_patches < n_ref:
            return cls_pe, patch_pe[:, :num_patches, :]
        else:
            repeat = patch_pe[:, -1:, :].repeat(1, num_patches - n_ref, 1)
            return cls_pe, torch.cat([patch_pe, repeat], dim=1)

    patch_2d = patch_pe[0].transpose(0, 1).reshape(1, d, h_ref, w_ref)
    patch_2d = F.interpolate(patch_2d, size=(h_new, w_new), mode="bicubic", align_corners=False)
    patch_new = patch_2d.reshape(1, d, h_new * w_new).transpose(1, 2)
    return cls_pe, patch_new


class _BaseViTEarly(nn.Module):
    def __init__(self, pretrained: bool = True, out_dim: int = 2):
        super().__init__()
        w = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        base = vit_b_16(weights=w)
        self.d_model = base.heads.head.in_features  # 768 for vit_b_16
        base.heads = nn.Identity()
        self.backbone = base
        self.head = nn.Linear(self.d_model, out_dim)

    def _patch_embeddings(self, x_img: torch.Tensor):
        # TorchVision ViT helper: conv_proj + flatten + linear proj -> (B, N, D)
        return self.backbone._process_input(x_img)

    def _encode(self, tokens: torch.Tensor, extra_pos: torch.Tensor | None = None):
        """
        Encode a sequence with explicit positional embeddings without letting
        torchvision's Encoder add its own pos_embedding again (which would
        mismatch when we change sequence length by adding meta/prompt tokens).

        tokens: (B, L, D)
        extra_pos: (1, L, D) or (B, L, D) broadcastable; if None, use the
                   model's stored positional embeddings (trimmed to L).
        """
        if extra_pos is None:
            pos = self.backbone.encoder.pos_embedding[:, : tokens.size(1), :]
        else:
            # Ensure pos is (1, L, D) for broadcasting
            pos = extra_pos
            if pos.dim() == 3 and pos.size(0) != 1:
                pos = pos[:1]
        x = tokens + pos
        # Manually run the encoder internals (dropout + blocks + layernorm)
        x = self.backbone.encoder.dropout(x)
        for blk in self.backbone.encoder.layers:
            x = blk(x)
        x = self.backbone.encoder.ln(x)
        return x


class ViTEarlyFusionMetaToken(_BaseViTEarly):
    """Append a metadata token to the sequence (after [CLS])."""
    def __init__(self, tab_dim: int, pretrained: bool = True, out_dim: int = 2):
        super().__init__(pretrained=pretrained, out_dim=out_dim)
        d = self.d_model
        self.meta_proj = nn.Sequential(nn.Linear(tab_dim, d), nn.ReLU(), nn.Linear(d, d))
        self.meta_pos = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.meta_pos, std=0.02)

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        B = x_img.size(0)
        patches = self._patch_embeddings(x_img)  # (B, N, D)
        N = patches.size(1)
        cls_pe, patch_pe = _split_positional_embeddings(self.backbone, N)
        cls_tok = self.backbone.class_token.expand(B, -1, -1)
        meta_tok = self.meta_proj(clin_vec).unsqueeze(1)  # (B,1,D)
        tokens = torch.cat([cls_tok, meta_tok, patches], dim=1)  # (B, 2+N, D)
        pos = torch.cat([cls_pe, self.meta_pos, patch_pe], dim=1)
        x = self._encode(tokens, pos)
        cls_out = x[:, 0]
        logits = self.head(cls_out)
        return cls_out, logits


class ViTEarlyFusionFiLM(_BaseViTEarly):
    """FiLM modulation of patch embeddings with clinical vector."""
    def __init__(self, tab_dim: int, pretrained: bool = True, out_dim: int = 2):
        super().__init__(pretrained=pretrained, out_dim=out_dim)
        d = self.d_model
        self.film = nn.Linear(tab_dim, 2 * d)

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        B = x_img.size(0)
        patches = self._patch_embeddings(x_img)  # (B, N, D)
        N = patches.size(1)
        gamma, beta = self.film(clin_vec).chunk(2, dim=-1)
        patches = patches * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        cls_tok = self.backbone.class_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tok, patches], dim=1)  # (B, 1+N, D)
        cls_pe, patch_pe = _split_positional_embeddings(self.backbone, N)
        pos = torch.cat([cls_pe, patch_pe], dim=1)
        x = self._encode(tokens, pos)
        cls_out = x[:, 0]
        logits = self.head(cls_out)
        return cls_out, logits


class ViTEarlyFusionPrompt(_BaseViTEarly):
    """Generate K prompt tokens from metadata and prepend to sequence."""
    def __init__(self, tab_dim: int, K: int = 4, pretrained: bool = True, out_dim: int = 2):
        super().__init__(pretrained=pretrained, out_dim=out_dim)
        d = self.d_model
        self.K = int(K)
        self.prompt_gen = nn.Sequential(nn.Linear(tab_dim, d * self.K))
        self.prompt_pos = nn.Parameter(torch.zeros(1, self.K, d))
        nn.init.trunc_normal_(self.prompt_pos, std=0.02)

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        B = x_img.size(0)
        patches = self._patch_embeddings(x_img)  # (B, N, D)
        N = patches.size(1)
        cls_tok = self.backbone.class_token.expand(B, -1, -1)
        prompts = self.prompt_gen(clin_vec).view(B, self.K, -1)
        tokens = torch.cat([cls_tok, prompts, patches], dim=1)  # (B, 1+K+N, D)
        cls_pe, patch_pe = _split_positional_embeddings(self.backbone, N)
        # prompt pos: learned, or zeros if you prefer
        pos = torch.cat([cls_pe, self.prompt_pos, patch_pe], dim=1)
        x = self._encode(tokens, pos)
        cls_out = x[:, 0]
        logits = self.head(cls_out)
        return cls_out, logits


class ViTEarlyFusionCLSCrossAttn(_BaseViTEarly):
    """
    Update the CLS token via cross-attention to a metadata token, then proceed as usual.
    Keeps sequence length unchanged (1+N).
    """
    def __init__(self, tab_dim: int, pretrained: bool = True, out_dim: int = 2, nheads: int = 8):
        super().__init__(pretrained=pretrained, out_dim=out_dim)
        d = self.d_model
        self.meta_proj = nn.Linear(tab_dim, d)
        self.cross_attn = nn.MultiheadAttention(d, num_heads=nheads, batch_first=True)
        self.cls_gate = nn.Parameter(torch.tensor(1.0))  # residual scaling

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        B = x_img.size(0)
        patches = self._patch_embeddings(x_img)  # (B, N, D)
        N = patches.size(1)
        cls_tok = self.backbone.class_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tok, patches], dim=1)  # (B, 1+N, D)
        cls_pe, patch_pe = _split_positional_embeddings(self.backbone, N)
        pos = torch.cat([cls_pe, patch_pe], dim=1)
        tokens = tokens + pos

        # Cross-attention: CLS queries metadata; metadata is single token
        cls = tokens[:, :1, :]  # (B,1,D)
        meta = self.meta_proj(clin_vec).unsqueeze(1)  # (B,1,D)
        cls_updated, _ = self.cross_attn(query=cls, key=meta, value=meta)
        tokens = torch.cat([cls + self.cls_gate * cls_updated, tokens[:, 1:, :]], dim=1)

        # Encode through transformer manually (we already added pos)
        x = self.backbone.encoder.dropout(tokens)
        for blk in self.backbone.encoder.layers:
            x = blk(x)
        x = self.backbone.encoder.ln(x)
        cls_out = x[:, 0]
        logits = self.head(cls_out)
        return cls_out, logits


class ViTMidFusionDeepPrompt(_BaseViTEarly):
    """
    Deep prompt tuning with metadata:
    - At every transformer block, prepend K prompt tokens generated from the
      clinical vector, add per-layer learned prompt positional embeddings,
      run the block, then strip the prompts before the next block.
    - Conditioning occurs at all depths ("middle fusion").
    """
    def __init__(self, tab_dim: int, K: int = 4, pretrained: bool = True, out_dim: int = 2):
        super().__init__(pretrained=pretrained, out_dim=out_dim)
        self.d_model = self.d_model  # explicit alias
        self.K = int(K)
        d = self.d_model
        # Metadata → prompt token generator
        self.prompt_mlp = nn.Sequential(
            nn.Linear(tab_dim, d), nn.ReLU(), nn.Linear(d, d * self.K)
        )
        # Per-layer prompt positional embeddings
        L = len(self.backbone.encoder.layers)
        self.prompt_pos = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.K, d)) for _ in range(L)
        ])
        for p in self.prompt_pos:
            nn.init.trunc_normal_(p, std=0.02)

    def forward(self, x_img: torch.Tensor, clin_vec: torch.Tensor):
        B = x_img.size(0)
        patches = self._patch_embeddings(x_img)  # (B, N, D)
        N = patches.size(1)
        cls_pe, patch_pe = _split_positional_embeddings(self.backbone, N)

        # Build base sequence with absolute pos once
        cls_tok = self.backbone.class_token.expand(B, -1, -1)  # (B,1,D)
        seq = torch.cat([cls_tok, patches], dim=1)             # (B,1+N,D)
        pos = torch.cat([cls_pe, patch_pe], dim=1)             # (1,1+N,D)
        x = seq + pos                                          # (B,1+N,D)

        # Prompts from metadata (shared across layers, with per-layer pos offsets)
        base_prompts = self.prompt_mlp(clin_vec).view(B, self.K, -1)  # (B,K,D)

        # Manually run encoder with deep prompts
        x = self.backbone.encoder.dropout(x)
        for i, blk in enumerate(self.backbone.encoder.layers):
            prompts_i = base_prompts + self.prompt_pos[i]  # (B,K,D)
            x = torch.cat([x[:, :1, :], prompts_i, x[:, 1:, :]], dim=1)  # [CLS] + K + patches
            x = blk(x)
            # strip prompts before next block
            x = torch.cat([x[:, :1, :], x[:, 1 + self.K :, :]], dim=1)
        x = self.backbone.encoder.ln(x)

        cls_out = x[:, 0]
        logits = self.head(cls_out)
        return cls_out, logits
