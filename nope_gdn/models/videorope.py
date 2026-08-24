import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import TubeletEmbedding
from .heads import VideoClassificationHead


def rotate_half(x):
    """Rotate half the hidden dims of the input for RoPE application."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    """Apply rotary position embedding with proper rotation."""
    return (x * cos) + (rotate_half(x) * sin)


class VideoRoPE3D(nn.Module):
    """
    3D Rotary Position Embeddings following VideoRoPE.
    """

    def __init__(self, head_dim, num_heads, base=10000, temporal_spacing=2.0):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.temporal_spacing = temporal_spacing

        # mrope_section: splits head_dim//2 into [h, w, t] frequency pairs
        # Same 3:3:2 ratio as Qwen2-VL's [24, 24, 16] for head_dim=128
        half_head = head_dim // 2
        h_pairs = (half_head * 3) // 8
        w_pairs = h_pairs
        t_pairs = half_head - h_pairs - w_pairs
        n_spatial = h_pairs + w_pairs  # always even (= 2 * h_pairs)
        assert n_spatial % 2 == 0, "n_spatial must be even for h/w interleaving"

        self.mrope_section = [h_pairs, w_pairs, t_pairs]  # e.g. [9, 9, 6] for head_dim=48
        self.n_spatial = n_spatial
        self.t_pairs = t_pairs

        # Single shared inv_freq over head_dim (matching original exactly).
        # All 3 axes multiply against the SAME full inv_freq; the interleaving
        # in forward() selects which axis's position ID applies to each dim.
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # shape: (head_dim // 2,)

        self.register_buffer('inv_freq', inv_freq)

    def create_diagonal_positions(self, T, H, W, device, temporal_scale=1.0):
        """
        Create diagonal-layout 3D positions for all T*H*W tokens.

        pos_t = t * temporal_spacing
        pos_h = t * temporal_spacing + (h - (H-1)//2)
        pos_w = t * temporal_spacing + (w - (W-1)//2)

        Returns dict with 't_pos', 'h_pos', 'w_pos', each shape (T*H*W,).
        """
        t_base = torch.arange(T, device=device).float() * self.temporal_spacing * temporal_scale
        h_offset = torch.arange(H, device=device).float() - (H - 1) // 2
        w_offset = torch.arange(W, device=device).float() - (W - 1) // 2

        t_expanded = t_base.repeat_interleave(H * W)
        h_off_expanded = h_offset.repeat_interleave(W).repeat(T)
        w_off_expanded = w_offset.repeat(T * H)

        return {
            't_pos': t_expanded,
            'h_pos': t_expanded + h_off_expanded,
            'w_pos': t_expanded + w_off_expanded,
        }

    def forward(self, token_positions):
        """
        Compute per-head cos/sin with the interleaved mrope_section layout
        used by the original VideoRoPE.

        Returns:
            (cos, sin) each of shape (num_tokens, num_heads, head_dim)
        """
        t_pos = token_positions['t_pos']  # (N,)
        h_pos = token_positions['h_pos']  # (N,)
        w_pos = token_positions['w_pos']  # (N,)

        half_head = self.head_dim // 2
        ns = self.n_spatial

        # Compute angles for each axis against the FULL shared inv_freq.
        # Each is (N, half_head).
        angles_h_full = torch.outer(h_pos, self.inv_freq)
        angles_w_full = torch.outer(w_pos, self.inv_freq)
        angles_t_full = torch.outer(t_pos, self.inv_freq)

        # Interleaved spatial band: even dims -> h, odd dims -> w,
        # both using the SAME frequency index as their position in head_dim.
        # This matches the original's even j -> row 1 (h), odd j -> row 2 (w).
        spatial_angles = torch.empty(
            t_pos.shape[0], ns,
            device=t_pos.device, dtype=angles_h_full.dtype,
        )
        spatial_angles[:, 0::2] = angles_h_full[:, 0:ns:2]   # dims 0,2,4,... <- h_pos * inv_freq[0,2,4,...]
        spatial_angles[:, 1::2] = angles_w_full[:, 1:ns:2]   # dims 1,3,5,... <- w_pos * inv_freq[1,3,5,...]

        # Contiguous temporal band: t with tail frequencies.
        temporal_angles = angles_t_full[:, ns:half_head]     # (N, t_pairs)

        # Concat: [interleaved spatial | contiguous temporal] = (N, half_head)
        angles_per_head = torch.cat([spatial_angles, temporal_angles], dim=-1)

        # Duplicate for rotate_half: (N, head_dim)
        freqs_per_head = torch.cat([angles_per_head, angles_per_head], dim=-1)

        cos = freqs_per_head.cos()  # (N, head_dim)
        sin = freqs_per_head.sin()  # (N, head_dim)

        # Expand to all heads (every head gets same cos/sin)
        # Shape: (N, num_heads, head_dim)
        cos = cos.unsqueeze(1).expand(-1, self.num_heads, -1)
        sin = sin.unsqueeze(1).expand(-1, self.num_heads, -1)

        return cos, sin


class RoPEAttention(nn.Module):
    """
    Multi-head attention with 3D VideoRoPE applied AFTER head splitting.

    Every head gets the identical [h, w, t] frequency mix via mrope_section.
    bias=True on qkv_proj and out_proj (matches VideoMAE for weight transfer).
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 rope_dims: dict = None, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, rope_cache=None, attn_mask=None):
        B, N, D = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.embed_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Split into heads FIRST (before RoPE)
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (B, num_heads, N, head_dim)

        # Apply 3D VideoRoPE AFTER head splitting (per-head)
        if rope_cache is not None:
            cos, sin = rope_cache  # each (N, num_heads, head_dim)
            # Reshape to (1, num_heads, N, head_dim) for broadcasting
            # Cast to Q/K dtype so AMP (fp16/bf16) doesn't silently up-cast Q/K to fp32,
            # which would inflate memory and disable the fastest attention kernels.
            cos = cos.permute(1, 0, 2).unsqueeze(0).to(device=q.device, dtype=q.dtype)  # (1, num_heads, N, head_dim)
            sin = sin.permute(1, 0, 2).unsqueeze(0).to(device=q.device, dtype=q.dtype)
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

        dropout_p = self.attn_dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class RoPEViTBlock(nn.Module):
    """ViT block with VideoRoPE attention."""

    def __init__(self, embed_dim, num_heads, rope_dims=None,
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = RoPEAttention(embed_dim, num_heads, rope_dims, dropout)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        h = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, h), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, embed_dim), nn.Dropout(dropout)
        )

    def forward(self, x, rope_cache=None, attn_mask=None):
        x = x + self.attn(self.norm1(x), rope_cache, attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class RoPEVideoClassifier(nn.Module):
    """
    Standard Video ViT with 3D VideoRoPE (ICML 2025) — the strong baseline.

    No GDN, no processor stage. Total depth = encoder_depth + processor_depth
    for fair parameter comparison with the NoPE+GDN model.
    """

    def __init__(self, img_size=224, num_frames=16,
                 tubelet_size=(2, 16, 16), in_channels=3,
                 embed_dim=384, total_depth=16, num_heads=6,
                 num_classes=174, mlp_ratio=4.0, dropout=0.1,
                 head_dropout=0.0, temporal_spacing=2.0):
        super().__init__()

        self.tubelet_embed = TubeletEmbedding(
            img_size, num_frames, tubelet_size, in_channels, embed_dim)

        grid = self.tubelet_embed.get_grid_dims()

        # VideoRoPE operates per-head (matching original paper)
        self.rope = VideoRoPE3D(
            head_dim=embed_dim // num_heads,
            num_heads=num_heads,
            temporal_spacing=temporal_spacing,
        )

        # Store tubelet size for dynamic grid computation at any frame count
        self.tubelet_size = tubelet_size

        self.blocks = nn.ModuleList([
            RoPEViTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio,
                         dropout=dropout)
            for _ in range(total_depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = VideoClassificationHead(embed_dim, num_classes, head_dropout)

    def forward(self, video, temporal_scale=1.0):
        x = self.tubelet_embed(video)

        # Compute grid dims dynamically from actual input (variable-length support)
        T = video.shape[2] // self.tubelet_size[0]
        H = video.shape[3] // self.tubelet_size[1]
        W = video.shape[4] // self.tubelet_size[2]

        # Compute diagonal-layout 3D positions for current input
        token_positions = self.rope.create_diagonal_positions(
            T, H, W, x.device, temporal_scale=temporal_scale)
        rope_cache = self.rope(token_positions)

        for block in self.blocks:
            x = block(x, rope_cache=rope_cache)
        x = self.norm(x)
        return self.head(x)
