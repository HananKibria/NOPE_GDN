import torch
import torch.nn as nn

from .backbone import TubeletEmbedding
from .heads import VideoClassificationHead
from .videorope import RoPEViTBlock


class AxialMixedRoPE3D(nn.Module):
    """3D Axial / Mixed RoPE for video transformers.

    Reference: Heo et al., "Rotary Position Embedding for Vision Transformer",
    ECCV 2024 (extended from 2D images to 3D videos).

    mode: 'axial' (fixed freqs) or 'mixed' (learnable per-head freqs).
    t_ratio: fraction of rotation pairs allocated to the temporal axis
    (default 0.25 mirrors VideoRoPE's "low-freq temporal" idea).
    """

    def __init__(self, head_dim: int, num_heads: int,
                 mode: str = "axial", base: float = 10000.0,
                 t_ratio: float = 0.25):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even"
        assert mode in ("axial", "mixed"), f"unknown mode: {mode}"
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.mode = mode

        half = head_dim // 2  # number of rotation pairs
        n_t = max(1, int(round(half * t_ratio)))
        n_h = (half - n_t) // 2
        n_w = half - n_t - n_h
        assert n_t + n_h + n_w == half
        self.n_t, self.n_h, self.n_w = n_t, n_h, n_w

        # Standard RoPE frequency schedules per axis (each over its own block size).
        freq_t = 1.0 / (base ** (torch.arange(0, n_t).float() / max(n_t, 1)))
        freq_h = 1.0 / (base ** (torch.arange(0, n_h).float() / max(n_h, 1)))
        freq_w = 1.0 / (base ** (torch.arange(0, n_w).float() / max(n_w, 1)))

        if mode == "axial":
            # Fixed per-axis frequencies; shared across heads.
            self.register_buffer("inv_freq_t", freq_t)
            self.register_buffer("inv_freq_h", freq_h)
            self.register_buffer("inv_freq_w", freq_w)
        else:  # mixed
            # Per-head learnable (theta_t, theta_h, theta_w) for every pair.
            # Init = axial (each pair sees only its assigned axis).
            theta_t = torch.zeros(num_heads, half)
            theta_h = torch.zeros(num_heads, half)
            theta_w = torch.zeros(num_heads, half)
            theta_t[:, :n_t]            = freq_t.unsqueeze(0)
            theta_h[:, n_t:n_t + n_h]   = freq_h.unsqueeze(0)
            theta_w[:, n_t + n_h:]      = freq_w.unsqueeze(0)
            self.theta_t = nn.Parameter(theta_t)
            self.theta_h = nn.Parameter(theta_h)
            self.theta_w = nn.Parameter(theta_w)

    def forward(self, T: int, H: int, W: int, device, temporal_scale: float = 1.0):
        """Compute (cos, sin) each of shape (N, num_heads, head_dim) where N = T*H*W.

        Token order must match TubeletEmbedding.flatten(2): T outer, then H, then W.
        """
        # Position interpolation for length generalization:
        # scale temporal positions by `temporal_scale` (e.g. T_train/T_test)
        # so inference positions land back inside the trained range.
        t_idx = torch.arange(T, device=device).float() * temporal_scale
        h_idx = torch.arange(H, device=device).float()
        w_idx = torch.arange(W, device=device).float()

        # Build per-token (t, h, w) coordinates in flatten order
        t_grid = t_idx.repeat_interleave(H * W)        # (N,)
        h_grid = h_idx.repeat_interleave(W).repeat(T)  # (N,)
        w_grid = w_idx.repeat(T * H)                   # (N,)

        if self.mode == "axial":
            ang_t = torch.outer(t_grid, self.inv_freq_t)  # (N, n_t)
            ang_h = torch.outer(h_grid, self.inv_freq_h)  # (N, n_h)
            ang_w = torch.outer(w_grid, self.inv_freq_w)  # (N, n_w)
            angles = torch.cat([ang_t, ang_h, ang_w], dim=-1)   # (N, half)
            freqs  = torch.cat([angles, angles], dim=-1)        # (N, head_dim)
            cos = freqs.cos().unsqueeze(1).expand(-1, self.num_heads, -1)
            sin = freqs.sin().unsqueeze(1).expand(-1, self.num_heads, -1)
        else:  # mixed: per-head learnable mixing
            # angles[n, h, k] = t_n * theta_t[h,k] + h_n * theta_h[h,k] + w_n * theta_w[h,k]
            angles = (
                t_grid[:, None, None] * self.theta_t[None, :, :] +
                h_grid[:, None, None] * self.theta_h[None, :, :] +
                w_grid[:, None, None] * self.theta_w[None, :, :]
            )  # (N, num_heads, half)
            freqs = torch.cat([angles, angles], dim=-1)  # (N, num_heads, head_dim)
            cos = freqs.cos()
            sin = freqs.sin()

        return cos, sin


class AxialMixedRoPEVideoClassifier(nn.Module):
    """Video ViT with 3D Axial or Mixed RoPE.

    Drop-in sibling of RoPEVideoClassifier — same depth, same blocks, same
    parameter budget (modulo a tiny num_heads * 3 * head_dim term for mixed).
    """

    def __init__(self, img_size=224, num_frames=16,
                 tubelet_size=(2, 16, 16), in_channels=3,
                 embed_dim=384, total_depth=16, num_heads=6,
                 num_classes=174, mlp_ratio=4.0, dropout=0.1,
                 head_dropout=0.0, rope_mode="axial"):
        super().__init__()

        self.tubelet_embed = TubeletEmbedding(
            img_size, num_frames, tubelet_size, in_channels, embed_dim)
        self.tubelet_size = tubelet_size

        self.rope = AxialMixedRoPE3D(
            head_dim=embed_dim // num_heads,
            num_heads=num_heads,
            mode=rope_mode,
        )

        self.blocks = nn.ModuleList([
            RoPEViTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio,
                         dropout=dropout)
            for _ in range(total_depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = VideoClassificationHead(embed_dim, num_classes, head_dropout)

    def forward(self, video, temporal_scale: float = 1.0):
        x = self.tubelet_embed(video)
        T = video.shape[2] // self.tubelet_size[0]
        H = video.shape[3] // self.tubelet_size[1]
        W = video.shape[4] // self.tubelet_size[2]
        rope_cache = self.rope(T, H, W, device=x.device,
                               temporal_scale=temporal_scale)
        for block in self.blocks:
            x = block(x, rope_cache=rope_cache)
        x = self.norm(x)
        return self.head(x)
