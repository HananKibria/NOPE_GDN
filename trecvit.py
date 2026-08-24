from typing import Tuple, Optional, Dict
from main import nn,torch,F
import math


try:
    from fla.ops.hgrn import fused_recurrent_hgrn as _fused_hgrn
    HGRN_AVAILABLE = True
    print("✅ FLA fused_recurrent_hgrn available — using fused Triton LRU")
except ImportError:
    HGRN_AVAILABLE = False
    print("⚠️ FLA HGRN not available — using Python-loop LRU fallback (CPU/MPS path)")
    print("  Install with: pip install fla-core")
    
class _TRecViT_RMSNorm(nn.Module):
    """RMSNorm fallback for older PyTorch (no nn.RMSNorm)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight
    
def _trecvit_rmsnorm(dim):
    return nn.RMSNorm(dim) if hasattr(nn, "RMSNorm") else _TRecViT_RMSNorm(dim)

class BlockDiagonalLinear(nn.Module):
    """Linear layer with a block-diagonal weight matrix."""
    def __init__(self, in_features, out_features, num_blocks, bias=True):
        super().__init__()
        assert in_features % num_blocks == 0, \
            f"in_features ({in_features}) must be divisible by num_blocks ({num_blocks})"
        assert out_features % num_blocks == 0, \
            f"out_features ({out_features}) must be divisible by num_blocks ({num_blocks})"
        self.num_blocks = num_blocks
        self.in_block = in_features // num_blocks
        self.out_block = out_features // num_blocks
        # PyTorch Linear-style init: U(-1/√k, +1/√k) with k = in_features
        bound = 1.0 / math.sqrt(self.in_block)
        self.weight = nn.Parameter(
            torch.empty(num_blocks, self.out_block, self.in_block).uniform_(-bound, bound)
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        """x: [..., in_features] -> [..., out_features]"""
        shape = x.shape
        x = x.reshape(*shape[:-1], self.num_blocks, self.in_block)
        out = torch.einsum("gmk,...gk->...gm", self.weight, x)
        out = out.reshape(*shape[:-1], -1)
        if self.bias is not None:
            out = out + self.bias
        return out
    
class RealLRU(nn.Module):
    def __init__(self, width, lru_width, num_heads,
                 conv1d_temporal_width=4,
                 min_rad=0.5, max_rad=0.999,
                 residual_init_scale=1.0):
        super().__init__()
        self.width = width
        self.lru_width = lru_width
        self.num_heads = num_heads
        self.conv1d_temporal_width = conv1d_temporal_width

        # Full Linears for the LRU/output paths (paper uses full Dense here)
        self.linear_x = nn.Linear(width, lru_width)
        self.linear_y = nn.Linear(width, lru_width)

        # Block-diagonal for the two RG-LRU gates (paper: BlockDiagonalLinear)
        self.input_gate = BlockDiagonalLinear(width, lru_width, num_blocks=num_heads)
        self.a_gate     = BlockDiagonalLinear(width, lru_width, num_blocks=num_heads)

        # Causal depthwise Conv1d. padding=0 — caller supplies left-pad.
        self.conv1d = nn.Conv1d(
            lru_width, lru_width,
            kernel_size=conv1d_temporal_width,
            padding=0,
            groups=lru_width,
        )

        # log_a init: uniform such that initial UNGATED α ∈ [min_rad, max_rad]
        c = 8.0
        s_lo = -math.log(max_rad) / c
        s_hi = -math.log(min_rad) / c
        log_a_lo = math.log(math.exp(s_lo) - 1.0)
        log_a_hi = math.log(math.exp(s_hi) - 1.0)
        self.log_a = nn.Parameter(
            torch.empty(lru_width).uniform_(log_a_lo, log_a_hi)
        )

        self.out_proj = nn.Linear(lru_width, width)
        if residual_init_scale != 1.0:
            with torch.no_grad():
                self.out_proj.weight.mul_(residual_init_scale)

    def forward_chunk(self, x, lru_state=None, conv_state=None):
        """Process a chunk with explicit state passing.

        Args:
            x:           [B*N, T, width]
            lru_state:   [B*N, lru_width] or None (start of stream — h_0 = 0)
            conv_state:  [B*N, lru_width, K-1] or None (zeros)

        Returns:
            out:             [B*N, T, width]
            new_lru_state:   [B*N, lru_width]
            new_conv_state:  [B*N, lru_width, K-1]
        """
        bn, T, _ = x.shape
        K = self.conv1d_temporal_width
        is_sequence_start = lru_state is None

        # --- LRU input branch: linear_x -> Conv1d (NO SiLU between Conv and LRU)
        u = self.linear_x(x)
        u_t = u.transpose(1, 2)
        if conv_state is None:
            conv_state = torch.zeros(
                bn, self.lru_width, K - 1, dtype=u_t.dtype, device=u_t.device
            )
        conv_input = torch.cat([conv_state, u_t], dim=-1)
        u_t = self.conv1d(conv_input)
        new_conv_state = conv_input[..., -(K - 1):]
        u = u_t.transpose(1, 2)

        # --- Output-gate branch (computed here but applied AFTER recurrence)
        y_branch = F.gelu(self.linear_y(x))

        # --- Two RG-LRU data-dependent gates (block-diagonal)
        gate_x = torch.sigmoid(self.input_gate(x))
        gate_a = torch.sigmoid(self.a_gate(x))

        a_real = -8.0 * F.softplus(self.log_a)
        log_alpha = a_real * gate_a

        multiplier = torch.sqrt(torch.clamp(
            1.0 - torch.exp(2.0 * log_alpha), min=1e-6
        ))
        # Reset-position handling: at sequence start (no carry-in), multiplier=1
        # for t=0 because h_0 = 0 already and we don't want to also rescale.
        if is_sequence_start and T > 0:
            ones = torch.ones_like(multiplier[:, :1])
            multiplier = torch.cat([ones, multiplier[:, 1:]], dim=1)

        gated_input = multiplier * gate_x * u

        # --- Recurrence with state continuity
        if HGRN_AVAILABLE and gated_input.is_cuda:
            o, new_lru_state = _fused_hgrn(
                gated_input.contiguous(), log_alpha.contiguous(),
                initial_state=lru_state,
                output_final_state=True,
            )
        else:
            if lru_state is None:
                h = torch.zeros(
                    bn, self.lru_width, dtype=gated_input.dtype, device=gated_input.device
                )
            else:
                h = lru_state
            alpha = torch.exp(log_alpha)
            outs = []
            for t in range(T):
                h = alpha[:, t] * h + gated_input[:, t]
                outs.append(h)
            o = torch.stack(outs, dim=1)
            new_lru_state = h

        # Output gating + write-back projection
        out = self.out_proj(o * y_branch)
        return out, new_lru_state, new_conv_state

    def forward(self, x):
        out, _, _ = self.forward_chunk(x, None, None)
        return out
    
class LRUResidualBlock(nn.Module):
    """RMSNorm → RealLRU → residual."""
    def __init__(self, width, lru_width, num_heads, conv1d_temporal_width=4,
                 min_rad=0.5, residual_init_scale=1.0):
        super().__init__()
        self.norm = _trecvit_rmsnorm(width)
        self.lru = RealLRU(
            width, lru_width, num_heads, conv1d_temporal_width,
            min_rad=min_rad, residual_init_scale=residual_init_scale,
        )

    def forward_chunk(self, x, lru_state=None, conv_state=None):
        out, new_lru, new_conv = self.lru.forward_chunk(
            self.norm(x), lru_state, conv_state
        )
        return x + out, new_lru, new_conv

    def forward(self, x):
        out, _, _ = self.forward_chunk(x, None, None)
        return out
    
class TRecViTTokenizer(nn.Module):
    """3D Conv tokenizer + learned 2D spatial posemb + cls token (one per frame).

    Output: [B, T_p, n_spatial+1, width]. T_p derived from runtime conv output.
    """
    def __init__(self, img_size=224, num_frames=32, patch_size=(2, 16, 16),
                 in_channels=3, width=384):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels, width,
            kernel_size=patch_size, stride=patch_size, padding=0,
        )
        self.T_p = num_frames // patch_size[0]
        self.H_p = img_size // patch_size[1]
        self.W_p = img_size // patch_size[2]
        self.n_spatial = self.H_p * self.W_p
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_spatial, width) / math.sqrt(width)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))

    def forward(self, video):
        B = video.shape[0]
        x = self.proj(video)
        x = x.flatten(3).permute(0, 2, 3, 1).contiguous()
        x = x + self.pos_embed.unsqueeze(1)
        T_p = x.shape[1]
        cls = self.cls_token.expand(B, T_p, -1, -1)
        return torch.cat([cls, x], dim=2)

class _TRecViTSpatialBlock(nn.Module):
    """Standard ViT Encoder1DBlock — default init, NO 2/depth scaling."""
    def __init__(self, width, num_heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            width, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, width),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        y = self.norm1(x)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x
    
class LRUViT(nn.Module):
    """Alternating temporal RG-LRU + spatial SA. `depth` layers each.

    `num_heads` is shared between the spatial attention 
     and the LRU's BlockDiagonalLinear gates 
    """
    def __init__(self, depth, width, mlp_dim, num_heads,
                 conv1d_temporal_width=4, state_multiplier=2, min_rad=0.5,
                 dropout=0.0):
        super().__init__()
        lru_width = state_multiplier * width
        # Residual init scale applies ONLY to the recurrent residual block.
        residual_init_scale = math.sqrt(2.0 / depth)
        self.temporal = nn.ModuleList([
            LRUResidualBlock(
                width, lru_width, num_heads, conv1d_temporal_width,
                min_rad=min_rad, residual_init_scale=residual_init_scale,
            )
            for _ in range(depth)
        ])
        self.spatial = nn.ModuleList([
            _TRecViTSpatialBlock(width, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(width)

    def forward_chunk(self, x, states=None):
        B, T_p, N, C = x.shape
        if states is None:
            states = [(None, None)] * len(self.temporal)
        new_states = []
        for layer_idx, (lru_block, sa_block) in enumerate(
            zip(self.temporal, self.spatial)
        ):
            xt = x.permute(0, 2, 1, 3).reshape(B * N, T_p, C)
            lru_state_in, conv_state_in = states[layer_idx]
            xt, new_lru, new_conv = lru_block.forward_chunk(
                xt, lru_state_in, conv_state_in
            )
            new_states.append((new_lru, new_conv))
            x = xt.reshape(B, N, T_p, C).permute(0, 2, 1, 3).contiguous()
            xs = x.reshape(B * T_p, N, C)
            xs = sa_block(xs)
            x = xs.reshape(B, T_p, N, C)
        return self.final_norm(x), new_states

    def forward(self, x):
        out, _ = self.forward_chunk(x, None)
        return out
    
class TRecViTClassifier(nn.Module):
    """Tokenizer → LRUViT → pre_logits (Dense+tanh) → temporal mean → cls head."""

    def __init__(self, img_size=224, num_frames=32, patch_size=(2, 16, 16),
                 in_channels=3, width=384, depth=16, num_heads=6, mlp_ratio=4.0,
                 conv1d_temporal_width=4, state_multiplier=2, min_rad=0.5,
                 rep_size=3072, num_classes=174, dropout=0.0, head_dropout=0.0):
        super().__init__()
        mlp_dim = int(width * mlp_ratio)
        self.tokenizer = TRecViTTokenizer(
            img_size=img_size, num_frames=num_frames, patch_size=patch_size,
            in_channels=in_channels, width=width,
        )
        self.encoder = LRUViT(
            depth=depth, width=width, mlp_dim=mlp_dim, num_heads=num_heads,
            conv1d_temporal_width=conv1d_temporal_width,
            state_multiplier=state_multiplier, min_rad=min_rad, dropout=dropout,
        )
        self.pre_logits = nn.Linear(width, rep_size)
        self.head_dropout = nn.Dropout(head_dropout)
        self.cls_head = nn.Linear(rep_size, num_classes)

    def forward_chunk(self, video_chunk, states=None):
        tokens = self.tokenizer(video_chunk)
        encoded, new_states = self.encoder.forward_chunk(tokens, states)
        cls_per_frame = encoded[:, :, 0, :]
        x = torch.tanh(self.pre_logits(cls_per_frame))
        x = x.mean(dim=1)
        x = self.head_dropout(x)
        return self.cls_head(x), new_states

    def forward(self, video):
        logits, _ = self.forward_chunk(video, None)
        return logits
