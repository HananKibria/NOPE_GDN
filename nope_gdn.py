from typing import Tuple, Optional, Dict
from main import nn,torch,F
from gated_delta_layer import GatedDeltaLayer
try:
    from fla.ops.kda import chunk_kda
    FLA_AVAILABLE = True
    print("FLA chunk_kda kernel available — using fused Triton GDN")
except ImportError:
    FLA_AVAILABLE = False
    print("FLA chunk_kda kernel not available — using Python-loop GDN(slower)")
    print("To install FLA, run: pip install fla-core")
    
class TubeletEmbedding(nn.Module):
    
    def __init__(self, img_size: int =224, num_frames: int =16, 
                 tubelet_size: Tuple[int,int,int] = (2,16,16), in_channels: int =3,
                 embed_dim: int =768):
        super().__init__()
        t, h, w = tubelet_size
        self.tubelet_size = tubelet_size
        self.img_size = img_size
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        
        self.projection = nn.Conv3d(in_channels, embed_dim, 
                                    kernel_size=tubelet_size, stride=tubelet_size, padding=0
        )
        
        self.num_temporal_patches = num_frames // t
        self.num_spatial_patches_h = img_size // h
        self.num_spatial_patches_w = img_size // w
        self.num_patches =(self.num_temporal_patches 
                           * self.num_spatial_patches_h 
                           * self.num_spatial_patches_w)
        
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) -> (B, T'*H'*W', D)"""
        x= self.projection(video)
        return x.flatten(2).transpose(1, 2)

    def get_grid_dims(self) -> Dict[str, int]:
        return {
            'T': self.num_temporal_patches,
            'H': self.num_spatial_patches_h,
            'W': self.num_spatial_patches_w,
            'total': self.num_patches
        }
            
class NopeMultiheadAttention(nn.Module):
    """Scaled Dot Product multi-head attention with No positional encoding.
    Purely content based attention pattern.
    """
    def __init__(self, embed_dim: int , num_heads: int,
                 dropout: float =0.1, bias: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv_proj= nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim
        
        qkv= self.qkv_proj(x).reshape(B, N, 3, H, d)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # q, k, v: (B, H, N, d)
        
        sdpa_mask = None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                sdpa_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                sdpa_mask = sdpa_mask.masked_fill(attn_mask, float('-inf'))
            else:
                sdpa_mask = attn_mask
            if sdpa_mask.dim() == 2:
                sdpa_mask = sdpa_mask[None, None, :, :]
            elif sdpa_mask.dim() == 3:
                sdpa_mask = sdpa_mask[None, :, :, :]
                
        # NOTE on dropout: SDPA's in-kernel dropout uses an internal RNG that
        # can cause `torch.utils.checkpoint` (use_reentrant=False) to save a
        # different number of tensors on forward vs recomputation, raising
        # CheckpointError. We therefore set dropout_p=0.0 inside SDPA and apply
        # self.attn_dropout as an explicit nn.Dropout op on the output, which
        # integrates correctly with checkpoint's RNG state preservation. This
        # matches the pattern used in standard ViT implementations.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask,
                                             dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.attn_dropout(out)
        return self.out_proj(out)

class DropPath(nn.Module):
    """Stochastic Depth (drop path) regularization."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim -1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        return x * mask / keep_prob
    
    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob:.3f}'
    
class NoPEVITBlock(nn.Module):
    """
    Pre-norm ViT block with optional spatial-only attention factorization.

    When spatial_tokens is set, attention operates SPATIAL-ONLY: the temporal
    dimension is folded into the batch so each frame attends independently.
    This reduces attention from O((T'·H'·W')²) to O((H'·W')²) per frame.

    Factorization principle (TRecViT, Pătrăucean et al. 2024):
        Time  → handled by GDN recurrence in the processor  (linear in T)
        Space → handled by self-attention here               (quadratic in H'·W' only)
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 spatial_tokens: int = None, drop_path: float = 0.0):
        super().__init__()
        self.spatial_tokens = spatial_tokens
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = NopeMultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, attn_mask= None):
        B, N, D = x.shape
        normed= self.norm1(x)
        
        if self.spatial_tokens is not None:
            # ── Spatial-only attention ──
            # Fold temporal dim into batch: (B, T'*S, D) → (B*T', S, D)
            S = self.spatial_tokens
            T= N // S
            assert N == T * S, (
                f"Token count {N} not divisible by spatial_tokens {S}. "
                f"Check tubelet_size vs input resolution."
            )
            normed = normed.view(B, T, S, D).reshape(B * T, S, D)
            attn_out = self.attn(normed)
            attn_out = attn_out.view(B, T, S, D).reshape(B, N, D)
            x = x + self.drop_path(attn_out)
        else:
            x = x + self.drop_path(self.attn(normed, attn_mask=attn_mask))
        
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class NoPEVideoEncoder(nn.Module):
    """
    Tubelet embedding + NoPE ViT blocks.
    No positional encoding — Conv3d is the only implicit position source.

    When factorized_attention=True (default), attention operates SPATIAL-ONLY:
    each frame's tokens attend to each other independently. Cross-frame
    interaction is deferred to the downstream GDN processor.
    """
    
    def __init__(self, img_size = 224, num_frames = 16,
                 tubelet_size = (2, 16, 16), in_channels=3,
                 embed_dim = 768, depth = 12, num_heads = 12,
                 mlp_ratio = 4.0, dropout = 0.1, use_grad_checkpoint = False, 
                 factorized_attention = True, drop_path_rate = 0.0):
        super().__init__()
        self.tubelet_embed = TubeletEmbedding(img_size, num_frames, tubelet_size, in_channels, embed_dim)
        
        #spatial_tokens for factorized attention
        grid = self.tubelet_embed.get_grid_dims()
        spatial_tokens = grid['H'] * grid['W'] if factorized_attention else None
        
        #linearly increasing drop_path rates per block
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks= nn.ModuleList([
            NoPEVITBlock(embed_dim, num_heads, mlp_ratio, dropout, 
                         spatial_tokens=spatial_tokens, drop_path=dpr[i])
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.use_grad_checkpoint = use_grad_checkpoint
        self.embed_dim = embed_dim
        self.depth = depth
        self.factorized_attention = factorized_attention
        
    def forward(self, video, attn_mask = None):
        x = self.tubelet_embed(video)
        for block in self.blocks:
            if self.use_grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, attn_mask, use_reentrant=False)
            else:
                x = block(x, attn_mask = attn_mask)
        return self.norm(x)
    
    def get_grid_dims(self):
        return self.tubelet_embed.get_grid_dims()


    
