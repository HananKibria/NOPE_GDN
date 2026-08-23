from typing import Tuple, Optional, Dict
from main import nn,torch,F
from gated_delta_layer import GatedDeltaLayer, BiGatedDeltaLayer
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
    
class HybridBlock(nn.Module):
    """
    Single processor block — either GDN (position-aware) or NoPE
    (global/spatial context). Both include pre-norm MLP.

    block_type='gdn':
      Pre-norm → GatedDeltaLayer → residual → Pre-norm → MLP → residual
      Provides: local ordering (conv), learned decay (alpha), memory (state)

    block_type='nope':
      Pre-norm → NoPE-MHA → residual → Pre-norm → MLP → residual
      Provides: spatial context mixing, receives position from residual stream
      When spatial_tokens is set, attention is spatial-only (factorized).
    """
    def __init__(self, dim: int, num_heads: int,
                block_type: str = 'gdn',
                head_dim: int = None, chunk_size: int = 64,
                mlp_ratio: float = 4.0, dropout: float = 0.1,
                channel_wise_decay: bool = True,
                allow_neg_eigval: bool = False,
                spatial_tokens: int = None,
                drop_path: float = 0.0,
                bidirectional: bool = False,
                flip_mode: str = 'temporal',
                gdn_temporal_only: bool = False,
                decay_target_dt: float = None,
                 a_init_range: Tuple[float, float] = (1.0, 16.0)):
        super().__init__()
        self.block_type = block_type
        self.bidirectional = bidirectional and (block_type == 'gdn')

        self.gdn_temporal_only = gdn_temporal_only and (block_type == 'gdn')
        
        # spatial_tokens has DUAL semantics:
        #   - For NoPE blocks: None → global attention; value → fold spatial
        #     dim into batch for spatial-only (factorized) attention.
        #   - For GDN+bidi  : value used by BiGatedDeltaLayer to reconstruct
        #     the (T, S) grid for temporal/spatial axis-specific flipping.
        self.spatial_tokens = spatial_tokens
        
        self.flip_mode = flip_mode
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        if block_type == 'gdn':
            hd = head_dim or (dim // num_heads)
            if bidirectional:
                if num_heads % 2 != 0:
                    raise ValueError(
                        f"bidirectional GDN requires num_heads to be even "
                        f"(half-H parity); got num_heads={num_heads}. "
                    )
                bi_flip = 'flat' if self.gdn_temporal_only else flip_mode
                bi_spatial = None if self.gdn_temporal_only else spatial_tokens
                self.layer = BiGatedDeltaLayer(
                    dim, num_heads // 2, hd, chunk_size,
                    channel_wise_decay=channel_wise_decay,
                    allow_neg_eigval=allow_neg_eigval,
                    flip_mode=bi_flip,
                    spatial_tokens=bi_spatial,
                    decay_target_dt=decay_target_dt,
                    a_init_range=a_init_range)
            else:
                 self.layer = GatedDeltaLayer(
                    dim, num_heads, hd, chunk_size,
                    channel_wise_decay=channel_wise_decay,
                    allow_neg_eigval=allow_neg_eigval,
                    decay_target_dt=decay_target_dt,
                    a_init_range=a_init_range)
        else:
            self.layer = NopeMultiheadAttention(dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, h), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, dim), nn.Dropout(dropout)
        )
        
    def _run_time_only_gdn(self, xt, state=None, conv_state=None, streaming=False):
        BS = xt.shape[0]
        H = getattr(self.layer, 'num_heads', None) or getattr(
            self.layer, 'num_heads_per_dir', 8)
        cap = getattr(self, '_gdn_max_seqs', 0) or max(64, 12000 // H)
        if BS <= cap:
            if streaming:
                return self.layer.forward_chunk(xt, state, conv_state)
            out, _ = self.layer(xt, None)
            return out, None, None
        outs, sts, cq, ck, cv = [], [], [], [], []
        for i in range(0, BS, cap):
            sl = slice(i, min(i + cap, BS))
            if streaming:
                st = None if state is None else state[sl]
                cst = None if conv_state is None else tuple(c[sl] for c in conv_state)
                o, ns, ncs = self.layer.forward_chunk(xt[sl], st, cst)
                sts.append(ns); cq.append(ncs[0]); ck.append(ncs[1]); cv.append(ncs[2])
            else:
                o, _ = self.layer(xt[sl], None)
            outs.append(o)
        out = torch.cat(outs, dim=0)
        if not streaming:
            return out, None, None
        return out, torch.cat(sts, 0), (torch.cat(cq, 0), torch.cat(ck, 0), torch.cat(cv, 0))
    
    def forward(self, x, attn_mask=None, state=None):
    
        """
        Returns: (output, new_state)
        new_state is only meaningful for GDN blocks.
        """
        B, N, D = x.shape
        normed = self.norm1(x)
        
        if self.block_type == 'gdn':
            if self.gdn_temporal_only:
                S = self.spatial_tokens
                assert S is not None and N % S == 0, (
                    f"gdn_temporal_only needs spatial_tokens dividing N; "
                    f"got N={N}, spatial_tokens={S}")
                Tp = N // S
                xt = normed.view(B, Tp, S, D).permute(0, 2, 1, 3).reshape(B * S, Tp, D)
                out_t, _, _ = self._run_time_only_gdn(xt, streaming=False)
                out = out_t.view(B, S, Tp, D).permute(0, 2, 1, 3).reshape(B, N, D)
                new_state = None
            else:
                out, new_state = self.layer(normed, state)
        elif self.spatial_tokens is not None:
            # ── Spatial-only attention in processor NoPE blocks ──
            S = self.spatial_tokens
            T = N // S
            assert N == T * S, (
                f"Token count {N} not divisible by spatial_tokens {S}")
            normed = normed.view(B, T, S, D).reshape(B * T, S, D)
            out = self.layer(normed)               # O(S²) per frame
            out = out.view(B, T, S, D).reshape(B, N, D)
            new_state = state
        else:
            # ── Global attention fallback ──
            out = self.layer(normed, attn_mask=attn_mask)
            new_state = state
        
        x = x + self.drop_path(out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, new_state
    
    def forward_chunk(self, x, attn_mask=None, state=None, conv_state=None):
        """Streaming counterpart of forward(). Threads the GDN recurrent state
        and short-conv state across temporal chunks; attention/NoPE blocks are
        spatial-only (stateless across time). Returns (x, new_state,
        new_conv_state). Not supported for bidirectional GDN (the backward
        branch needs future frames)."""
        B, N, D = x.shape
        normed = self.norm1(x)
        new_state, new_conv_state = None, None
        
        if self.block_type == 'gdn':
            if self.bidirectional:
                raise RuntimeError(
                    "Streaming (forward_chunk) is not supported for bidirectional "
                    "GDN; use bidirectional=False for streaming inference.")
            if self.gdn_temporal_only:
                S = self.spatial_tokens
                assert S is not None and N % S == 0, (
                    f"gdn_temporal_only needs spatial_tokens dividing N; "
                    f"got N={N}, spatial_tokens={S}")
                Tp = N // S
                xt = normed.view(B, Tp, S, D).permute(0, 2, 1, 3).reshape(B * S, Tp, D)
                out_t, new_state, new_conv_state = self._run_time_only_gdn(
                    xt, state, conv_state, streaming=True)
                out = out_t.view(B, S, Tp, D).permute(0, 2, 1, 3).reshape(B, N, D)
            else:
                out, new_state, new_conv_state = self.layer.forward_chunk(
                    normed, state, conv_state)
        elif self.spatial_tokens is not None:
            S = self.spatial_tokens
            T = N // S
            assert N == T * S
            nv = normed.view(B, T, S, D).reshape(B * T, S, D)
            out = self.layer(nv)
            out = out.view(B, T, S, D).reshape(B, N, D)
        else:
            out = self.layer(normed, attn_mask=attn_mask)

        x = x + self.drop_path(out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, new_state, new_conv_state
    
class NoPEGDNVideoBackbone(nn.Module):
    """
    Complete 3D NoPE + Gated DeltaNet video backbone.

    Two-stage architecture:
      Stage 1 — NoPE Encoder:  Spatial-only attention, no PE
      Stage 2 — Hybrid Processor: 3:1 GDN-to-NoPE ratio, causal-capable

    Factorized attention (default, factorized_attention=True):
      All softmax attention — both encoder and processor NoPE blocks —
      operates SPATIAL-ONLY: temporal dim folded into batch, each frame's
      H'×W' tokens attend independently.

      Scaling analysis with 224×224, tubelet (2,16,16):
        Spatial tokens per frame: S = (224/16)² = 196
        Attention cost per NoPE block: O(T' × 196² × d)  — linear in T'
        GDN cost per block:            O(T'×196 × d²)    — linear in T'
        Total per [GDN,GDN,GDN,NoPE] group: O(T'·S·d²) + O(T'·S²·d)

        At 512 frames (T'=256):  NoPE sees 196 tokens (trivial)
        Without factorization:   NoPE would see 50,176 tokens (intractable)

    Args:
        img_size:              Spatial input resolution
        num_frames:            Number of video frames
        tubelet_size:          (t, h, w) 3D patch dimensions
        in_channels:           Input channels (3 for RGB)
        encoder_dim:           Encoder embedding dimension
        encoder_depth:         Number of encoder ViT blocks
        encoder_heads:         Number of encoder attention heads
        processor_dim:         Processor embedding dimension
        processor_depth:       Number of processor hybrid blocks
        processor_heads:       Number of processor heads (GDN and NoPE)
        gdn_ratio:             GDN blocks per NoPE block (default 3 → 3:1)
        chunk_size:            GDN chunk size for WY algorithm
        channel_wise_decay:    True=KDA channel-wise g (official), False=scalar α (ablation)
        allow_neg_eigval:      True=β*2, allows negative eigenvalue in Householder
        factorized_attention:  True=spatial-only softmax (default), False=global (ablation)
        mlp_ratio:             MLP expansion ratio
        dropout:               Dropout probability
    """
    
    def __init__(self, img_size=224, num_frames=16,
                 tubelet_size=(2, 16, 16), in_channels=3,
                 encoder_dim=768, encoder_depth=6, encoder_heads=12,
                 processor_dim=768, processor_depth=12, processor_heads=8,
                 gdn_ratio=3, chunk_size=64,
                 channel_wise_decay=True,
                 allow_neg_eigval=False,
                 factorized_attention=True,
                 mlp_ratio=4.0, dropout=0.1,
                 use_grad_checkpoint=False,
                 drop_path_rate=0.0,
                 bidirectional=False,
                 flip_mode='temporal',
                 gdn_temporal_only=False,
                 decay_target_dt=None,
                 a_init_range=(1.0, 16.0)):
        super().__init__()
        self.bidirectional = bidirectional
        self.flip_mode = flip_mode
        self.gdn_temporal_only = gdn_temporal_only
        self.decay_target_dt = decay_target_dt
        self.a_init_range = a_init_range

        # Total depth for linearly increasing drop path
        total_depth = encoder_depth + processor_depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        encoder_dpr = drop_path_rate * (encoder_depth / total_depth) if total_depth > 0 else 0.0
        
        # Stage 1: NoPE Encoder 
        self.encoder = NoPEVideoEncoder(
            img_size, num_frames, tubelet_size, in_channels,
            encoder_dim, encoder_depth, encoder_heads,
            mlp_ratio, dropout, use_grad_checkpoint,
            factorized_attention=factorized_attention,
            drop_path_rate=encoder_dpr
        )
        
        grid = self.encoder.tubelet_embed.get_grid_dims()
        grid_spatial = grid['H'] * grid['W']
        nope_spatial_tokens = grid_spatial if factorized_attention else None
        gdn_spatial_tokens = grid_spatial
        
        spatial_tokens = nope_spatial_tokens

        # Dimension bridge
        self.dim_proj = nn.Linear(encoder_dim, processor_dim) \
            if encoder_dim != processor_dim else nn.Identity()
            
        # Stage 2: Hybrid Processor 
        self.processor_blocks = nn.ModuleList()
        period = gdn_ratio + 1   # block pattern repeats every (ratio+1) blocks
        for i in range(processor_depth):
            if (i + 1) % period == 0:
                bt = 'nope'
            else:
                bt = 'gdn'
            block_spatial = (nope_spatial_tokens if bt == 'nope'
                             else gdn_spatial_tokens)
            self.processor_blocks.append(
                HybridBlock(processor_dim, processor_heads,
                            block_type=bt,
                            head_dim=processor_dim // processor_heads,
                            chunk_size=chunk_size,
                            mlp_ratio=mlp_ratio, dropout=dropout,
                            channel_wise_decay=channel_wise_decay,
                            allow_neg_eigval=allow_neg_eigval,
                            spatial_tokens=block_spatial,
                            drop_path=dpr[encoder_depth + i],
                            bidirectional=bidirectional,
                            flip_mode=flip_mode,
                            gdn_temporal_only=gdn_temporal_only,
                            decay_target_dt=decay_target_dt,
                            a_init_range=a_init_range)
            )

        self.processor_norm = nn.LayerNorm(processor_dim, eps=1e-6)
        self.use_grad_checkpoint = use_grad_checkpoint

        # Expose config
        self.encoder_dim = encoder_dim
        self.processor_dim = processor_dim
        self.processor_depth = processor_depth
        self.gdn_ratio = gdn_ratio
        self.factorized_attention = factorized_attention
        self.spatial_tokens = spatial_tokens
    
    def forward(self, video: torch.Tensor,
                processor_mask: Optional[torch.Tensor] = None,
                return_encoder_features: bool = False):
        """
        Args:
            video:                 (B, C, T, H, W)
            processor_mask:        Optional causal/block-causal mask for processor
            return_encoder_features: Also return pre-processor encoder features

        Returns:
            features: (B, N, processor_dim)
            encoder_features: (B, N, encoder_dim) — only if return_encoder_features
        """
        
        # Stage 1: Encode (bidirectional, no PE)
        enc_features = self.encoder(video)
        
        # Bridge dimensions
        x = self.dim_proj(enc_features)
        
        # Stage 2: Hybrid process 
        states = [None] * len(self.processor_blocks)
        for i, block in enumerate(self.processor_blocks):
            if self.use_grad_checkpoint and self.training:
                x, states[i] = torch.utils.checkpoint.checkpoint(
                    block, x, processor_mask, states[i],
                    use_reentrant=False)
            else:
                x, states[i] = block(x, attn_mask=processor_mask,
                                     state=states[i])

        x = self.processor_norm(x)

        if return_encoder_features:
            return x, enc_features
        return x
    
    def forward_chunk(self, video_chunk, states=None):
        enc = self.encoder(video_chunk)
        x = self.dim_proj(enc)
        if states is None:
            states = [(None, None)] * len(self.processor_blocks)
        new_states = []
        for i, block in enumerate(self.processor_blocks):
            st, cst = states[i] if states[i] is not None else (None, None)
            x, ns, ncs = block.forward_chunk(x, state=st, conv_state=cst)
            new_states.append((ns, ncs))
        x = self.processor_norm(x)
        return x, new_states
    
    def get_block_types(self) -> List[str]:
        """Return the type of each processor block."""
        return [b.block_type for b in self.processor_blocks]

    def count_blocks_by_type(self) -> Dict[str, int]:
        types = self.get_block_types()
        return {'gdn': types.count('gdn'), 'nope': types.count('nope')}

    def get_gdn_states(self, video, processor_mask=None):
        """Run forward and return all GDN recurrent states for inspection."""
        enc = self.encoder(video)
        x = self.dim_proj(enc)
        all_states = {}
        states = [None] * len(self.processor_blocks)
        for i, block in enumerate(self.processor_blocks):
            x, states[i] = block(x, attn_mask=processor_mask, state=states[i])
            if block.block_type == 'gdn' and states[i] is not None:
                all_states[f'block_{i}'] = states[i].detach()
        return all_states