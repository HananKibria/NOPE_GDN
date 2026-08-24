import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoClassificationHead(nn.Module):
    """
    Global average pooling + linear classifier.

    Pools over all tokens (spatial + temporal) to produce
    a single video-level prediction.
    """

    def __init__(self, embed_dim: int, num_classes: int,
                 dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) token features -> logits: (B, num_classes)."""
        x = self.norm(x)
        x = x.mean(dim=1)           # global average pooling over tokens
        x = self.dropout(x)
        return self.fc(x)


class TemporalPoolingHead(nn.Module):
    """
    Temporal-aware classification head for SSv2.

    Pools spatial tokens per frame, applies learned temporal attention (a
    single query over T' frame-level features) and classifies the weighted
    summary. Adds ~2*D² + D params (~1.2M at D=768). Works at any frame count.
    """

    def __init__(self, embed_dim: int, num_classes: int,
                 num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Learned query for temporal attention (single token)
        self.temporal_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Temporal cross-attention: query attends over T' frame features
        self.t_proj_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.t_proj_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.t_proj_out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.t_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor, S: int = 196) -> torch.Tensor:
        """x: (B, T'*S, D) spatiotemporal tokens, S spatial tokens per frame.
        Returns logits: (B, num_classes)."""
        B, N, D = x.shape
        T = N // S

        # Step 1: Spatial mean pooling per frame → (B, T', D)
        x = x.view(B, T, S, D).mean(dim=2)  # (B, T', D)

        # Step 2: Temporal cross-attention
        q = self.temporal_query.expand(B, -1, -1)  # (B, 1, D)
        k = self.t_proj_k(x)                        # (B, T', D)
        v = self.t_proj_v(x)                        # (B, T', D)

        H, d = self.num_heads, self.head_dim
        q = q.view(B, 1, H, d).transpose(1, 2)     # (B, H, 1, d)
        k = k.view(B, T, H, d).transpose(1, 2)     # (B, H, T', d)
        v = v.view(B, T, H, d).transpose(1, 2)     # (B, H, T', d)

        out = F.scaled_dot_product_attention(q, k, v)  # (B, H, 1, d)
        out = out.transpose(1, 2).reshape(B, D)         # (B, D)
        out = self.t_proj_out(out)

        # Residual with mean-pooled features (fallback)
        out = out + x.mean(dim=1)
        out = self.t_norm(out)

        # Step 3: Classify
        out = self.dropout(out)
        return self.fc(out)
