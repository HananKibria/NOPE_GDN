import torch
import torch.nn as nn

from .backbone import NoPEGDNVideoBackbone
from .heads import VideoClassificationHead, TemporalPoolingHead


class NoPEGDNClassifier(nn.Module):
    """
    NoPE + GDN video classifier for SSv2.

    Video → TubeletEmbed → NoPE Encoder → DimProj → GDN/NoPE Processor → Head.
    Position comes from GDN layers only (conv, decay gates, recurrent state);
    the encoder and NoPE attention blocks have ZERO positional encoding.
    """

    def __init__(self, img_size=224, num_frames=16,
                 tubelet_size=(2, 16, 16), in_channels=3,
                 encoder_dim=384, encoder_depth=12, encoder_heads=6,
                 processor_dim=384, processor_depth=4, processor_heads=6,
                 gdn_ratio=3, chunk_size=64,
                 channel_wise_decay=True, allow_neg_eigval=False,
                 factorized_attention=True,
                 num_classes=174, mlp_ratio=4.0, dropout=0.1,
                 head_dropout=0.0, drop_path_rate=0.0,
                 use_grad_checkpoint=False,
                 bidirectional=False,
                 flip_mode='temporal',
                 gdn_temporal_only=False,
                 decay_target_dt=None,
                 a_init_range=(1.0, 16.0)):
        super().__init__()

        self.backbone = NoPEGDNVideoBackbone(
            img_size=img_size, num_frames=num_frames,
            tubelet_size=tubelet_size, in_channels=in_channels,
            encoder_dim=encoder_dim, encoder_depth=encoder_depth,
            encoder_heads=encoder_heads,
            processor_dim=processor_dim, processor_depth=processor_depth,
            processor_heads=processor_heads,
            gdn_ratio=gdn_ratio, chunk_size=chunk_size,
            channel_wise_decay=channel_wise_decay,
            allow_neg_eigval=allow_neg_eigval,
            factorized_attention=factorized_attention,
            mlp_ratio=mlp_ratio, dropout=dropout,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            bidirectional=bidirectional,
            flip_mode=flip_mode,
            gdn_temporal_only=gdn_temporal_only,
            decay_target_dt=decay_target_dt,
            a_init_range=a_init_range,
        )

        # spatial_tokens kept for the temporal head and downstream inspection
        self.spatial_tokens = (img_size // tubelet_size[1]) * (img_size // tubelet_size[2])
        # head; -F (flat raster scan) -> temporal-attention head
        self.temporal_head = not gdn_temporal_only
        if self.temporal_head:
            self.head = TemporalPoolingHead(
                processor_dim, num_classes,
                num_heads=processor_heads, dropout=head_dropout)
        else:
            self.head = VideoClassificationHead(
                processor_dim, num_classes, dropout=head_dropout)

    def _apply_head(self, features):
        if self.temporal_head:
            return self.head(features, S=self.spatial_tokens)
        return self.head(features)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        features = self.backbone(video)
        return self._apply_head(features)

    def forward_chunk(self, video_chunk, states=None):
        """Streaming classification over a temporal chunk. Pass the states
        returned by the previous chunk; returns (logits_for_chunk, new_states)
        with a per-chunk pooled prediction, mirroring
        TRecViTClassifier.forward_chunk."""
        features, new_states = self.backbone.forward_chunk(video_chunk, states)
        return self._apply_head(features), new_states
