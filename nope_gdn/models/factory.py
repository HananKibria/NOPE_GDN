from typing import Dict

import torch.nn as nn

from .axial_rope import AxialMixedRoPEVideoClassifier
from .classifier import NoPEGDNClassifier
from .trecvit import TRecViTClassifier
from .videorope import RoPEVideoClassifier


def build_model(variant: str, model_cfg) -> nn.Module:
    """
    Build a model variant.

    Args:
        variant:   'nope_gdn', 'rope' (alias: 'videorope'), 'axial_rope',
                   'mixed_rope', or 'trecvit'
        model_cfg: ModelConfig dataclass

    Returns:
        model: nn.Module
    """
    total_depth = model_cfg.encoder_depth + model_cfg.processor_depth

    if variant == "nope_gdn":
        model = NoPEGDNClassifier(
            img_size=model_cfg.img_size,
            num_frames=model_cfg.num_frames,
            tubelet_size=model_cfg.tubelet_size,
            in_channels=model_cfg.in_channels,
            encoder_dim=model_cfg.encoder_dim,
            encoder_depth=model_cfg.encoder_depth,
            encoder_heads=model_cfg.encoder_heads,
            processor_dim=model_cfg.processor_dim,
            processor_depth=model_cfg.processor_depth,
            processor_heads=model_cfg.processor_heads,
            gdn_ratio=model_cfg.gdn_ratio,
            chunk_size=model_cfg.chunk_size,
            channel_wise_decay=model_cfg.channel_wise_decay,
            allow_neg_eigval=model_cfg.allow_neg_eigval,
            factorized_attention=model_cfg.factorized_attention,
            num_classes=model_cfg.num_classes,
            mlp_ratio=model_cfg.encoder_mlp_ratio,
            dropout=model_cfg.dropout,
            head_dropout=model_cfg.head_dropout,
            drop_path_rate=model_cfg.drop_path_rate,
            use_grad_checkpoint=model_cfg.use_grad_checkpoint,
            bidirectional=getattr(model_cfg, 'bidirectional', False),
            flip_mode=getattr(model_cfg, 'flip_mode', 'temporal'),
            gdn_temporal_only=getattr(model_cfg, 'gdn_temporal_only', False),
            decay_target_dt=getattr(model_cfg, 'decay_target_dt', None),
            a_init_range=getattr(model_cfg, 'a_init_range', (1.0, 16.0)),
        )

    elif variant in ("rope", "videorope"):
        model = RoPEVideoClassifier(
            img_size=model_cfg.img_size,
            num_frames=model_cfg.num_frames,
            tubelet_size=model_cfg.tubelet_size,
            in_channels=model_cfg.in_channels,
            embed_dim=model_cfg.encoder_dim,
            total_depth=total_depth,
            num_heads=model_cfg.encoder_heads,
            num_classes=model_cfg.num_classes,
            mlp_ratio=model_cfg.encoder_mlp_ratio,
            dropout=model_cfg.dropout,
            head_dropout=model_cfg.head_dropout,
        )

    elif variant in ("axial_rope", "mixed_rope"):
        rope_mode = "axial" if variant == "axial_rope" else "mixed"
        model = AxialMixedRoPEVideoClassifier(
            img_size=model_cfg.img_size,
            num_frames=model_cfg.num_frames,
            tubelet_size=model_cfg.tubelet_size,
            in_channels=model_cfg.in_channels,
            embed_dim=model_cfg.encoder_dim,
            total_depth=total_depth,
            num_heads=model_cfg.encoder_heads,
            num_classes=model_cfg.num_classes,
            mlp_ratio=model_cfg.encoder_mlp_ratio,
            dropout=model_cfg.dropout,
            head_dropout=model_cfg.head_dropout,
            rope_mode=rope_mode,
        )

    elif variant == "trecvit":
        model = TRecViTClassifier(
            img_size=model_cfg.img_size,
            num_frames=model_cfg.num_frames,
            patch_size=model_cfg.tubelet_size,
            in_channels=model_cfg.in_channels,
            width=model_cfg.encoder_dim,
            depth=model_cfg.encoder_depth,
            num_heads=model_cfg.encoder_heads,
            mlp_ratio=model_cfg.encoder_mlp_ratio,
            conv1d_temporal_width=4,
            state_multiplier=1,
            min_rad=0.5,
            rep_size=3072,
            num_classes=model_cfg.num_classes,
            dropout=model_cfg.dropout,
            head_dropout=model_cfg.head_dropout,
        )

    else:
        raise ValueError(f"Unknown variant: {variant}. "
                         f"Choose from: nope_gdn, rope, videorope, "
                         f"axial_rope, mixed_rope, trecvit")

    return model


def count_params(model: nn.Module) -> Dict[str, int]:
    """Count parameters by component."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
