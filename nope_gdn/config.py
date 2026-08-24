from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class DataConfig:
    """Dataset and loading configuration."""
    data_root: str = "./data/ssv2"
    video_dir: str = "20bn-something-something-v2"
    anno_dir: str = "annotations"
    num_classes: int = 174
    num_workers: int = 8
    pin_memory: bool = True

    # Video sampling
    num_frames: int = 64
    img_size: int = 224
    tubelet_size: Tuple[int, int, int] = (2, 16, 16)

    # Augmentation
    rand_augment: str = ""
    mixup_alpha: float = 0.5
    cutmix_alpha: float = 0.2
    mixup_prob: float = 0.3
    label_smoothing: float = 0.1
    color_jitter: float = 0.2
    reprob: float = 0.2

    # NOTE: NO horizontal flip — SSv2 is direction-sensitive
    use_hflip: bool = False


@dataclass
class ModelConfig:
    """Architecture configuration — shared across all variants."""
    in_channels: int = 3
    img_size: int = 224
    num_frames: int = 64
    tubelet_size: Tuple[int, int, int] = (2, 16, 16)

    # Encoder (Stage 1)
    encoder_dim: int = 384
    encoder_depth: int = 12
    encoder_heads: int = 6
    encoder_mlp_ratio: float = 4.0

    # Processor (Stage 2) — only used by NoPE+GDN variant
    processor_dim: int = 384
    processor_depth: int = 4
    processor_heads: int = 6
    gdn_ratio: int = 3
    chunk_size: int = 64
    channel_wise_decay: bool = True
    allow_neg_eigval: bool = False
    factorized_attention: bool = True
    bidirectional: bool = False         # Bidirectional GDN (half-H per direction)
    flip_mode: str = 'temporal'         # Bidi backward-branch axis: 'temporal' | 'spatial' | 'flat'
    gdn_temporal_only: bool = True      # GDN scans time-only per spatial location (vs flat T'xS raster)
    decay_target_dt: float = 0.05       # KDA dt_bias init: None=zeros (short mem); e.g. 0.05 = long-memory init
    a_init_range: Tuple[float, float] = (0.5, 8.0)  # KDA A_log ~ log(U[a_lo,a_hi])

    # Classification head
    num_classes: int = 174
    head_dropout: float = 0.2

    # General
    dropout: float = 0.1
    drop_path_rate: float = 0.1
    use_grad_checkpoint: bool = False


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    experiment_name: str = "nope_gdn"
    output_dir: str = "./outputs"
    seed: int = 42

    model_variant: str = "nope_gdn"

    epochs: int = 50
    batch_size: int = 16
    grad_accum_steps: int = 2

    lr: float = 5e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.999)
    layer_decay: float = 0.70

    warmup_epochs: int = 5
    warmup_lr: float = 1e-6
    scheduler: str = "cosine"
    step_decay_epochs: Tuple[int, ...] = (30, 40)
    step_decay_rate: float = 0.1

    drop_path_rate: float = 0.1
    ema_decay: float = 0.9999
    clip_grad_norm: float = 1.0
    freeze_encoder_epochs: int = 5
    use_grad_checkpoint: bool = True

    log_interval: int = 50
    eval_interval: int = 1
    save_interval: int = 10

    device: str = "cuda"
    amp: bool = True
    compile_model: bool = False

    resume: str = ""


@dataclass
class FullConfig:
    """Complete configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def get_config(variant: str = "nope_gdn", size: str = "small") -> FullConfig:
    """Get a preset configuration."""
    cfg = FullConfig()
    cfg.train.model_variant = variant
    cfg.train.experiment_name = f"{variant}_{size}"

    if size == "tiny":
        cfg.model.encoder_dim = 192
        cfg.model.encoder_depth = 6
        cfg.model.encoder_heads = 3
        cfg.model.processor_dim = 192
        cfg.model.processor_depth = 4
        cfg.model.processor_heads = 3
        cfg.train.batch_size = 32
    elif size == "small":
        cfg.model.encoder_dim = 384
        cfg.model.encoder_depth = 12
        cfg.model.encoder_heads = 6
        cfg.model.processor_dim = 384
        cfg.model.processor_depth = 4
        cfg.model.processor_heads = 6
        cfg.train.batch_size = 16
    elif size == "base":
        cfg.model.encoder_dim = 768
        cfg.model.encoder_depth = 12
        cfg.model.encoder_heads = 12
        cfg.model.processor_dim = 768
        cfg.model.processor_depth = 4
        cfg.model.processor_heads = 8
        cfg.train.batch_size = 8

    return cfg
