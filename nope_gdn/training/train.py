import json
import time
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler

from ..config import get_config
from ..data.debug import make_debug_loader
from ..data.diving48 import build_diving48_dataloaders
from ..data.kinetics import build_kinetics_dataloaders
from ..data.ssv2 import HAS_AV, build_dataloaders
from ..models.factory import build_model, count_params
from ..models.pretrained import (PRETRAINED_MAP, VIDEOMAE_MAP,
                                 load_pretrained_vit,
                                 load_videomae_into_rope,
                                 load_videomae_pretrained,
                                 load_videomae_pretrained_trecvit)
from .engine import train_one_epoch, validate, validate_tta
from .optimizer import boost_proc_head_stochastic_reg, build_llrd_optimizer
from .utils import (CosineWarmupScheduler, Logger, MixupCutmix, ModelEMA,
                    load_checkpoint, save_checkpoint, set_seed)


_KEEP = object()  # sentinel: leave the ModelConfig value unchanged


def run_training(variant="nope_gdn", size="small", epochs=None,
                 batch_size=None, lr=None, debug=True, device=None,
                 data_root=None, num_workers=8,
                 pretrained=False,
                 resume=None,
                 init_weights=None,
                 use_ema_weights=False,
                 gdn_temporal_only=None,
                 decay_target_dt=_KEEP,
                 a_init_range=_KEEP,
                 processor_heads=None, allow_neg_eigval=None,
                 proc_head_wd=None, proc_head_lr_mult=1.0,
                 proc_drop_path=None, proc_dropout=None, head_dropout=None):
    """
    Main training entry point (Jupyter-friendly, no argparse).

    Args:
        variant:    'nope_gdn', 'rope', or 'learned_pe'
        size:       'tiny', 'small', or 'base'
        epochs:     Override number of epochs
        batch_size: Override batch size
        lr:         Override learning rate
        debug:      If True, use synthetic data (no real dataset needed)
        device:     Override device ('cuda', 'cpu', 'mps')
        data_root:  Path to SSv2 data root (required when debug=False)
                    e.g. './data/ssv2' or '/Volumes/Drive/Data/ssv2'
        num_workers: Dataloader workers (default 8)
        pretrained:  If True, init encoder from pretrained DeiT (requires timm)
        resume:      Path to checkpoint .pt file to resume training from
        init_weights: Path to checkpoint .pt to load model weights only (fresh optimizer/scheduler)
        use_ema_weights: If True, load EMA shadow weights instead of raw model weights from init_weights
        gdn_temporal_only: Override GDN scan mode. True=time-only per spatial location,
                    False=flat T'xS raster, None=use ModelConfig default.
        decay_target_dt: Override KDA decay init. float (e.g. 0.05)=long-memory dt_bias,
                    None=zeros (short memory). Omit to use ModelConfig default.
        a_init_range: Override KDA A_log init range (a_lo, a_hi). Omit to use config default.
    """
    # ---- Config ----
    cfg = get_config(variant=variant, size=size)

    if epochs:
        cfg.train.epochs = epochs
    if batch_size:
        cfg.train.batch_size = batch_size
    if lr:
        cfg.train.lr = lr
    if device:
        cfg.train.device = device
    if data_root:
        cfg.data.data_root = data_root
    if resume:
        cfg.train.resume = resume
    if gdn_temporal_only is not None:
        cfg.model.gdn_temporal_only = gdn_temporal_only
    if processor_heads is not None:
        cfg.model.processor_heads = processor_heads
    if allow_neg_eigval is not None:
        cfg.model.allow_neg_eigval = allow_neg_eigval
    if decay_target_dt is not _KEEP:
        cfg.model.decay_target_dt = decay_target_dt
    if a_init_range is not _KEEP:
        cfg.model.a_init_range = a_init_range

    # Auto-detect device
    if cfg.train.device == "cuda" and not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            cfg.train.device = "mps"
            cfg.train.amp = False
        else:
            cfg.train.device = "cpu"
            cfg.train.amp = False

    # ---- Output dir ----
    exp_dir = Path(cfg.train.output_dir) / cfg.train.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(str(exp_dir / "train.log"))
    logger.log("=" * 60)
    logger.log(f"SSv2 Training — {variant} ({size})")
    logger.log(f"Output: {exp_dir}")
    logger.log(f"Device: {cfg.train.device}")
    logger.log("=" * 60)

    # ---- Seed ----
    set_seed(cfg.train.seed)

    # ---- Data ----
    logger.log("Loading dataset...")
    if debug:
        logger.log("  DEBUG MODE: Using synthetic data")
        train_loader = make_debug_loader(cfg, is_train=True)
        val_loader = make_debug_loader(cfg, is_train=False)
    else:
        if not data_root and not cfg.data.data_root:
            raise ValueError(
                "data_root is required when debug=False.\n"
                "Usage: run_training(debug=False, data_root='./data/ssv2')\n"
                "\n"
                "Expected structure inside data_root:\n"
                "  20bn-something-something-v2/  (video files)\n"
                "  annotations/  (labels.json, train.json, validation.json)")
        if not HAS_AV:
            raise RuntimeError(
                "PyAV not installed. Install: pip install av")
        logger.log(f"  Data root: {cfg.data.data_root}")
        train_loader, val_loader = build_dataloaders(
            cfg.data, cfg.train.batch_size, num_workers)

    # ---- Model ----
    logger.log(f"Building model: {variant}")
    cfg.model.drop_path_rate = cfg.train.drop_path_rate
    cfg.model.use_grad_checkpoint = cfg.train.use_grad_checkpoint
    model = build_model(variant, cfg.model)
    if any(v is not None for v in (proc_drop_path, proc_dropout, head_dropout)):
        boost_proc_head_stochastic_reg(model, proc_drop_path=proc_drop_path,
                                       proc_dropout=proc_dropout, head_dropout=head_dropout)
    # ---- Pretrained init ----
    if pretrained:
        if pretrained == "videomae":
            vmae_ckpt = VIDEOMAE_MAP.get(size)
            if vmae_ckpt is None:
                logger.log(f"  No VideoMAE checkpoint for size={size}, falling back to DeiT")
                ckpt_name = PRETRAINED_MAP.get(size, "deit_small_patch16_224")
                logger.log(f"  Loading pretrained: {ckpt_name}")
                model = load_pretrained_vit(model, ckpt_name)
            else:
                logger.log(f"  Loading VideoMAE pretrained: {vmae_ckpt}")
                if variant == "trecvit":
                    model = load_videomae_pretrained_trecvit(model, vmae_ckpt)
                elif variant in ("rope", "axial_rope", "mixed_rope"):
                    model = load_videomae_into_rope(model, vmae_ckpt)
                else:
                    model = load_videomae_pretrained(model, vmae_ckpt)
        else:
            ckpt_name = PRETRAINED_MAP.get(size, "deit_small_patch16_224")
            logger.log(f"  Loading pretrained: {ckpt_name}")
            model = load_pretrained_vit(model, ckpt_name)

    # ---- Load weights only (no optimizer/scheduler) ----
    if init_weights:
        logger.log(f"  Loading model weights from: {init_weights}")
        ckpt = torch.load(init_weights, map_location="cpu", weights_only=False)
        if use_ema_weights and ckpt.get("ema_state"):
            src_sd = ckpt["ema_state"]["shadow"]
            logger.log(f"  Using EMA shadow weights")
        else:
            src_sd = ckpt["model_state"]
            if use_ema_weights:
                logger.log("  \u26a0\ufe0f No EMA state found, falling back to model weights")

        # Partial load: match by name AND shape (skips mismatched head, etc.)
        model_sd = model.state_dict()
        loaded, skipped = 0, []
        for name, param in src_sd.items():
            if name in model_sd and model_sd[name].shape == param.shape:
                model_sd[name] = param
                loaded += 1
            elif name in model_sd:
                skipped.append(f"{name} ({list(param.shape)}\u2192{list(model_sd[name].shape)})")
        model.load_state_dict(model_sd)
        logger.log(f"  \u2705 Loaded {loaded} params (epoch {ckpt.get('epoch', '?')}, "
                   f"acc={ckpt.get('best_acc', '?')}%)")
        if skipped:
            logger.log(f"  \u26a0\ufe0f Skipped {len(skipped)} shape-mismatched params (new head?):")
            for s in skipped:
                logger.log(f"      {s}")

    model = model.to(cfg.train.device)

    params = count_params(model)
    logger.log(f"  Parameters: {params['total']/1e6:.2f}M "
               f"(trainable: {params['trainable']/1e6:.2f}M)")

    if cfg.train.compile_model and hasattr(torch, "compile"):
        logger.log("  Compiling model with torch.compile...")
        model = torch.compile(model)

    # ---- Optimizer with Layer-wise LR Decay ----
    # Assigns lower LR to early (pretrained) encoder layers, higher LR to
    # later layers and the randomly-initialized GDN processor + head.
    # This prevents destroying pretrained features while letting new layers learn.

    def get_layer_id(name, encoder_depth=cfg.model.encoder_depth,
                     processor_depth=cfg.model.processor_depth):
        """Assign a layer index to each parameter for LR scaling.
        Layer 0 = embedding (lowest LR), last layer = head (highest LR).

        Handles all model architectures:
          NoPE+GDN: backbone.encoder.blocks.{i} (encoder) + processor_blocks (full LR)
          RoPE:     blocks.{i} flat — blocks 0..encoder_depth-1 are pretrained
                    (layer-wise decayed), blocks encoder_depth..total get full LR
          LearnedPE: same flat blocks.{i} naming as RoPE
          TRecViT:  encoder.spatial.{i} (VideoMAE-transferred, decayed) +
                    encoder.temporal.{i} (LRU, random init, full LR);
                    tokenizer is the embedding layer; pre_logits + cls_head are
                    random-init heads at full LR.
        """
        total_depth = encoder_depth + processor_depth
        if 'tubelet_embed' in name:
            return 0
        elif 'backbone.encoder.blocks.' in name:
            block_id = int(name.split('backbone.encoder.blocks.')[1].split('.')[0])
            return block_id + 1
        elif 'backbone.encoder.norm' in name:
            return encoder_depth
        elif name.startswith('blocks.'):
            # RoPE / LearnedPE: flat blocks.{0..total-1}.
            # blocks 0..encoder_depth-1 = pretrained (layer 1..encoder_depth),
            # blocks encoder_depth..end = random init (full LR).
            block_id = int(name.split('blocks.')[1].split('.')[0])
            if block_id < encoder_depth:
                return block_id + 1
            else:
                return encoder_depth + 1
        elif name.startswith('tokenizer.'):
            # TRecViT tokenizer (proj + cls_token + pos_embed) = embedding layer
            return 0
        elif name.startswith('encoder.spatial.'):
            # TRecViT spatial blocks (VideoMAE-transferred): per-block decay
            block_id = int(name.split('encoder.spatial.')[1].split('.')[0])
            if block_id < encoder_depth:
                return block_id + 1
            else:
                return encoder_depth + 1
        elif name.startswith('encoder.final_norm'):
            # TRecViT final norm of pretrained spatial encoder
            return encoder_depth
        elif name.startswith('encoder.temporal.'):
            # TRecViT LRU temporal blocks: random init, full LR
            return encoder_depth + 1
        elif (name.startswith('norm.') or name.startswith('head.') or
              name.startswith('cls_head.') or name.startswith('pre_logits.') or
              name.startswith('pos_embed')):
            # Final norm, classification head, learned PE, trecvit head: full LR
            return encoder_depth + 1
        else:  # dim_proj, processor_blocks, processor_norm, rope buffers, etc.
            return encoder_depth + 1

    num_layers = cfg.model.encoder_depth + 2  # processor + head get full LR (not decayed)
    layer_decay = cfg.train.layer_decay
    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No weight decay for bias, norm, and special params
        # 'rope.theta_' covers learnable per-head frequencies in mixed_rope
        # (analogous to positional embeddings — should not be weight-decayed).
        no_wd = (param.dim() <= 1 or 'bias' in name or 'norm' in name
                 or 'rope.theta_' in name
                 or getattr(param, '_no_weight_decay', False))
        wd = 0.0 if no_wd else cfg.train.weight_decay

        layer_id = get_layer_id(name)
        lr_scale = layer_decay ** (num_layers - 1 - layer_id)
        group_key = f'layer_{layer_id}_wd_{wd}'

        if group_key not in param_groups:
            param_groups[group_key] = {
                'params': [],
                'weight_decay': wd,
                'lr': cfg.train.lr * lr_scale,
                'lr_scale': lr_scale,
            }
        param_groups[group_key]['params'].append(param)

    if proc_head_wd is not None or proc_head_lr_mult != 1.0:
        optimizer = build_llrd_optimizer(model, cfg, proc_head_wd=proc_head_wd,
                                         proc_head_lr_mult=proc_head_lr_mult)
    else:
        optimizer = torch.optim.AdamW(
            list(param_groups.values()),
            lr=cfg.train.lr, betas=cfg.train.betas)

    logger.log(f"  Optimizer: AdamW (base_lr={cfg.train.lr}, "
               f"layer_decay={layer_decay}, wd={cfg.train.weight_decay})")
    logger.log(f"  Layer-wise LR: embedding={cfg.train.lr * layer_decay**(num_layers-1):.2e} "
               f"-> head={cfg.train.lr:.2e} ({num_layers} layers)")
    logger.log(f"  Param groups: {len(param_groups)}")

    # ---- Scheduler ----
    steps_per_epoch = len(train_loader) // cfg.train.grad_accum_steps
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=cfg.train.warmup_epochs,
        total_epochs=cfg.train.epochs,
        warmup_lr=cfg.train.warmup_lr,
        base_lr=cfg.train.lr,
        min_lr=cfg.train.min_lr,
        steps_per_epoch=steps_per_epoch,
    )

    # ---- AMP scaler ----
    scaler = GradScaler(enabled=cfg.train.amp and torch.cuda.is_available())

    # ---- Mixup/CutMix ----
    mixup_fn = MixupCutmix(
        mixup_alpha=cfg.data.mixup_alpha,
        cutmix_alpha=cfg.data.cutmix_alpha,
        prob=cfg.data.mixup_prob,
        num_classes=cfg.data.num_classes,
        label_smoothing=cfg.data.label_smoothing,
    ) if cfg.data.mixup_prob > 0 else None

    # ---- EMA ----
    ema = None
    if cfg.train.ema_decay > 0:
        ema = ModelEMA(model, decay=cfg.train.ema_decay)
        logger.log(f"  EMA enabled (decay={cfg.train.ema_decay})")

    # ---- Resume ----
    start_epoch = 0
    best_acc = 0.0
    metrics_history = []

    if cfg.train.resume:
        logger.log(f"Resuming from {cfg.train.resume}")
        start_epoch, best_acc, metrics_history = load_checkpoint(
            cfg.train.resume, model, optimizer, scheduler, scaler, ema=ema)
        start_epoch += 1
        logger.log(f"  Resumed at epoch {start_epoch}, best_acc={best_acc:.2f}%")

    # ---- Training ----
    logger.log(f"\nStarting training for {cfg.train.epochs} epochs...")
    logger.log(f"  Effective batch size: "
               f"{cfg.train.batch_size * cfg.train.grad_accum_steps}")

    # ---- Encoder freeze helper (variant-aware) ----
    def freeze_encoder(model):
        """Freeze pretrained components.

        nope_gdn:        freezes backbone.encoder.* + backbone.dim_proj
        rope/learned_pe: freezes blocks.0..encoder_depth-1 + tubelet_embed
                         (random-init blocks encoder_depth..end + final norm +
                         head stay trainable, matching rope_variableLength.ipynb)
        trecvit:         freezes tokenizer.proj, tokenizer.pos_embed,
                         encoder.spatial.*, encoder.final_norm
                         (LRU temporal layers stay trainable — they are random
                         init and have no pretrained weights, so freezing them
                         would kill all temporal learning during the freeze phase)
        """
        m = model._orig_mod if hasattr(model, '_orig_mod') else model
        encoder_depth = cfg.model.encoder_depth
        if cfg.train.model_variant == "trecvit":
            freeze_prefixes = (
                "tokenizer.proj.",
                "tokenizer.pos_embed",
                "encoder.spatial.",
                "encoder.final_norm.",
            )
            for name, param in m.named_parameters():
                if any(name.startswith(p) for p in freeze_prefixes):
                    param.requires_grad = False
        else:
            for name, param in m.named_parameters():
                # NoPE+GDN: freeze whole nested encoder
                if 'backbone.encoder' in name or 'backbone.dim_proj' in name:
                    param.requires_grad = False
                # RoPE / LearnedPE: freeze pretrained blocks 0..encoder_depth-1
                elif name.startswith('blocks.'):
                    block_id = int(name.split('blocks.')[1].split('.')[0])
                    if block_id < encoder_depth:
                        param.requires_grad = False
                # RoPE / LearnedPE: tubelet embedding is VideoMAE-transferred
                elif name.startswith('tubelet_embed.'):
                    param.requires_grad = False
        frozen = sum(p.numel() for p in m.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        logger.log(f"  ❄️  Encoder frozen ({frozen/1e6:.1f}M frozen, "
                   f"{trainable/1e6:.1f}M trainable)")

    def unfreeze_encoder(model):
        """Unfreeze encoder for end-to-end finetuning."""
        m = model._orig_mod if hasattr(model, '_orig_mod') else model
        for param in m.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        logger.log(f"  \U0001f525 Encoder unfrozen ({trainable/1e6:.1f}M trainable)")

    encoder_frozen = False
    if cfg.train.freeze_encoder_epochs > 0 and start_epoch < cfg.train.freeze_encoder_epochs:
        freeze_encoder(model)
        encoder_frozen = True

    total_start = time.time()

    for epoch in range(start_epoch, cfg.train.epochs):
        # Unfreeze encoder at the right epoch
        if encoder_frozen and epoch >= cfg.train.freeze_encoder_epochs:
            unfreeze_encoder(model)
            encoder_frozen = False

        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            mixup_fn, epoch, cfg, logger, ema=ema)

        if (epoch + 1) % cfg.train.eval_interval == 0 or epoch == cfg.train.epochs - 1:
            if ema is not None:
                ema.apply_shadow(model)
            val_metrics = validate(model, val_loader, cfg, logger)
            if ema is not None:
                ema.restore(model)
        else:
            val_metrics = {"val_loss": 0, "val_acc1": 0, "val_acc5": 0}

        epoch_metrics = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "time": time.time() - epoch_start,
        }
        metrics_history.append(epoch_metrics)

        is_best = val_metrics["val_acc1"] > best_acc
        if is_best:
            best_acc = val_metrics["val_acc1"]
            save_checkpoint(model, optimizer, scheduler, scaler,
                            epoch, best_acc, metrics_history,
                            str(exp_dir / "best_model.pt"), ema=ema)
            logger.log(f"  ★ New best: {best_acc:.2f}%")

        if (epoch + 1) % cfg.train.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, scaler,
                            epoch, best_acc, metrics_history,
                            str(exp_dir / f"ckpt_epoch_{epoch+1}.pt"), ema=ema)

        elapsed = time.time() - total_start
        logger.log(
            f"  Epoch {epoch} done in {epoch_metrics['time']/60:.1f}min  |  "
            f"Total: {elapsed/3600:.1f}h  |  "
            f"Best: {best_acc:.2f}%"
        )

    # ---- TTA evaluation on best model ----
    logger.log("")
    logger.log("=" * 60)
    logger.log("Running TTA evaluation (2 clips x 3 crops)...")
    if ema is not None:
        ema.apply_shadow(model)
    tta_metrics = validate_tta(model, cfg, logger)
    if ema is not None:
        ema.restore(model)
    logger.log(f"  TTA Result -- Acc@1: {tta_metrics['tta_acc1']:.2f}%  "
               f"Acc@5: {tta_metrics['tta_acc5']:.2f}%")
    logger.log("=" * 60)

    # ---- Save final ----
    save_checkpoint(model, optimizer, scheduler, scaler,
                    cfg.train.epochs - 1, best_acc, metrics_history,
                    str(exp_dir / "final_model.pt"), ema=ema)

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

    logger.log(f"\n{'='*60}")
    logger.log(f"Training complete!")
    logger.log(f"  Best Val Acc@1: {best_acc:.2f}%")
    logger.log(f"  Total time: {(time.time() - total_start)/3600:.2f}h")
    logger.log(f"  Outputs: {exp_dir}")
    logger.log(f"{'='*60}")
    logger.close()

    return model, metrics_history


def run_training_diving48(variant="nope_gdn", size="base",
                          epochs=40, batch_size=16, lr=3e-4,
                          num_frames=32,
                          data_root="/content/diving48",
                          num_workers=4,
                          pretrained="videomae",
                          init_weights=None, use_ema_weights=False,
                          resume=None, device="cuda",
                          video_dir="videos",
                          anno_dir="annotations",
                          # --- Training-dynamics knobs (override if needed) ---
                          freeze_encoder=None,
                          freeze_encoder_epochs=None,
                          layer_decay=None,
                          warmup_epochs=None,
                          ema_decay=None,
                          # --- Regularization knobs (default: inherit SSv2 config) ---
                          mixup_alpha=None,
                          cutmix_alpha=None,
                          mixup_prob=None,
                          label_smoothing=None,
                          color_jitter=None,
                          reprob=None,
                          rand_augment=None,
                          drop_path_rate=None,
                          head_dropout=None):
    """
    Train on Diving48 by reusing the existing run_training() pipeline and
    monkey-patching `build_dataloaders` + the num_classes config.

    All regularization (mixup, cutmix, RandAugment, label smoothing, random
    erasing, drop_path, head dropout) is INHERITED from the SSv2 config
    defaults, which are appropriate for fine-tuning a pretrained model
    on a small dataset (Diving48 = 15k train samples).

    Notes on init_weights:
        - If init_weights points to an SSv2-trained NoPE+GDN or RoPE checkpoint,
          the classification head (174 classes) is automatically skipped
          via shape-matching, and the new 48-class head stays random-init.
        - Encoder + GDN weights transfer cleanly.
        - Set use_ema_weights=True to use the EMA shadow copy (recommended).

    Notes on training-dynamics when init_weights is set:
        - freeze_encoder=False, freeze_encoder_epochs=0 (full fine-tune)
        - layer_decay=0.85 (same as SSv2 default)
        - warmup_epochs=2 (short warmup — features already useful)
        - ema_decay=0.999 (warmer than 0.9999, faster EMA warmup)

    Notes on horizontal flip:
        - Diving48 is DIRECTION-SENSITIVE (forward vs backward dive).
        - Horizontal flip stays DISABLED (cfg.data.use_hflip=False, SSv2 default).
    """
    global build_dataloaders   # we'll swap this temporarily
    original_build_dataloaders = build_dataloaders

    def _diving48_build_dataloaders(data_cfg, bs, nw):
        return build_diving48_dataloaders(
            data_root=data_root,
            batch_size=bs,
            num_frames=data_cfg.num_frames,
            img_size=data_cfg.img_size,
            num_workers=nw,
            pin_memory=data_cfg.pin_memory,
            video_dir=video_dir,
            anno_dir=anno_dir,
            # NEW — pull regularization from cfg
            color_jitter=data_cfg.color_jitter,
            reprob=data_cfg.reprob,
            rand_augment=data_cfg.rand_augment,
            use_hflip=True,   # hardcoded True for Diving48 (overrides SSv2 default of False)
        )

    # ---- Override num_classes and Diving48-specific recipe BEFORE run_training builds the model ----
    global get_config
    original_get_config = get_config

    def _diving48_get_config(variant="nope_gdn", size="small"):
        cfg = original_get_config(variant=variant, size=size)

        # Dataset-specific overrides (always applied for Diving48)
        cfg.model.num_classes = 48
        cfg.data.num_classes = 48
        cfg.data.num_frames = num_frames
        cfg.model.num_frames = num_frames
        cfg.train.experiment_name = f"{variant}_{size}_diving48"

        # Regularization: INHERIT FROM SSv2 CONFIG DEFAULTS
        # (only override if caller explicitly passes a value)
        # DataConfig defaults (already set by get_config):
        #   rand_augment    = "rand-m7-mstd0.5-inc1"
        #   mixup_alpha     = 0.8
        #   cutmix_alpha    = 1.0
        #   mixup_prob      = 0.5
        #   label_smoothing = 0.2
        #   color_jitter    = 0.4
        #   reprob          = 0.4
        #   use_hflip       = False  (Diving48 is direction-sensitive, keep False)
        # ModelConfig defaults:
        #   head_dropout    = 0.3
        #   dropout         = 0.2
        # TrainConfig defaults:
        #   drop_path_rate  = 0.2
        #   weight_decay    = 0.2
        #   layer_decay     = 0.85

        # Apply caller overrides if provided
        if mixup_alpha is not None:
            cfg.data.mixup_alpha = mixup_alpha
        if cutmix_alpha is not None:
            cfg.data.cutmix_alpha = cutmix_alpha
        if mixup_prob is not None:
            cfg.data.mixup_prob = mixup_prob
        if label_smoothing is not None:
            cfg.data.label_smoothing = label_smoothing
        if color_jitter is not None:
            cfg.data.color_jitter = color_jitter
        if reprob is not None:
            cfg.data.reprob = reprob
        if rand_augment is not None:
            cfg.data.rand_augment = rand_augment
        if drop_path_rate is not None:
            cfg.train.drop_path_rate = drop_path_rate
        if head_dropout is not None:
            cfg.model.head_dropout = head_dropout

        # Training-dynamics: warmup + scheduler
        cfg.train.warmup_epochs = min(3, (epochs or cfg.train.epochs) // 10)

        # Apply caller overrides
        if layer_decay is not None:
            cfg.train.layer_decay = layer_decay
        if warmup_epochs is not None:
            cfg.train.warmup_epochs = warmup_epochs
        if ema_decay is not None:
            cfg.train.ema_decay = ema_decay
        if freeze_encoder is not None:
            cfg.train.freeze_encoder = freeze_encoder
        if freeze_encoder_epochs is not None:
            cfg.train.freeze_encoder_epochs = freeze_encoder_epochs

        # Auto-adjust TRAINING DYNAMICS when init_weights is set
        # (regularization is NOT adjusted — it stays at SSv2 defaults)
        if init_weights is not None:
            # Unfreeze encoder (warm-start features are already task-adapted)
            if freeze_encoder is None:
                cfg.train.freeze_encoder = False
            if freeze_encoder_epochs is None:
                cfg.train.freeze_encoder_epochs = 0
            # Shorter warmup — features already useful
            if warmup_epochs is None:
                cfg.train.warmup_epochs = 2
            # Warmer EMA — reaches transferred weights faster
            if ema_decay is None:
                cfg.train.ema_decay = 0.999
            # layer_decay stays at 0.85 (SSv2 default)

        return cfg

    try:
        # Install patches
        build_dataloaders = _diving48_build_dataloaders
        get_config = _diving48_get_config

        # Delegate to the real run_training with our monkey-patches in place
        model, history = run_training(
            variant=variant,
            size=size,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            debug=False,
            device=device,
            data_root=data_root,
            num_workers=num_workers,
            pretrained=pretrained,
            resume=resume,
            init_weights=init_weights,
            use_ema_weights=use_ema_weights,
        )
        return model, history
    finally:
        # Restore originals so SSv2 still works afterwards
        build_dataloaders = original_build_dataloaders
        get_config = original_get_config


def run_training_kinetics400(variant="nope_gdn", size="base",
                             epochs=30, batch_size=32, lr=5e-4,
                             num_frames=32, stride=2, backend="auto",
                             num_workers=8, pretrained="videomae",
                             init_weights=None, use_ema_weights=False,
                             resume=None, device="cuda",
                             data_root="/content/kinetics400", gdn_temporal_only=True,
                             train_list=None, val_list=None):
    global build_dataloaders, get_config, validate_tta
    o_bdl, o_cfg, o_tta = build_dataloaders, get_config, validate_tta

    def _make(bs, nf, im, nw):
        return build_kinetics_dataloaders(bs, nf, im, num_workers=nw,
                                          stride=stride, backend=backend,
                                          train_list=train_list, val_list=val_list)

    def _bdl(data_cfg, bs, nw): return _make(bs, data_cfg.num_frames, data_cfg.img_size, nw)

    def _cfg(variant="nope_gdn", size="small"):
        cfg = o_cfg(variant=variant, size=size)
        cfg.model.num_classes = cfg.data.num_classes = 400
        cfg.model.num_frames  = cfg.data.num_frames  = num_frames
        cfg.data.data_root = data_root
        cfg.train.experiment_name = f"{variant}_{size}_k400"
        cfg.data.mixup_prob = 0.5
        cfg.data.label_smoothing = 0.1
        cfg.train.warmup_epochs = min(5, max(1, (epochs or cfg.train.epochs) // 6))
        cfg.train.freeze_encoder_epochs=5
        return cfg

    def _tta(model, cfg, logger, num_workers=4):
        _, val = _make(max(cfg.train.batch_size // 2, 1),
                       cfg.data.num_frames, cfg.data.img_size, num_workers)
        m = validate(model, val, cfg, logger)
        return m.get("acc1", 0.0), m.get("acc5", 0.0)

    try:
        build_dataloaders, get_config, validate_tta = _bdl, _cfg, _tta
        return run_training(
            variant=variant, size=size, epochs=epochs, batch_size=batch_size, lr=lr,
            debug=False, device=device, data_root=data_root, num_workers=num_workers,
            pretrained=pretrained, resume=resume, init_weights=init_weights,
            use_ema_weights=use_ema_weights, gdn_temporal_only=gdn_temporal_only,decay_target_dt=0.05,a_init_range=(0.5,8.0))
    finally:
        build_dataloaders, get_config, validate_tta = o_bdl, o_cfg, o_tta

