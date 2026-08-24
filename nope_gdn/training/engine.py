import time

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..data.ssv2 import SSv2DatasetTTA
from .utils import AverageMeter, accuracy


def train_one_epoch(model, loader, optimizer, scheduler, scaler,
                    mixup_fn, epoch, cfg, logger, ema=None):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    use_amp = cfg.train.amp and torch.cuda.is_available()
    num_batches = len(loader)
    end = time.time()

    for step, (videos, targets) in enumerate(loader):
        data_time.update(time.time() - end)

        videos = videos.to(cfg.train.device, non_blocking=True)
        targets = targets.to(cfg.train.device, non_blocking=True)

        # Apply mixup/cutmix
        if mixup_fn is not None:
            videos, targets_mixed = mixup_fn(videos, targets)
            use_soft_labels = True
        else:
            targets_mixed = targets
            use_soft_labels = False

        # Forward pass with AMP
        with autocast(enabled=use_amp):
            logits = model(videos)

            if use_soft_labels:
                loss = F.cross_entropy(logits, targets_mixed, label_smoothing=0.0)
            else:
                loss = F.cross_entropy(logits, targets, label_smoothing=cfg.data.label_smoothing)

        # Backward pass with gradient accumulation
        loss_scaled = loss / cfg.train.grad_accum_steps

        # Skip NaN losses to prevent training collapse
        if torch.isnan(loss_scaled) or torch.isinf(loss_scaled):
            optimizer.zero_grad(set_to_none=True)
            if step % cfg.train.log_interval == 0:
                logger.log(f"  \u26a0\ufe0f NaN/Inf loss at step {step}, skipping")
            continue

        if use_amp:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        if (step + 1) % cfg.train.grad_accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)

            if cfg.train.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.train.clip_grad_norm)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            lr = scheduler.step()

            # Update EMA
            if ema is not None:
                ema.update(model)

        # Metrics
        with torch.no_grad():
            acc1, = accuracy(logits, targets, topk=(1,))

        B = videos.shape[0]
        loss_meter.update(loss.item(), B)
        acc1_meter.update(acc1.item(), B)
        batch_time.update(time.time() - end)
        end = time.time()

        # Logging
        if (step + 1) % cfg.train.log_interval == 0 or step == 0:
            lr_current = optimizer.param_groups[0]['lr']
            eta = batch_time.avg * (num_batches - step - 1)
            logger.log(
                f"  Epoch [{epoch}][{step+1}/{num_batches}]  "
                f"Loss: {loss_meter.avg:.4f}  "
                f"Acc@1: {acc1_meter.avg:.2f}%  "
                f"LR: {lr_current:.2e}  "
                f"Data: {data_time.avg:.3f}s  "
                f"Batch: {batch_time.avg:.3f}s  "
                f"ETA: {eta/60:.1f}min"
            )

    return {
        "train_loss": loss_meter.avg,
        "train_acc1": acc1_meter.avg,
        "lr": optimizer.param_groups[0]['lr'],
    }


@torch.no_grad()
def validate(model, loader, cfg, logger):
    """Evaluate on validation set."""
    model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    use_amp = cfg.train.amp and torch.cuda.is_available()

    for videos, targets in loader:
        videos = videos.to(cfg.train.device, non_blocking=True)
        targets = targets.to(cfg.train.device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(videos)
            loss = F.cross_entropy(logits, targets)

        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))

        B = videos.shape[0]
        loss_meter.update(loss.item(), B)
        acc1_meter.update(acc1.item(), B)
        acc5_meter.update(acc5.item(), B)

    logger.log(
        f"  Validation:  Loss: {loss_meter.avg:.4f}  "
        f"Acc@1: {acc1_meter.avg:.2f}%  Acc@5: {acc5_meter.avg:.2f}%"
    )

    return {
        "val_loss": loss_meter.avg,
        "val_acc1": acc1_meter.avg,
        "val_acc5": acc5_meter.avg,
    }
@torch.no_grad()
def validate_tta(model, cfg, logger, num_workers=8):
    """
    TTA validation: 2 temporal clips x 3 spatial crops.
    Averages logits across 6 views before argmax.
    """
    model.eval()
    use_amp = cfg.train.amp and torch.cuda.is_available()

    dataset = SSv2DatasetTTA(
        data_root=cfg.data.data_root,
        num_frames=cfg.data.num_frames,
        img_size=cfg.data.img_size,
        video_dir=cfg.data.video_dir,
        anno_dir=cfg.data.anno_dir)

    loader = DataLoader(
        dataset, batch_size=max(cfg.train.batch_size // 2, 1),
        shuffle=False, num_workers=num_workers,
        pin_memory=cfg.data.pin_memory, drop_last=False)

    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    logger.log(f"  TTA eval: {len(dataset)} videos, "
               f"2 clips x 3 crops = 6 views each")

    for views, targets in loader:
        B, V, C, T_len, H, W = views.shape
        targets = targets.to(cfg.train.device, non_blocking=True)
        views = views.view(B * V, C, T_len, H, W).to(
            cfg.train.device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(views)

        logits = logits.view(B, V, -1).mean(dim=1)
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        acc1_meter.update(acc1.item(), B)
        acc5_meter.update(acc5.item(), B)

    logger.log(
        f"  TTA Validation:  Acc@1: {acc1_meter.avg:.2f}%  "
        f"Acc@5: {acc5_meter.avg:.2f}%")

    return {
        "tta_acc1": acc1_meter.avg,
        "tta_acc5": acc5_meter.avg,
    }
