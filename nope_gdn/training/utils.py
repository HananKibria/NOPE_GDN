import math
import os
import random
from datetime import datetime

import torch
import torch.nn.functional as F


class CosineWarmupScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 warmup_lr: float, base_lr: float, min_lr: float,
                 steps_per_epoch: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.warmup_lr = warmup_lr
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.step_count = 0

    def step(self):
        self.step_count += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group['lr'] = lr * param_group['lr_scale']
            else:
                param_group['lr'] = lr
        return lr

    def _compute_lr(self):
        if self.step_count <= self.warmup_steps:
            alpha = self.step_count / max(self.warmup_steps, 1)
            return self.warmup_lr + (self.base_lr - self.warmup_lr) * alpha
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1)
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress))

    def state_dict(self):
        return {"step_count": self.step_count}

    def load_state_dict(self, state):
        self.step_count = state["step_count"]


class AverageMeter:
    """Tracks running average."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(logits, targets, topk=(1, 5)):
    """Compute top-k accuracy. Handles both hard and soft labels."""
    maxk = max(topk)
    batch_size = logits.size(0)

    if targets.dim() == 2:
        targets = targets.argmax(dim=1)

    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    correct = pred.eq(targets.unsqueeze(1).expand_as(pred))

    result = []
    for k in topk:
        correct_k = correct[:, :k].reshape(-1).float().sum()
        result.append(correct_k * 100.0 / batch_size)
    return result


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MixupCutmix:
    """Mixup + CutMix data augmentation for video.

    With probability `prob`, applies either Mixup or CutMix (chosen per-batch
    via `switch_prob` when both are configured); otherwise returns the batch
    with label smoothing only.

    Mixup: linearly interpolates pixel values across paired clips.
    CutMix: cuts a random spatial rectangle and pastes from a paired clip.
            For video, the SAME spatial box is applied to every frame in the
            clip (no temporal cut). `lam` is recomputed from the actual area
            of the cut after boundary clipping.

    Expected input shape: videos (B, C, T, H, W), targets (B,) long.
    """
    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, prob=0.5,
                 switch_prob=0.5, num_classes=174, label_smoothing=0.1):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def _smooth_one_hot(self, targets):
        oh = F.one_hot(targets, self.num_classes).float()
        return oh * (1.0 - self.label_smoothing) + \
               self.label_smoothing / self.num_classes

    def _mixup(self, videos, indices, lam):
        return lam * videos + (1.0 - lam) * videos[indices]

    def _cutmix(self, videos, indices, lam):
        # videos: (B, C, T, H, W) — apply the same spatial box to every frame.
        _, _, _, H, W = videos.shape
        cut_ratio = math.sqrt(max(1.0 - lam, 0.0))
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        cy = int(torch.randint(0, H, (1,)).item())
        cx = int(torch.randint(0, W, (1,)).item())
        y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)

        mixed = videos.clone()
        mixed[:, :, :, y1:y2, x1:x2] = videos[indices][:, :, :, y1:y2, x1:x2]
        # Re-derive lam from actual kept area (boundary clipping shifts it).
        lam_eff = 1.0 - ((y2 - y1) * (x2 - x1)) / float(H * W)
        return mixed, lam_eff

    def __call__(self, videos, targets):
        if self.prob <= 0 or random.random() > self.prob:
            return videos, self._smooth_one_hot(targets)

        # Decide mixup vs cutmix
        use_mixup = self.mixup_alpha > 0
        use_cutmix = self.cutmix_alpha > 0
        if use_mixup and use_cutmix:
            do_cutmix = random.random() < self.switch_prob
        elif use_cutmix:
            do_cutmix = True
        elif use_mixup:
            do_cutmix = False
        else:
            return videos, self._smooth_one_hot(targets)

        indices = torch.randperm(videos.size(0), device=videos.device)

        if do_cutmix:
            lam = float(torch.distributions.Beta(
                self.cutmix_alpha, self.cutmix_alpha).sample())
            mixed_videos, lam = self._cutmix(videos, indices, lam)
        else:
            lam = float(torch.distributions.Beta(
                self.mixup_alpha, self.mixup_alpha).sample())
            lam = max(lam, 1.0 - lam)
            mixed_videos = self._mixup(videos, indices, lam)

        targets_oh = F.one_hot(targets, self.num_classes).float()
        mixed_targets = lam * targets_oh + (1.0 - lam) * targets_oh[indices]
        mixed_targets = mixed_targets * (1.0 - self.label_smoothing) + \
                        self.label_smoothing / self.num_classes
        return mixed_videos, mixed_targets


class ModelEMA:
    """
    Exponential Moving Average of model parameters.
    Maintains a shadow copy of model weights that is a running average,
    which often generalizes better than the raw training weights.
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply_shadow(self, model):
        """Replace model params with EMA params (for eval)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original model params (after eval)."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {'shadow': {k: v.cpu() for k, v in self.shadow.items()},
                'decay': self.decay}

    def load_state_dict(self, state):
        self.decay = state['decay']
        for name in state['shadow']:
            if name in self.shadow:
                self.shadow[name] = state['shadow'][name].to(self.shadow[name].device)
            else:
                self.shadow[name] = state['shadow'][name]

    def to(self, device):
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        return self


def save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                    best_acc, metrics_history, save_path, ema=None):
    """Save training checkpoint."""
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler else None,
        "best_acc": best_acc,
        "metrics_history": metrics_history,
        "ema_state": ema.state_dict() if ema else None,
    }
    torch.save(state, save_path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None,
                    ema=None, strict_optim=False):
    """Load checkpoint and return (start_epoch, best_acc, metrics_history).

    strict_optim=False (default) tolerates an optimizer-state mismatch (different
    number of param groups, e.g. when the layer-id scheme changed between runs).
    On mismatch we warn and skip just the optimizer state; everything else
    (model, EMA, scheduler, scaler, metadata) still loads.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except ValueError as e:
            if strict_optim:
                raise
            saved_n = len(ckpt["optimizer_state"].get("param_groups", []))
            curr_n = len(optimizer.param_groups)
            print(f"  \u26a0\ufe0f Optimizer state has {saved_n} param groups but current "
                  f"optimizer has {curr_n}; skipping optimizer state load.")
            print(f"     Reason: {e}")
            print(f"     AdamW momentum lost; warmup steps will recover. "
                  f"Pass strict_optim=True to enforce a match.")
    if scheduler and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler and ckpt.get("scaler_state"):
        scaler.load_state_dict(ckpt["scaler_state"])
    if ema and ckpt.get("ema_state"):
        ema.load_state_dict(ckpt["ema_state"])
    return (
        ckpt.get("epoch", 0),
        ckpt.get("best_acc", 0),
        ckpt.get("metrics_history", []),
    )


class Logger:
    """Logs to both console and file."""
    def __init__(self, log_path: str = None):
        self.log_path = log_path
        self.file = None
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.file = open(log_path, "a")

    def log(self, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        if self.file:
            self.file.write(line + "\n")
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()
