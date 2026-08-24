import torch
import torch.nn as nn


def _is_proc_head(name):
    return ("processor" in name or "dim_proj" in name
            or name == "head" or name.startswith("head."))


def boost_proc_head_stochastic_reg(model, proc_drop_path=None, proc_dropout=None,
                                   head_dropout=None, verbose=True):
    """Raise stochastic depth / dropout on the processor + head ONLY, in place.
    Param-free, so apply to a freshly-built model before training (None = leave)."""
    n_dp = n_do = 0
    for name, m in model.named_modules():
        if not _is_proc_head(name):
            continue
        if type(m).__name__ == "DropPath" and proc_drop_path is not None:
            m.drop_prob = proc_drop_path
            n_dp += 1
        elif isinstance(m, nn.Dropout):
            tgt = head_dropout if (name.startswith("head") and head_dropout is not None) else proc_dropout
            if tgt is not None:
                m.p = tgt
                n_do += 1
    if verbose:
        print(f"  proc+head reg: {n_dp} DropPath -> {proc_drop_path}; "
              f"{n_do} Dropout updated (proc={proc_dropout}, head={head_dropout})")
    return model


def build_llrd_optimizer(model, cfg, proc_head_wd=None, proc_head_lr_mult=1.0, verbose=True):
    """Faithful re-implementation of run_training's layer-wise-LR-decay AdamW builder,
    plus an extra knob: a separate (higher) weight_decay and optional lr multiplier for
    the random-init processor+head stack only. proc_head_wd=None reproduces the baseline
    exactly. Uses NoPE-GDN parameter naming (backbone.encoder.* vs processor/head)."""
    enc_depth   = cfg.model.encoder_depth
    num_layers  = enc_depth + 2                  # processor + head = top layer, lr_scale 1.0
    layer_decay = cfg.train.layer_decay
    base_wd     = cfg.train.weight_decay
    ph_wd       = base_wd if proc_head_wd is None else proc_head_wd

    def layer_id(name):
        if "tubelet_embed" in name:
            return 0
        if "backbone.encoder.blocks." in name:
            return int(name.split("backbone.encoder.blocks.")[1].split(".")[0]) + 1
        if "backbone.encoder.norm" in name:
            return enc_depth
        return enc_depth + 1                      # processor / dim_proj / head / final norm

    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_wd = (p.dim() <= 1 or "bias" in name or "norm" in name
                 or "rope.theta_" in name or getattr(p, "_no_weight_decay", False))
        ph = _is_proc_head(name)
        wd = 0.0 if no_wd else (ph_wd if ph else base_wd)
        lid = layer_id(name)
        lr_scale = layer_decay ** (num_layers - 1 - lid)
        if ph:
            lr_scale *= proc_head_lr_mult
        key = f"L{lid}_wd{wd}_ph{int(ph)}"
        if key not in groups:
            groups[key] = {"params": [], "weight_decay": wd,
                           "lr": cfg.train.lr * lr_scale, "lr_scale": lr_scale}
        groups[key]["params"].append(p)

    opt = torch.optim.AdamW(list(groups.values()), lr=cfg.train.lr, betas=cfg.train.betas)
    if verbose:
        n_ph = sum(len(g["params"]) for k, g in groups.items() if k.endswith("ph1"))
        print(f"  LLRD AdamW: {len(groups)} groups | base_wd={base_wd} "
              f"proc_head_wd={ph_wd} proc_head_lr_mult={proc_head_lr_mult} "
              f"| {n_ph} proc/head weight tensors boosted")
    return opt
