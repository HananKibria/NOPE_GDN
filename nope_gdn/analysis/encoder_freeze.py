"""Frozen / reset-encoder ablation — did the model NEED its encoder to change?

For each TRAINED model we revert its VideoMAE-pretrained encoder back toward
the init and measure accuracy — keeping the processor/head trained:

    W_encoder(α) = (1−α)·W_init + α·W_trained        (α=1 trained … α=0 pristine)

Accuracy vs α: a STEEP drop as α→0 means the accuracy depends heavily on the
encoder having been adapted; a FLAT curve means the pretrained encoder was
already enough. No retraining. Reuses the §17 init reconstruction from
weight_drift. Caveat: the processor was trained on the *adapted* encoder's
features, so a reset also breaks encoder↔processor co-adaptation — the α-curve
SHAPE (and the gap between variants) is the signal.
"""
import json
from pathlib import Path

import torch
import numpy as np

from ..config import get_config
from ..models.factory import build_model
from .weight_drift import (VMAE_DEFAULT, _apply_pretrained_init,
                           _load_trained_state, _stage_of, _block_idx)


def _encoder_pretrained_keys(variant, cfg_model, vmae_ckpt, device, init_fn=None):
    """Return the encoder-stage parameter keys the pretrained init overwrote,
    plus the reconstructed init state dict."""
    m = build_model(variant, cfg_model).to(device).eval()
    s_rand = {k: v.detach().clone() for k, v in m.state_dict().items()}
    (init_fn or (lambda mm: _apply_pretrained_init(variant, mm, vmae_ckpt)))(m)
    s_init = {k: v.detach().clone() for k, v in m.state_dict().items()}
    keys = [k for k in s_init
            if _stage_of(k) == "encoder" and not torch.equal(s_init[k], s_rand[k])]
    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    return keys, s_init


@torch.no_grad()
def _blend_encoder(trained_sd, init_sd, keys, alpha):
    """W_encoder = (1-α)·init + α·trained, on the given keys; rest = trained."""
    out = {k: v.clone() for k, v in trained_sd.items()}
    for k in keys:
        if k in init_sd and k in trained_sd:
            wi, wt = init_sd[k].float(), trained_sd[k].float().to(init_sd[k].device)
            out[k] = ((1.0 - alpha) * wi + alpha * wt).to(trained_sd[k].dtype)
    return out


def run_encoder_freeze_ablation(variants, eval_fn, alphas=(1.0, 0.75, 0.5, 0.25, 0.0),
                                size="base", vmae_ckpt=VMAE_DEFAULT, device="cuda",
                                out_dir="./figures_freeze_ablation", init_fns=None,
                                model_overrides=None, make_plots=True):
    """variants: {label: {"variant": ..., "ckpt": path-or-state-dict}}.
    eval_fn(model) -> top-1 accuracy (any consistent scale, e.g. 0–100)."""
    init_fns = init_fns or {}
    results = {}
    for label, spec in variants.items():
        v = spec["variant"]
        if spec.get("ckpt") is None:
            print(f"[skip] {label}: no trained checkpoint."); continue
        try:
            cfg = get_config(variant=v, size=size)
            for k, val in (model_overrides or {}).items():
                setattr(cfg.model, k, val)
            trained_sd = _load_trained_state(spec["ckpt"], device)
            keys, init_sd = _encoder_pretrained_keys(
                v, cfg.model, vmae_ckpt, device, init_fns.get(label))
            model = build_model(v, cfg.model).to(device).eval()
            accs = {}
            for a in alphas:
                model.load_state_dict(_blend_encoder(trained_sd, init_sd, keys, a),
                                      strict=False)
                accs[float(a)] = float(eval_fn(model))
            a_hi, a_lo = max(alphas), min(alphas)
            drop = accs[float(a_hi)] - accs[float(a_lo)]
            rel = drop / accs[float(a_hi)] * 100 if accs[float(a_hi)] else float('nan')
            results[label] = {"acc_by_alpha": accs, "n_encoder_keys": len(keys),
                              "acc_trained": accs[float(a_hi)], "acc_reset": accs[float(a_lo)],
                              "abs_drop": drop, "rel_drop_pct": rel}
            print(f"[{label}] {len(keys)} enc params | trained α=1: {accs[float(a_hi)]:.2f} "
                  f"-> reset α=0: {accs[float(a_lo)]:.2f}  (drop {drop:.2f}, {rel:.1f}%)")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[skip] {label}: {type(e).__name__}: {e}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "freeze_ablation_raw.json", "w") as f:
        json.dump({lab: {k: v for k, v in r.items()} for lab, r in results.items()},
                  f, indent=2, default=str)

    _print_freeze_summary(results, alphas)
    if make_plots and results:
        try:
            _plot_freeze(results, alphas, out_dir)
            print(f"\nFigures + raw JSON saved to: {out_dir}")
        except Exception as e:
            print(f"  plotting skipped: {type(e).__name__}: {e}")
    return results


def _print_freeze_summary(results, alphas):
    print("\n=== Encoder reset ablation (accuracy vs α; α=1 trained, α=0 pristine init) ===")
    print(f"{'variant':26s} " + "".join(f"α={a:<5}" for a in alphas) +
          f"{'drop':>8}{'rel%':>8}")
    for lab, r in results.items():
        row = "".join(f"{r['acc_by_alpha'].get(float(a), float('nan')):<7.2f}" for a in alphas)
        print(f"{lab:26s} {row}{r['abs_drop']:>8.2f}{r['rel_drop_pct']:>7.1f}%")
    print("\nBigger drop (steeper α-curve) = model needed its encoder to change more.")


def _plot_freeze(results, alphas, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(results)
    markers = ['o', 's', '^', 'D']
    alphas = sorted(alphas)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # absolute accuracy vs alpha
    for i, lab in enumerate(labels):
        ab = results[lab]["acc_by_alpha"]
        axes[0].plot(alphas, [ab.get(float(a), np.nan) for a in alphas],
                     marker=markers[i % len(markers)], linewidth=2, label=lab)
    axes[0].set_xlabel("α  (0 = encoder reset to VideoMAE init, 1 = trained)")
    axes[0].set_ylabel("Top-1 accuracy"); axes[0].set_title("Accuracy vs encoder reset")
    axes[0].grid(True, alpha=0.3); axes[0].legend()
    # normalized to α=1 (retained accuracy fraction)
    for i, lab in enumerate(labels):
        ab = results[lab]["acc_by_alpha"]; base = ab.get(1.0) or np.nan
        axes[1].plot(alphas, [ab.get(float(a), np.nan) / base if base else np.nan for a in alphas],
                     marker=markers[i % len(markers)], linewidth=2, label=lab)
    axes[1].axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    axes[1].set_xlabel("α"); axes[1].set_ylabel("accuracy retained (÷ α=1)")
    axes[1].set_title("Reliance on encoder adaptation (lower = needed it more)")
    axes[1].grid(True, alpha=0.3); axes[1].legend()
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "freeze_ablation.pdf", bbox_inches='tight')
    plt.savefig(Path(out_dir) / "freeze_ablation.png", bbox_inches='tight', dpi=150)
    plt.close()


def _freeze_smoke():
    """Mock smoke test (synthetic init + trained + eval; validates the surgery
    + α-sweep + plot)."""
    print("\nMock smoke test (synthetic init + trained + eval; validates the α-sweep)...")
    ov = {"encoder_depth": 3, "processor_depth": 4, "encoder_dim": 96,
          "processor_dim": 96, "encoder_heads": 3, "processor_heads": 3, "img_size": 32}

    def mock_init(m):
        with torch.no_grad():
            for n, p in m.named_parameters():
                if _stage_of(n) == "encoder" and _block_idx(n) >= 0:
                    p.copy_(torch.full_like(p, 0.05))
        return m

    def make_trained(v):
        cfg = get_config(variant=v, size="tiny")
        for k, val in ov.items(): setattr(cfg.model, k, val)
        m = build_model(v, cfg.model).eval(); mock_init(m)
        sd = {k: val.detach().clone() for k, val in m.state_dict().items()}
        torch.manual_seed(0)
        for k, val in sd.items():
            if val.numel() and torch.equal(val, torch.full_like(val, 0.05)):
                sd[k] = val + 0.02 * torch.randn_like(val)
        return sd

    @torch.no_grad()
    def mock_eval(model):     # proxy "accuracy": mean top-1 softmax over fixed random clips
        torch.manual_seed(123)
        x = torch.randn(2, 3, 8, 32, 32)
        return float(model(x).softmax(-1).max(-1).values.mean().item() * 100)

    variants, init_fns = {}, {}
    for lab, v in [("nope_gdn (global-pool)", "nope_gdn"), ("axial_rope", "axial_rope")]:
        variants[lab] = {"variant": v, "ckpt": make_trained(v)}
        init_fns[lab] = mock_init
    run_encoder_freeze_ablation(variants, mock_eval, size="tiny", init_fns=init_fns,
                                model_overrides=ov, device="cpu",
                                out_dir="./figures_freeze_ablation_smoke")


if __name__ == "__main__":
    _freeze_smoke()
