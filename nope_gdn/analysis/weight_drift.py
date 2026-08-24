"""Weight drift from pretrained init — how much did each model have to change?

For each model we reconstruct the SAME initialization it was trained from
(run_training's dispatch):
    nope_gdn                 -> load_videomae_pretrained          (encoder 0..11)
    axial_rope / video_rope  -> load_videomae_into_rope           (blocks 0..11)
    trecvit                  -> load_videomae_pretrained_trecvit  (spatial blocks)
then compare to the trained checkpoint, per parameter:
    rel_drift = ‖W_trained − W_init‖_F / ‖W_init‖_F
Parameters the init actually overwrote are "pretrained-reused"; the rest are
"trained from scratch" (we report their share of the model instead).
"""
import re
import json
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

from ..config import get_config
from ..models.factory import build_model
from ..models.pretrained import (load_videomae_pretrained,
                                 load_videomae_pretrained_trecvit,
                                 load_videomae_into_rope)

VMAE_DEFAULT = "MCG-NJU/videomae-base"


def _apply_pretrained_init(variant, model, vmae_ckpt=VMAE_DEFAULT):
    """Reconstruct the training-time pretrained init (mirrors run_training)."""
    if variant == "trecvit":
        return load_videomae_pretrained_trecvit(model, vmae_ckpt)
    if variant in ("rope", "axial_rope", "mixed_rope"):
        return load_videomae_into_rope(model, vmae_ckpt)
    return load_videomae_pretrained(model, vmae_ckpt)


def _load_trained_state(trained, device="cpu"):
    """Accept a path or an already-loaded state dict. Prefer EMA shadow."""
    if isinstance(trained, dict) and not any(
            k in trained for k in ("ema_state", "model_state")):
        return trained                                   # already a bare state dict
    ck = trained if isinstance(trained, dict) else torch.load(
        trained, map_location=device, weights_only=False)
    ema = ck.get("ema_state")
    if ema and "shadow" in ema:
        return ema["shadow"]
    return ck.get("model_state", ck)


def _stage_of(name):
    if "processor" in name or ".temporal." in name or "gdn" in name.lower():
        return "processor/temporal"
    if "encoder" in name or re.search(r'(^|\.)blocks\.', name) or "spatial" in name:
        return "encoder"
    if "head" in name or "pre_logits" in name or name.startswith("fc"):
        return "head"
    return "other"


def _block_idx(name):
    # matches "...blocks.7.xxx" and top-level "blocks.7.xxx" (no leading dot)
    m = re.search(r'(?:^|\.)(?:blocks|layer|spatial|temporal|processor_blocks)\.(\d+)(?=\.|$)', name)
    return int(m.group(1)) if m else -1


def _param_type(name):
    """Coarse parameter-type bucket (works across NoPE/RoPE/TRecViT naming)."""
    n = name.lower()
    if "norm" in n:                                    return "norm (LN)"
    if "qkv" in n or "in_proj" in n:                   return "attn: QKV"
    if "out_proj" in n:                                return "attn: out"
    if "mlp" in n or "intermediate" in n or "output.dense" in n or "fc" in n:
        return "MLP"
    if "tubelet" in n or "tokenizer" in n or "patch" in n:
        return "patch-embed"
    return "other"


@torch.no_grad()
def measure_weight_drift(variant, trained, size="base", vmae_ckpt=VMAE_DEFAULT,
                         init_fn=None, device="cpu", model_overrides=None):
    """Return per-parameter drift rows for one variant.

    init_fn(model) -> model applies the pretrained init in place (default: the
    training-time dispatch). Pass a custom init_fn for testing without a download.
    """
    cfg = get_config(variant=variant, size=size)
    for k, v in (model_overrides or {}).items():
        setattr(cfg.model, k, v)
    model = build_model(variant, cfg.model).to(device).eval()

    s_rand = {k: v.detach().clone() for k, v in model.state_dict().items()}
    (init_fn or (lambda m: _apply_pretrained_init(variant, m, vmae_ckpt)))(model)
    s_init = {k: v.detach().clone() for k, v in model.state_dict().items()}
    pretrained = {k for k in s_init if not torch.equal(s_init[k], s_rand[k])}

    trained_sd = _load_trained_state(trained, device)
    rows = []
    for k, wi in s_init.items():
        wt = trained_sd.get(k)
        if wt is None or tuple(wt.shape) != tuple(wi.shape):
            continue
        wi_f = wi.float().flatten()
        wt_f = wt.float().flatten().to(wi_f.device)
        denom = wi_f.norm().item()
        rows.append({
            "param": k, "pretrained": k in pretrained,
            "rel_drift": ((wt_f - wi_f).norm().item() / denom) if denom > 1e-12 else float('nan'),
            "cos_to_init": float(torch.nn.functional.cosine_similarity(
                wt_f, wi_f, dim=0).item()) if denom > 1e-12 else float('nan'),
            "numel": int(wi_f.numel()),
            "stage": _stage_of(k), "block": _block_idx(k),
            "ptype": _param_type(k), "is_bias": k.endswith(".bias"),
        })
    del model
    return rows


def _summarize(rows):
    pre = [r for r in rows if r["pretrained"] and np.isfinite(r["rel_drift"])]
    scratch = [r for r in rows if not r["pretrained"]]
    total_numel = sum(r["numel"] for r in rows) or 1
    return {
        "mean_drift_pretrained": float(np.mean([r["rel_drift"] for r in pre])) if pre else float('nan'),
        "numel_wt_drift_pretrained": float(
            np.average([r["rel_drift"] for r in pre], weights=[r["numel"] for r in pre])) if pre else float('nan'),
        "mean_cos_to_init": float(np.mean([r["cos_to_init"] for r in pre])) if pre else float('nan'),
        "n_pretrained_params": len(pre),
        "n_scratch_params": len(scratch),
        "scratch_numel_frac": sum(r["numel"] for r in scratch) / total_numel,
        "pretrained_numel_frac": sum(r["numel"] for r in pre) / total_numel,
    }


def _per_block(rows):
    """Mean rel_drift of pretrained params per (stage, block)."""
    agg = defaultdict(list)
    for r in rows:
        if r["pretrained"] and np.isfinite(r["rel_drift"]) and r["block"] >= 0:
            agg[(r["stage"], r["block"])].append(r["rel_drift"])
    return {f"{s}[{b}]": float(np.mean(v)) for (s, b), v in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1]))}


def _by_param_type(rows):
    """Per param-type: unweighted mean, numel-weighted mean, count, param share."""
    agg = defaultdict(lambda: {"d": [], "n": []})
    total = sum(r["numel"] for r in rows if r["pretrained"] and np.isfinite(r["rel_drift"])) or 1
    for r in rows:
        if r["pretrained"] and np.isfinite(r["rel_drift"]):
            agg[r["ptype"]]["d"].append(r["rel_drift"])
            agg[r["ptype"]]["n"].append(r["numel"])
    out = {}
    for t, v in agg.items():
        out[t] = {"mean": float(np.mean(v["d"])),
                  "numel_wt": float(np.average(v["d"], weights=v["n"])),
                  "n_params": len(v["d"]),
                  "numel_frac": sum(v["n"]) / total}
    return out


def _weight_vs_bias(rows):
    """Split pretrained drift into 'matrix weights' vs 'bias + LayerNorm' — the
    latter often have tiny ‖W_init‖ and inflate the unweighted mean."""
    W = [r["rel_drift"] for r in rows if r["pretrained"] and np.isfinite(r["rel_drift"])
         and not r["is_bias"] and r["ptype"] != "norm (LN)"]
    B = [r["rel_drift"] for r in rows if r["pretrained"] and np.isfinite(r["rel_drift"])
         and (r["is_bias"] or r["ptype"] == "norm (LN)")]
    f = lambda x: float(np.mean(x)) if x else float("nan")
    return {"matrix_weights_mean": f(W), "n_matrix": len(W),
            "bias_and_norm_mean": f(B), "n_bias_norm": len(B)}


def run_weight_drift(variants=None, size="base", vmae_ckpt=VMAE_DEFAULT,
                     device="cpu", out_dir="./figures_weight_drift",
                     init_fns=None, model_overrides=None, make_plots=True):
    """variants: {label: {"variant": ..., "ckpt": path-or-state-dict}}."""
    variants = variants or {
        "nope_gdn (global-pool)": {"variant": "nope_gdn",   "ckpt": None},
        "axial_rope":             {"variant": "axial_rope", "ckpt": None},
    }
    init_fns = init_fns or {}
    results = {}
    for label, spec in variants.items():
        v = spec["variant"]
        if spec.get("ckpt") is None:
            print(f"[skip] {label}: no trained checkpoint provided.")
            continue
        try:
            rows = measure_weight_drift(
                v, spec["ckpt"], size=size, vmae_ckpt=vmae_ckpt, device=device,
                init_fn=init_fns.get(label), model_overrides=model_overrides)
            results[label] = {"rows": rows, "summary": _summarize(rows),
                              "per_block": _per_block(rows),
                              "by_type": _by_param_type(rows),
                              "weight_vs_bias": _weight_vs_bias(rows)}
            s = results[label]["summary"]
            print(f"[{label}] mean drift (pretrained)={s['mean_drift_pretrained']:.4f} "
                  f"| cos_to_init={s['mean_cos_to_init']:.4f} "
                  f"| from-scratch params={s['scratch_numel_frac']*100:.1f}% of model")
        except Exception as e:
            print(f"[skip] {label}: {type(e).__name__}: {e}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "weight_drift_raw.json", "w") as f:
        json.dump({lab: {"summary": r["summary"], "per_block": r["per_block"],
                         "by_type": r["by_type"], "weight_vs_bias": r["weight_vs_bias"]}
                   for lab, r in results.items()}, f, indent=2)

    _print_drift_summary(results)
    if make_plots and results:
        try:
            _plot_drift(results, out_dir)
            print(f"\nFigures + raw JSON saved to: {out_dir}")
        except Exception as e:
            print(f"  plotting skipped: {type(e).__name__}: {e}")
    return results


def _print_drift_summary(results):
    print("\n=== Weight drift from pretrained init ===")
    print(f"{'variant':26s} {'mean drift':>11} {'numel-wt':>10} {'cos→init':>9} "
          f"{'reused %':>9} {'scratch %':>10}")
    for lab, r in results.items():
        s = r["summary"]
        print(f"{lab:26s} {s['mean_drift_pretrained']:>11.4f} "
              f"{s['numel_wt_drift_pretrained']:>10.4f} {s['mean_cos_to_init']:>9.4f} "
              f"{s['pretrained_numel_frac']*100:>8.1f}% {s['scratch_numel_frac']*100:>9.1f}%")
    print("\nLower drift + lower scratch% = reused the pretrained model more (changed less).")

    # ---- per param-type breakdown (numel-weighted drift) ----
    types = sorted({t for r in results.values() for t in r["by_type"]})
    print("\n=== Per param-type drift (numel-weighted ‖ΔW‖/‖W_init‖) ===")
    print(f"{'variant':26s} " + "".join(f"{t:>13}" for t in types))
    for lab, r in results.items():
        bt = r["by_type"]
        print(f"{lab:26s} " + "".join(
            f"{bt[t]['numel_wt']:>13.3f}" if t in bt else f"{'—':>13}" for t in types))

    # ---- weights vs bias/norm (explains the ~1.0 mean) ----
    print("\n=== Matrix weights vs (bias + LayerNorm) — mean drift ===")
    print(f"{'variant':26s} {'matrix W':>12} {'bias+norm':>12}")
    for lab, r in results.items():
        wb = r["weight_vs_bias"]
        print(f"{lab:26s} {wb['matrix_weights_mean']:>12.3f} {wb['bias_and_norm_mean']:>12.3f}")
    print("If bias+norm >> matrix W, the ~1.0 mean drift is inflated by small-norm "
          "params (LN/bias init near 0/1); trust the matrix-W column.")


def _plot_drift(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(results)
    markers = ['o', 's', '^', 'D']

    # (1) per-encoder-block drift curve
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, lab in enumerate(labels):
        pb = results[lab]["per_block"]
        enc = {int(re.search(r'\[(\d+)\]', k).group(1)): v
               for k, v in pb.items() if k.startswith("encoder")}
        if enc:
            xs = sorted(enc); ax.plot(xs, [enc[x] for x in xs],
                                      marker=markers[i % len(markers)], linewidth=2, label=lab)
    ax.set_xlabel("Encoder block index")
    ax.set_ylabel("relative weight drift  ‖ΔW‖ / ‖W_init‖")
    ax.set_title("Per-block drift of pretrained (VideoMAE) encoder weights")
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "weight_drift_per_block.pdf", bbox_inches='tight')
    plt.savefig(Path(out_dir) / "weight_drift_per_block.png", bbox_inches='tight', dpi=150)
    plt.close()

    # (2) summary bars: mean drift + from-scratch fraction
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(labels))
    axes[0].bar(x, [results[l]["summary"]["mean_drift_pretrained"] for l in labels],
                color='steelblue')
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=20, ha='right')
    axes[0].set_ylabel("mean ‖ΔW‖ / ‖W_init‖ (pretrained params)")
    axes[0].set_title("How much the reused pretrained weights changed")
    axes[0].grid(True, axis='y', alpha=0.3)
    axes[1].bar(x, [results[l]["summary"]["scratch_numel_frac"] * 100 for l in labels],
                color='indianred')
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=20, ha='right')
    axes[1].set_ylabel("% of parameters trained from scratch")
    axes[1].set_title("How much of the model had no pretrained init")
    axes[1].grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "weight_drift_summary.pdf", bbox_inches='tight')
    plt.savefig(Path(out_dir) / "weight_drift_summary.png", bbox_inches='tight', dpi=150)
    plt.close()

    # (3) per param-type grouped bars (numel-weighted drift)
    types = sorted({t for l in labels for t in results[l]["by_type"]})
    if types:
        fig, ax = plt.subplots(figsize=(1.6 * len(types) + 4, 5))
        w = 0.8 / max(len(labels), 1)
        xt = np.arange(len(types))
        for i, lab in enumerate(labels):
            bt = results[lab]["by_type"]
            ax.bar(xt + i * w, [bt[t]["numel_wt"] if t in bt else 0 for t in types],
                   width=w, label=lab)
        ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
        ax.set_xticks(xt + w * (len(labels) - 1) / 2); ax.set_xticklabels(types, rotation=15)
        ax.set_ylabel("numel-weighted  ‖ΔW‖ / ‖W_init‖")
        ax.set_title("Weight drift by parameter type")
        ax.legend(fontsize=8); ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(out_dir) / "weight_drift_by_type.pdf", bbox_inches='tight')
        plt.savefig(Path(out_dir) / "weight_drift_by_type.png", bbox_inches='tight', dpi=150)
        plt.close()


def _mock_smoke():
    """Mock smoke test (no download / no checkpoints): validates the drift math
    + pretrained-vs-scratch detection using a synthetic "pretrained init"
    (overwrites encoder blocks) and a synthetic trained ckpt."""
    print("\nMock smoke test (synthetic init + trained; validates the drift math)...")
    ov = {"encoder_depth": 3, "processor_depth": 4, "encoder_dim": 96,
          "processor_dim": 96, "encoder_heads": 3, "processor_heads": 3}

    def mock_init(m):                       # overwrites ONLY encoder blocks, in place
        with torch.no_grad():
            for n, p in m.named_parameters():
                if _stage_of(n) == "encoder" and _block_idx(n) >= 0:
                    p.copy_(torch.full_like(p, 0.05))
        return m

    def make_trained(v):                    # init (encoder=0.05) + known ~0.4 rel drift
        cfg = get_config(variant=v, size="tiny")
        for k, val in ov.items(): setattr(cfg.model, k, val)
        m = build_model(v, cfg.model).eval(); mock_init(m)
        sd = {k: val.detach().clone() for k, val in m.state_dict().items()}
        torch.manual_seed(0)
        for k, val in sd.items():
            if val.numel() and torch.equal(val, torch.full_like(val, 0.05)):
                sd[k] = val + 0.02 * torch.randn_like(val)     # ‖Δ‖/‖W‖ ≈ 0.02/0.05 = 0.4
        return sd

    variants, init_fns = {}, {}
    for lab, v in [("nope_gdn (global-pool)", "nope_gdn"), ("axial_rope", "axial_rope")]:
        variants[lab] = {"variant": v, "ckpt": make_trained(v)}
        init_fns[lab] = mock_init
    run_weight_drift(variants, size="tiny", init_fns=init_fns,
                     model_overrides=ov, out_dir="./figures_weight_drift_smoke")


if __name__ == "__main__":
    _mock_smoke()
