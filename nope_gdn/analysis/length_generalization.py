"""Length-generalization mechanism — encoder attention concentration vs length.

For each attention layer, at each sequence length: H_norm = H / log N
(normalized entropy; 1.0 = uniform, 0.0 = delta) and N_eff = exp(H) (effective
# of attended tokens). Variants: nope_gdn (spatial-only), axial_rope and
video_rope (global), trecvit (spatial-only). Spatial-only attention stays flat
because N = #spatial tokens is length-independent, while global attention
diffuses as N = T·S grows past the training length.
"""
import math
import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

from ..config import get_config
from ..models.factory import build_model


class AttentionCapture:
    """Recomputes post-softmax attention and stores H_norm / N_eff per layer."""
    RECOGNIZED = ("NoPEMultiheadAttention", "RoPEAttention", "_TRecViTSpatialBlock")

    def __init__(self, scope="all", n_encoder=12):
        self.scope = scope
        self.n_encoder = n_encoder
        self.captures = defaultdict(list)
        self._orig_forwards = {}

    def _select_layers(self, model):
        mods = [(n, m) for n, m in model.named_modules()
                if type(m).__name__ in self.RECOGNIZED]
        if self.scope == "all":
            return mods
        if self.scope == "encoder":
            return mods[:self.n_encoder]
        if self.scope == "processor":
            return mods[self.n_encoder:]
        raise ValueError(f"Unknown scope: {self.scope}")

    def attach(self, model):
        n = 0
        for name, module in self._select_layers(model):
            cls = type(module).__name__
            if cls == "NoPEMultiheadAttention":
                self._patch_nope(module, name)
            elif cls == "RoPEAttention":
                self._patch_rope(module, name)
            elif cls == "_TRecViTSpatialBlock":
                self._patch_trecvit(module, name)
            n += 1
        return n

    @staticmethod
    def _entropy_metrics(attn):
        """attn: (..., N, N) post-softmax. Returns (H_norm, N_eff) scalars."""
        N_tok = attn.shape[-1]
        H = -(attn * torch.log(attn.clamp(min=1e-10))).sum(dim=-1)   # (..., N)
        return (H / math.log(N_tok)).mean().item(), H.exp().mean().item()

    def _patch_nope(self, module, layer_name):
        self._orig_forwards[layer_name] = module.forward
        cap = self.captures

        def patched(x, attn_mask=None):
            B, N, D = x.shape
            Hh, d = module.num_heads, module.head_dim
            qkv = module.qkv_proj(x).reshape(B, N, 3, Hh, d)
            q, k, v = qkv.unbind(dim=2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            logits = (q @ k.transpose(-2, -1)) * module.scale
            if attn_mask is not None:
                if attn_mask.dim() == 2:   attn_mask = attn_mask[None, None]
                elif attn_mask.dim() == 3: attn_mask = attn_mask[:, None]
                logits = logits.masked_fill(attn_mask, float('-inf'))
            attn = logits.softmax(dim=-1)
            with torch.no_grad():
                hn, ne = self._entropy_metrics(attn)
            cap[layer_name].append({"H_norm": hn, "N_eff": ne})
            attn = module.attn_dropout(attn)
            out = (attn @ v).transpose(1, 2).reshape(B, N, D)
            return module.out_proj(out)
        module.forward = patched

    def _patch_rope(self, module, layer_name):
        self._orig_forwards[layer_name] = module.forward
        cap = self.captures
        rotate_half_fn = globals().get('rotate_half') or (
            lambda x: torch.cat([-x.chunk(2, -1)[1], x.chunk(2, -1)[0]], dim=-1))

        def patched(x, rope_cache=None, attn_mask=None):
            B, N, D = x.shape
            qkv = module.qkv_proj(x).reshape(B, N, 3, module.embed_dim)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            q = q.reshape(B, N, module.num_heads, module.head_dim).transpose(1, 2)
            k = k.reshape(B, N, module.num_heads, module.head_dim).transpose(1, 2)
            v = v.reshape(B, N, module.num_heads, module.head_dim).transpose(1, 2)
            if rope_cache is not None:
                cos, sin = rope_cache
                cos = cos.permute(1, 0, 2).unsqueeze(0).to(q.dtype)
                sin = sin.permute(1, 0, 2).unsqueeze(0).to(q.dtype)
                q = (q * cos) + (rotate_half_fn(q) * sin)
                k = (k * cos) + (rotate_half_fn(k) * sin)
            logits = (q @ k.transpose(-2, -1)) * module.scale
            if attn_mask is not None:
                if attn_mask.dim() == 2: attn_mask = attn_mask[None, None]
                logits = logits.masked_fill(attn_mask, float('-inf'))
            attn = logits.softmax(dim=-1)
            with torch.no_grad():
                hn, ne = self._entropy_metrics(attn)
            cap[layer_name].append({"H_norm": hn, "N_eff": ne})
            if module.training:
                attn = F.dropout(attn, p=module.attn_dropout.p)
            out = (attn @ v).transpose(1, 2).reshape(B, N, D)
            return module.out_proj(out)
        module.forward = patched

    def _patch_trecvit(self, module, layer_name):
        # _TRecViTSpatialBlock wraps nn.MultiheadAttention; ask it for weights.
        self._orig_forwards[layer_name] = module.forward
        cap = self.captures

        def patched(x):
            y = module.norm1(x)
            attn_out, attn_w = module.attn(
                y, y, y, need_weights=True, average_attn_weights=False)
            with torch.no_grad():                       # attn_w: (B, heads, N, N)
                hn, ne = self._entropy_metrics(attn_w)
            cap[layer_name].append({"H_norm": hn, "N_eff": ne})
            x = x + attn_out
            x = x + module.mlp(module.norm2(x))
            return x
        module.forward = patched

    def restore(self, model):
        for name, module in model.named_modules():
            if name in self._orig_forwards:
                module.forward = self._orig_forwards[name]
        self._orig_forwards.clear()

    def clear(self):
        self.captures.clear()

    def compute_metrics_per_layer(self):
        out = {}
        for layer, lst in self.captures.items():
            if lst:
                out[layer] = {"H_norm": float(np.mean([d["H_norm"] for d in lst])),
                              "N_eff":  float(np.mean([d["N_eff"] for d in lst]))}
        return out


def random_video_loader(length, n_samples, img_size, in_ch, device):
    """Random-tensor video source for smoke tests (pass a real loader for
    publishable numbers)."""
    return torch.randn(n_samples, in_ch, length, img_size, img_size, device=device)


@torch.no_grad()
def measure_variant_at_length(model, length, n_samples, batch_size, img_size,
                              in_ch, device, video_loader):
    videos = video_loader(length, n_samples, img_size, in_ch, device)
    cap = AttentionCapture(scope="all")
    cap.attach(model)
    for i in range(0, videos.shape[0], batch_size):
        model(videos[i:i + batch_size])
    metrics = cap.compute_metrics_per_layer()
    cap.restore(model)
    if device == "cuda":
        torch.cuda.empty_cache()
    return metrics


def _load_ckpt(model, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    ema = ck.get("ema_state") if isinstance(ck, dict) else None
    if ema and "shadow" in ema:
        sd = model.state_dict()
        for n in ema["shadow"]:
            if n in sd:
                sd[n] = ema["shadow"][n]
        model.load_state_dict(sd)
    else:
        model.load_state_dict(ck.get("model_state", ck), strict=False)
    return ck.get("epoch"), ck.get("best_acc")


DEFAULT_VARIANTS = {
    "nope_gdn (global-pool)": {"variant": "nope_gdn",   "ckpt": None},
    "axial_rope":             {"variant": "axial_rope", "ckpt": None},
    "video_rope":             {"variant": "rope",       "ckpt": None},
    "trecvit":                {"variant": "trecvit",    "ckpt": None},
}


def run_length_generalization(variants=None, lengths=(8, 16, 24, 32, 48, 64, 72),
                              train_length=32, size="base", img_size=224,
                              n_samples=64, batch_size=1, video_loader=None,
                              out_dir="./figures_len_gen", device=None,
                              model_overrides=None, make_plots=True):
    variants = variants or DEFAULT_VARIANTS
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    video_loader = video_loader or random_video_loader
    if video_loader is random_video_loader:
        print("⚠️  No video_loader given — using RANDOM inputs (entropy values are a "
              "smoke test, not real data). Pass a loader for publishable numbers.")

    all_metrics = {}          # label -> {length -> {layer -> {H_norm, N_eff}}}
    for label, spec in variants.items():
        v = spec["variant"]
        cfg = get_config(variant=v, size=size)
        cfg.model.num_frames = train_length
        cfg.model.img_size = img_size
        for kk, vv in (model_overrides or {}).items():
            setattr(cfg.model, kk, vv)
        in_ch = cfg.model.in_channels
        try:
            model = build_model(v, cfg.model).to(device).eval()
            if spec.get("ckpt"):
                ep, acc = _load_ckpt(model, spec["ckpt"], device)
                print(f"[{label}] loaded ckpt epoch={ep} best_acc={acc}")
            else:
                print(f"[{label}] no checkpoint — using randomly-initialized weights")
            per_len = {}
            for L in lengths:
                m = measure_variant_at_length(model, L, n_samples, batch_size,
                                              cfg.model.img_size, in_ch, device,
                                              video_loader)
                per_len[L] = m
                mean_h = np.mean([d["H_norm"] for d in m.values()]) if m else float('nan')
                print(f"    {label:24s} L={L:>3}: {len(m)} attn layers, "
                      f"mean H/logN={mean_h:.3f}")
            all_metrics[label] = per_len
        except Exception as e:
            print(f"[skip] {label}: {type(e).__name__}: {e}")
        finally:
            try: del model
            except Exception: pass
            if device == "cuda":
                torch.cuda.empty_cache()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "length_generalization_raw.json", "w") as f:
        json.dump({"lengths": list(lengths), "train_length": train_length,
                   "n_samples": n_samples, "img_size": img_size,
                   "metrics": {lab: {str(L): m for L, m in d.items()}
                               for lab, d in all_metrics.items()}}, f, indent=2)

    _print_summary(all_metrics, lengths)
    if make_plots:
        try:
            _plot_length_generalization(all_metrics, lengths, train_length, out_dir)
            print(f"\nFigures + raw JSON saved to: {out_dir}")
        except Exception as e:
            print(f"  plotting skipped: {type(e).__name__}: {e}")
    return all_metrics


def _mean_curve(per_len, lengths, key):
    return [float(np.mean([d[key] for d in per_len[L].values()])) if per_len.get(L) else float('nan')
            for L in lengths]


def _print_summary(all_metrics, lengths):
    print("\n=== Mean encoder attention H/log N per length ===")
    hdr = "  L   " + "".join(f"{lab[:16]:>18}" for lab in all_metrics)
    print(hdr)
    for L in lengths:
        row = f"{L:>4} " + "".join(
            f"{np.mean([d['H_norm'] for d in all_metrics[lab][L].values()]) if all_metrics[lab].get(L) else float('nan'):>18.4f}"
            for lab in all_metrics)
        print(row)


def _plot_length_generalization(all_metrics, lengths, train_length, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lengths = list(lengths)
    markers = ['o', 's', '^', 'D', 'v', 'P']
    labels = list(all_metrics)

    # (1) H/log N summary
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, lab in enumerate(labels):
        ax.plot(lengths, _mean_curve(all_metrics[lab], lengths, "H_norm"),
                marker=markers[i % len(markers)], linewidth=2, label=lab)
    ax.axvline(train_length, color='red', linestyle='--', alpha=0.7, label='train length')
    ax.set_xlabel("Sequence length (frames)")
    ax.set_ylabel("Normalized attention entropy  H / log N")
    ax.set_title("Encoder attention concentration vs. length\n(1.0 = uniform, 0.0 = delta)")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "lengthgen_Hnorm_summary.pdf", bbox_inches='tight')
    plt.savefig(Path(out_dir) / "lengthgen_Hnorm_summary.png", bbox_inches='tight', dpi=150)
    plt.close()

    # (2) N_eff summary (log scale)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, lab in enumerate(labels):
        ax.plot(lengths, _mean_curve(all_metrics[lab], lengths, "N_eff"),
                marker=markers[i % len(markers)], linewidth=2, label=lab)
    ax.axvline(train_length, color='red', linestyle='--', alpha=0.7, label='train length')
    ax.set_xlabel("Sequence length (frames)")
    ax.set_ylabel("Effective # attended tokens (exp H)")
    ax.set_title("Encoder effective attention spread vs. length")
    ax.set_yscale('log'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "lengthgen_Neff_summary.pdf", bbox_inches='tight')
    plt.savefig(Path(out_dir) / "lengthgen_Neff_summary.png", bbox_inches='tight', dpi=150)
    plt.close()

    # (3) per-variant per-layer heatmaps
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5), squeeze=False)
    for j, lab in enumerate(labels):
        pl = all_metrics[lab]
        layers = list(pl[lengths[0]].keys()) if pl.get(lengths[0]) else []
        mat = np.array([[pl[L][ln]["H_norm"] if pl.get(L) and ln in pl[L] else np.nan
                         for L in lengths] for ln in layers]) if layers else np.zeros((1, len(lengths)))
        ax = axes[0][j]
        im = ax.imshow(mat, aspect='auto', vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(range(len(lengths))); ax.set_xticklabels(lengths)
        ax.set_yticks(range(len(layers))); ax.set_yticklabels([f"L{i}" for i in range(len(layers))])
        ax.set_xlabel("Sequence length"); ax.set_title(f"{lab}: H/log N")
        if train_length in lengths:
            ax.axvline(lengths.index(train_length), color='red', linestyle='--', alpha=0.7)
    axes[0][0].set_ylabel("Attention layer")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Normalized entropy")
    plt.savefig(Path(out_dir) / "lengthgen_Hnorm_heatmap.pdf", bbox_inches='tight')
    plt.savefig(Path(out_dir) / "lengthgen_Hnorm_heatmap.png", bbox_inches='tight', dpi=150)
    plt.close()
