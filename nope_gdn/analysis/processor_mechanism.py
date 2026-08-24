"""Processor mechanism — recurrent gates & state vs length, for the two
RECURRENT variants:
  nope_gdn  -> GatedDeltaLayer (KDA):  decay = exp(g),  write gate = β = σ(b_proj)
  trecvit   -> RealLRU (RG-LRU):       decay = exp(-8·softplus(log_a)·σ(a_gate)),
                                       input gate = σ(input_gate)
Both carry a recurrent state over TIME (GDN: matrix S_t, ‖·‖_F; LRU: vector h_t,
‖·‖₂). Measured per recurrent layer and per sequence length: mean decay gate,
mean write/input gate, final recurrent-state norm, and the state-norm
trajectory over time. The ViT-RoPE variants have no recurrent processor state.
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


class GDNCapture:
    """GDN capture (nope_gdn) — decay = exp(g), gate = β, state = ‖S_t‖_F.
    Forces the sequential recurrence so the state trajectory is observable even
    when FLA's chunk kernel is available."""
    def __init__(self):
        self.decay = defaultdict(list)
        self.gate = defaultdict(list)
        self.state_norm_traj = defaultdict(list)
        self._orig_fwd = {}
        self._orig_seq = {}

    def attach(self, model):
        n = 0
        for name, module in model.named_modules():
            if type(module).__name__ == "GatedDeltaLayer":
                self._patch(module, name); n += 1
        return n

    def _patch(self, module, layer_name):
        self._orig_fwd[layer_name] = module.forward
        dec, gat = self.decay, self.gate

        def patched_fwd(x, state=None, use_parallel=None):
            B, L, _ = x.shape
            with torch.no_grad():
                if module.channel_wise_decay:
                    g = module.f_proj(x).view(B, L, module.num_heads, module.head_dim).float()
                    g = g + module.dt_bias.view(module.num_heads, module.head_dim)
                    g = -module.A_log.exp().view(1, 1, module.num_heads, 1) * F.softplus(g)
                    decay_val = g.exp()
                else:
                    decay_val = torch.sigmoid(module.a_proj(x))
                beta_val = torch.sigmoid(module.b_proj(x))
                if module.allow_neg_eigval:
                    beta_val = beta_val * 2.0
            dec[layer_name].append(decay_val.detach().cpu().flatten())
            gat[layer_name].append(beta_val.detach().cpu().flatten())
            return self._orig_fwd[layer_name](x, state=state, use_parallel=False)
        module.forward = patched_fwd

        self._orig_seq[layer_name] = module._sequential
        st = self.state_norm_traj

        def patched_seq(q, k, v, gate_or_alpha, beta, state, mode='kda'):
            B, L, H, D = q.shape
            if state is None:
                state = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
            I = torch.eye(D, device=q.device, dtype=q.dtype)
            outs, norms = [], torch.zeros(B, L)
            for t in range(L):
                q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]
                b_t = beta[:, t, :, None, None]
                householder = I - b_t * (k_t.unsqueeze(-1) @ k_t.unsqueeze(-2))
                if mode == 'kda':
                    state = (state * gate_or_alpha[:, t].exp().unsqueeze(-2)) @ householder
                else:
                    state = state @ (gate_or_alpha[:, t, :, None, None] * householder)
                state = state + b_t * (v_t.unsqueeze(-1) @ k_t.unsqueeze(-2))
                outs.append((state @ q_t.unsqueeze(-1)).squeeze(-1))
                with torch.no_grad():
                    norms[:, t] = state.flatten(2).norm(dim=-1).mean(dim=-1).cpu()
            st[layer_name].append(norms.detach())
            return torch.stack(outs, dim=1), state
        module._sequential = patched_seq

    def restore(self, model):
        for name, module in model.named_modules():
            if name in self._orig_fwd:
                module.forward = self._orig_fwd[name]
            if name in self._orig_seq:
                module._sequential = self._orig_seq[name]
        self._orig_fwd.clear(); self._orig_seq.clear()

    def summarize_per_layer(self):
        out = {}
        for layer in self.decay:
            decay = torch.cat(self.decay[layer]).numpy()
            gate = torch.cat(self.gate[layer]).numpy()
            traj = torch.cat(self.state_norm_traj[layer], dim=0)     # (samples, T)
            out[layer] = {"decay_mean": float(decay.mean()), "decay_std": float(decay.std()),
                          "gate_mean": float(gate.mean()), "gate_std": float(gate.std()),
                          "state_norm_final_mean": float(traj[:, -1].mean()),
                          "state_norm_traj_mean": traj.mean(dim=0).numpy().tolist()}
        return out


class LRUCapture:
    """LRU capture (trecvit) — decay = α = exp(log_alpha), gate = gate_x,
    state = ‖h_t‖₂."""
    def __init__(self):
        self.decay = defaultdict(list)
        self.gate = defaultdict(list)
        self.state_norm_traj = defaultdict(list)
        self._orig = {}

    def attach(self, model):
        n = 0
        for name, module in model.named_modules():
            if type(module).__name__ == "RealLRU":
                self._patch(module, name); n += 1
        return n

    def _patch(self, module, layer_name):
        self._orig[layer_name] = module.forward_chunk
        dec, gat, st = self.decay, self.gate, self.state_norm_traj

        def patched(x, lru_state=None, conv_state=None):
            with torch.no_grad():
                bn, T, _ = x.shape
                K = module.conv1d_temporal_width
                gate_x = torch.sigmoid(module.input_gate(x))
                gate_a = torch.sigmoid(module.a_gate(x))
                log_alpha = (-8.0 * F.softplus(module.log_a)) * gate_a
                alpha = torch.exp(log_alpha)
                dec[layer_name].append(alpha.detach().cpu().flatten())
                gat[layer_name].append(gate_x.detach().cpu().flatten())
                # faithful state-norm trajectory (replicates the RG-LRU scan)
                u = module.linear_x(x).transpose(1, 2)
                cs = torch.zeros(bn, module.lru_width, K - 1, device=x.device, dtype=u.dtype)
                u = module.conv1d(torch.cat([cs, u], dim=-1)).transpose(1, 2)
                mult = torch.sqrt(torch.clamp(1.0 - torch.exp(2.0 * log_alpha), min=1e-6))
                mult = torch.cat([torch.ones_like(mult[:, :1]), mult[:, 1:]], dim=1)  # start reset
                gated = mult * gate_x * u
                h = torch.zeros(bn, module.lru_width, device=x.device, dtype=gated.dtype)
                norms = []
                for t in range(T):
                    h = alpha[:, t] * h + gated[:, t]
                    norms.append(h.norm(dim=-1).mean().item())
                st[layer_name].append(norms)
            return self._orig[layer_name](x, lru_state, conv_state)
        module.forward_chunk = patched

    def restore(self, model):
        for name, module in model.named_modules():
            if name in self._orig:
                module.forward_chunk = self._orig[name]
        self._orig.clear()

    def summarize_per_layer(self):
        out = {}
        for layer in self.decay:
            decay = torch.cat(self.decay[layer]).numpy()
            gate = torch.cat(self.gate[layer]).numpy()
            traj = np.array(self.state_norm_traj[layer])            # (samples, T)
            avg = traj.mean(axis=0)
            out[layer] = {"decay_mean": float(decay.mean()), "decay_std": float(decay.std()),
                          "gate_mean": float(gate.mean()), "gate_std": float(gate.std()),
                          "state_norm_final_mean": float(avg[-1]),
                          "state_norm_traj_mean": avg.tolist()}
        return out


def _make_capture(model):
    """Pick the recurrence capture that matches this model."""
    types = {type(m).__name__ for m in model.modules()}
    if "GatedDeltaLayer" in types:
        return GDNCapture(), "GDN (exp g / β)"
    if "RealLRU" in types:
        return LRUCapture(), "RG-LRU (α / gate_x)"
    return None, None


def random_video_loader(length, n_samples, img_size, in_ch, device):
    """Random-tensor video source for smoke tests (shared signature with the
    length-generalization experiment)."""
    return torch.randn(n_samples, in_ch, length, img_size, img_size, device=device)


@torch.no_grad()
def _measure_proc_at_length(model, length, n_samples, batch_size, img_size,
                            in_ch, device, video_loader):
    cap, _ = _make_capture(model)
    if cap is None:
        return {}
    cap.attach(model)
    videos = video_loader(length, n_samples, img_size, in_ch, device)
    for i in range(0, videos.shape[0], batch_size):
        model(videos[i:i + batch_size])
    summary = cap.summarize_per_layer()
    cap.restore(model)
    if device == "cuda":
        torch.cuda.empty_cache()
    return summary


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


DEFAULT_PROC_VARIANTS = {
    "nope_gdn (global-pool)": {"variant": "nope_gdn", "ckpt": None},
    "trecvit":                {"variant": "trecvit",  "ckpt": None},
}


def run_processor_mechanism(variants=None, lengths=(8, 16, 24, 32, 48, 64, 72),
                            train_length=32, size="base", img_size=224,
                            n_samples=64, batch_size=1, video_loader=None,
                            out_dir="./figures_proc_mech", device=None,
                            model_overrides=None, make_plots=True):
    variants = variants or DEFAULT_PROC_VARIANTS
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    video_loader = video_loader or random_video_loader
    if video_loader is random_video_loader:
        print("⚠️  No video_loader — using RANDOM inputs (gate/state values are a "
              "smoke test, not real data). Pass a loader for publishable numbers.")

    all_metrics = {}
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
            _, kind = _make_capture(model)
            if kind is None:
                print(f"[skip] {label}: no recurrent (GDN/LRU) layer to profile.")
                continue
            if spec.get("ckpt"):
                ep, acc = _load_ckpt(model, spec["ckpt"], device)
                print(f"[{label}] {kind}; loaded ckpt epoch={ep} best_acc={acc}")
            else:
                print(f"[{label}] {kind}; random-init weights")
            per_len = {}
            for L in lengths:
                m = _measure_proc_at_length(model, L, n_samples, batch_size,
                                            cfg.model.img_size, in_ch, device, video_loader)
                per_len[L] = m
                if m:
                    d = np.mean([r["decay_mean"] for r in m.values()])
                    g = np.mean([r["gate_mean"] for r in m.values()])
                    s = np.mean([r["state_norm_final_mean"] for r in m.values()])
                    print(f"    {label:24s} L={L:>3}: {len(m)} layers | "
                          f"decay={d:.3f} gate={g:.3f} ‖state‖={s:.3f}")
            all_metrics[label] = per_len
        except Exception as e:
            print(f"[skip] {label}: {type(e).__name__}: {e}")
        finally:
            try: del model
            except Exception: pass
            if device == "cuda":
                torch.cuda.empty_cache()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "processor_mechanism_raw.json", "w") as f:
        json.dump({"lengths": list(lengths), "train_length": train_length,
                   "metrics": {lab: {str(L): m for L, m in d.items()}
                               for lab, d in all_metrics.items()}}, f, indent=2)
    _print_proc_summary(all_metrics, lengths)
    if make_plots:
        try:
            _plot_processor_mechanism(all_metrics, lengths, train_length, out_dir)
            print(f"\nFigures + raw JSON saved to: {out_dir}")
        except Exception as e:
            print(f"  plotting skipped: {type(e).__name__}: {e}")
    return all_metrics


def _layer_mean(per_len, lengths, key):
    return [float(np.mean([r[key] for r in per_len[L].values()])) if per_len.get(L) else float('nan')
            for L in lengths]


def _print_proc_summary(all_metrics, lengths):
    for lab, per in all_metrics.items():
        print(f"\n=== {lab}: processor recurrence vs length ===")
        print(f"{'L':>5} {'decay':>9} {'gate':>9} {'‖state‖_final':>15}")
        for L in lengths:
            if not per.get(L):
                continue
            d = np.mean([r['decay_mean'] for r in per[L].values()])
            g = np.mean([r['gate_mean'] for r in per[L].values()])
            s = np.mean([r['state_norm_final_mean'] for r in per[L].values()])
            print(f"{L:>5} {d:>9.4f} {g:>9.4f} {s:>15.4f}")


def _plot_processor_mechanism(all_metrics, lengths, train_length, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lengths = list(lengths)
    labels = list(all_metrics)
    markers = ['o', 's', '^', 'D']

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    def _panel(ax, key, ylabel, title, ylim=None, logy=False, normalize=False):
        for i, lab in enumerate(labels):
            y = _layer_mean(all_metrics[lab], lengths, key)
            if normalize and train_length in lengths:
                base = y[lengths.index(train_length)]
                y = [vv / base if base else float('nan') for vv in y]
            ax.plot(lengths, y, marker=markers[i % len(markers)], linewidth=2, label=lab)
        ax.axvline(train_length, color='red', linestyle='--', alpha=0.6)
        ax.set_xlabel("Sequence length (frames)"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        if ylim: ax.set_ylim(*ylim)
        if logy: ax.set_yscale('log')

    _panel(axes[0, 0], "decay_mean", "mean decay  (exp g / α)",
           "Decay gate vs length", ylim=(0, 1))
    _panel(axes[0, 1], "gate_mean", "mean write/input gate  (β / gate_x)",
           "Write / input gate vs length", ylim=(0, 1))
    _panel(axes[1, 0], "state_norm_final_mean", "‖state‖  (final)",
           "Final recurrent-state norm (raw)", logy=True)
    _panel(axes[1, 1], "state_norm_final_mean", "‖state‖ / ‖state‖@train",
           "Final state norm (relative to train length)", normalize=True)

    fig.suptitle("Processor mechanism: NoPE+GDN recurrence vs TRecViT RG-LRU", y=1.00)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "processor_mechanism_summary.pdf", bbox_inches='tight')
    plt.savefig(Path(out_dir) / "processor_mechanism_summary.png", bbox_inches='tight', dpi=150)
    plt.close()

    # state-norm trajectory: one panel per variant (state types differ in scale)
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5), squeeze=False)
    for j, lab in enumerate(labels):
        ax = axes[0][j]
        per = all_metrics[lab]
        for L in lengths:
            if not per.get(L):
                continue
            layers = list(per[L].keys())
            trajs = [per[L][ln]["state_norm_traj_mean"] for ln in layers]
            avg = np.mean(np.array(trajs), axis=0)
            ax.plot(np.arange(len(avg)), avg, alpha=0.85, label=f"L={L}")
        ax.set_xlabel("Time step t"); ax.set_ylabel("mean ‖state_t‖")
        ax.set_title(f"{lab}: state-norm trajectory"); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "processor_state_norm_traj.pdf", bbox_inches='tight')
    plt.savefig(Path(out_dir) / "processor_state_norm_traj.png", bbox_inches='tight', dpi=150)
    plt.close()
