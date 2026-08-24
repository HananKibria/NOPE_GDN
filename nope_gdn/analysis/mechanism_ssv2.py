"""Length-generalization mechanism analysis on real SSv2 frames — three-variant
(nope_gdn vs rope vs trecvit): position-based attention capture, GDN/LRU
recurrence capture, variable-length measurement, plotting, and the
variant-agnostic checkpoint loader (safe overlay-pattern EMA load) reused by
other experiments.
"""
import math
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from ..config import get_config
from ..data.ssv2 import SomethingSomethingV2
from ..models.factory import build_model

# Names of attention-bearing classes the capture recognizes
_ATTN_CLASS_NAMES = ("NoPEMultiheadAttention", "RoPEAttention", "_TRecViTSpatialBlock")


class AttentionCapture:
    """Capture post-softmax attention from selected layers.

    Position-based filtering: discover all attention-bearing modules in
    module-iteration order and slice based on `scope`:
      scope='encoder'   : first n_encoder layers
      scope='processor' : layers after the first n_encoder
      scope='all'       : all attention layers

    Recognized attention-bearing module classes:
      - NoPEMultiheadAttention  (nope_gdn encoder + 1 processor block)
      - RoPEAttention           (rope's all 16 blocks)
      - _TRecViTSpatialBlock    (trecvit's 12 spatial blocks; we patch the
                                 block itself because nn.MultiheadAttention
                                 doesn't expose pre-softmax weights cleanly)
    """
    def __init__(self, scope="encoder", n_encoder=12):
        self.scope = scope
        self.n_encoder = n_encoder
        self.captures = defaultdict(list)
        self._orig_forwards = {}

    def _select_layers(self, model):
        attn_modules = []
        for name, module in model.named_modules():
            if type(module).__name__ in _ATTN_CLASS_NAMES:
                attn_modules.append((name, module))
        if self.scope == "all":
            return attn_modules
        elif self.scope == "encoder":
            return attn_modules[:self.n_encoder]
        elif self.scope == "processor":
            return attn_modules[self.n_encoder:]
        else:
            raise ValueError(f"Unknown scope: {self.scope}")

    def attach(self, model):
        layers = self._select_layers(model)
        n_patched = 0
        for name, module in layers:
            cls_name = type(module).__name__
            if cls_name == "NoPEMultiheadAttention":
                self._patch_nope(module, name)
                n_patched += 1
            elif cls_name == "RoPEAttention":
                self._patch_rope(module, name)
                n_patched += 1
            elif cls_name == "_TRecViTSpatialBlock":
                self._patch_trecvit_spatial(module, name)
                n_patched += 1
        print(f"  Patched {n_patched} attention modules (scope='{self.scope}')")
        if n_patched > 0:
            print(f"    First: {layers[0][0]} ({type(layers[0][1]).__name__})")
            print(f"    Last:  {layers[-1][0]} ({type(layers[-1][1]).__name__})")
        return n_patched

    @staticmethod
    def _record_metrics(attn, layer_name, captures, do_diag=True):
        """Compute H_norm and N_eff from a (B, H_heads, N_q, N_k) attention tensor."""
        with torch.no_grad():
            N_tok = attn.shape[-1]
            log_attn = torch.log(attn.clamp(min=1e-10))
            H = -(attn * log_attn).sum(dim=-1)              # (B, H_heads, N_q)
            H_norm_val = (H / math.log(N_tok)).mean().item()
            N_eff_val = H.exp().mean().item()
            if do_diag and not captures[layer_name]:
                H_per_head = H.mean(dim=(0, 2))
                H_norm_per_head = H_per_head / math.log(N_tok)
                N_eff_per_head = H.exp().mean(dim=(0, 2))
                arr = H_norm_per_head.cpu().numpy()
                n_eff_arr = N_eff_per_head.cpu().numpy()
                print(f"  [diag N={N_tok}] {layer_name}")
                print(f"    H_norm per head: {np.array2string(arr, precision=3, separator=' ')}")
                print(f"    N_eff  per head: {np.array2string(n_eff_arr, precision=1, separator=' ')}")
        captures[layer_name].append({"H_norm": H_norm_val, "N_eff": N_eff_val})

    def _patch_nope(self, module, layer_name):
        orig = module.forward
        self._orig_forwards[layer_name] = orig
        captures = self.captures
        def patched(x, attn_mask=None):
            B, N, D = x.shape
            H_heads, d = module.num_heads, module.head_dim
            qkv = module.qkv_proj(x).reshape(B, N, 3, H_heads, d)
            q, k, v = qkv.unbind(dim=2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            attn_logits = (q @ k.transpose(-2, -1)) * module.scale
            if attn_mask is not None:
                if attn_mask.dim() == 2:   attn_mask = attn_mask[None, None]
                elif attn_mask.dim() == 3: attn_mask = attn_mask[:, None]
                attn_logits = attn_logits.masked_fill(attn_mask, float('-inf'))
            attn = attn_logits.softmax(dim=-1)
            AttentionCapture._record_metrics(attn, layer_name, captures)
            out = (attn @ v).transpose(1, 2).reshape(B, N, D)
            return module.out_proj(out)
        module.forward = patched

    def _patch_rope(self, module, layer_name):
        orig = module.forward
        self._orig_forwards[layer_name] = orig
        captures = self.captures
        rotate_half_fn = globals().get('rotate_half', None)
        if rotate_half_fn is None:
            def rotate_half_fn(x):
                x1, x2 = x.chunk(2, dim=-1)
                return torch.cat([-x2, x1], dim=-1)
        def patched(x, rope_cache=None, attn_mask=None):
            B, N, D = x.shape
            qkv = module.qkv_proj(x).reshape(B, N, 3, module.embed_dim)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            q = q.reshape(B, N, module.num_heads, module.head_dim).transpose(1, 2)
            k = k.reshape(B, N, module.num_heads, module.head_dim).transpose(1, 2)
            v = v.reshape(B, N, module.num_heads, module.head_dim).transpose(1, 2)
            if rope_cache is not None:
                cos, sin = rope_cache
                cos = cos.permute(1, 0, 2).unsqueeze(0).to(device=q.device, dtype=q.dtype)
                sin = sin.permute(1, 0, 2).unsqueeze(0).to(device=q.device, dtype=q.dtype)
                q = (q * cos) + (rotate_half_fn(q) * sin)
                k = (k * cos) + (rotate_half_fn(k) * sin)
            attn_logits = (q @ k.transpose(-2, -1)) * module.scale
            if attn_mask is not None:
                if attn_mask.dim() == 2: attn_mask = attn_mask[None, None]
                attn_logits = attn_logits.masked_fill(attn_mask, float('-inf'))
            attn = attn_logits.softmax(dim=-1)
            AttentionCapture._record_metrics(attn, layer_name, captures)
            out = (attn @ v).transpose(1, 2).reshape(B, N, D)
            return module.out_proj(out)
        module.forward = patched

    def _patch_trecvit_spatial(self, block, layer_name):
        """Patch _TRecViTSpatialBlock.forward to extract attention weights from
        its nn.MultiheadAttention by passing need_weights=True. The block is
        the unit we patch (not the inner attn module) because nn.MultiheadAttention
        only exposes attention weights through its return values."""
        orig = block.forward
        self._orig_forwards[layer_name] = orig
        captures = self.captures
        def patched(x):
            y = block.norm1(x)
            # Request unaveraged attention weights: (B, num_heads, L, L)
            attn_out, attn_weights = block.attn(
                y, y, y, need_weights=True, average_attn_weights=False
            )
            # attn_weights is post-softmax attention; record metrics
            AttentionCapture._record_metrics(attn_weights, layer_name, captures)
            x = x + attn_out
            x = x + block.mlp(block.norm2(x))
            return x
        block.forward = patched

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
            if not lst:
                continue
            H_norms = [d["H_norm"] for d in lst]
            N_effs = [d["N_eff"] for d in lst]
            out[layer] = {"H_norm": float(np.mean(H_norms)),
                          "N_eff":  float(np.mean(N_effs))}
        return out


class GDNCapture:
    """Monkey-patch GatedDeltaLayer to capture decay (g), beta, state norm trajectory."""
    def __init__(self):
        self.decay = defaultdict(list)
        self.beta = defaultdict(list)
        self.state_norm_traj = defaultdict(list)
        self._orig_fwd = {}
        self._orig_seq = {}

    def attach(self, model):
        n_patched = 0
        for name, module in model.named_modules():
            if type(module).__name__ == "GatedDeltaLayer":
                self._patch_layer(module, name)
                n_patched += 1
        print(f"  Patched {n_patched} GDN layers")
        return n_patched

    def _patch_layer(self, module, layer_name):
        orig_fwd = module.forward
        self._orig_fwd[layer_name] = orig_fwd
        decay_cap = self.decay
        beta_cap = self.beta

        def patched_fwd(x, state=None, use_parallel=None):
            B, L, _ = x.shape
            with torch.no_grad():
                if module.channel_wise_decay:
                    g = module.f_proj(x)
                    g = g.view(B, L, module.num_heads, module.head_dim).float()
                    g = g + module.dt_bias.view(module.num_heads, module.head_dim)
                    g = -module.A_log.exp().view(1, 1, module.num_heads, 1) * F.softplus(g)
                    decay_val = g.exp()
                else:
                    decay_val = torch.sigmoid(module.a_proj(x))
                beta_val = torch.sigmoid(module.b_proj(x))
                if module.allow_neg_eigval:
                    beta_val = beta_val * 2.0
            decay_cap[layer_name].append(decay_val.detach().cpu())
            beta_cap[layer_name].append(beta_val.detach().cpu())
            return orig_fwd(x, state=state, use_parallel=False)

        module.forward = patched_fwd

        orig_seq = module._sequential
        self._orig_seq[layer_name] = orig_seq
        state_cap = self.state_norm_traj

        def patched_sequential(q, k, v, gate_or_alpha, beta, state, mode='kda'):
            B, L, H, D = q.shape
            if state is None:
                state = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
            I = torch.eye(D, device=q.device, dtype=q.dtype)
            outputs = []
            norms = torch.zeros(B, L)
            for t in range(L):
                q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]
                b_t = beta[:, t, :, None, None]
                k_outer = k_t.unsqueeze(-1) @ k_t.unsqueeze(-2)
                householder = I - b_t * k_outer
                if mode == 'kda':
                    g_t = gate_or_alpha[:, t]
                    decay = g_t.exp()
                    state = (state * decay.unsqueeze(-2)) @ householder
                else:
                    a_t = gate_or_alpha[:, t, :, None, None]
                    state = state @ (a_t * householder)
                state = state + b_t * (v_t.unsqueeze(-1) @ k_t.unsqueeze(-2))
                outputs.append((state @ q_t.unsqueeze(-1)).squeeze(-1))
                with torch.no_grad():
                    norms[:, t] = state.flatten(2).norm(dim=-1).mean(dim=-1).cpu()
            state_cap[layer_name].append(norms.detach())
            return torch.stack(outputs, dim=1), state

        module._sequential = patched_sequential

    def restore(self, model):
        for name, module in model.named_modules():
            if name in self._orig_fwd:
                module.forward = self._orig_fwd[name]
            if name in self._orig_seq:
                module._sequential = self._orig_seq[name]
        self._orig_fwd.clear()
        self._orig_seq.clear()

    def clear(self):
        self.decay.clear(); self.beta.clear(); self.state_norm_traj.clear()

    def summarize_per_layer(self):
        out = {}
        for layer in self.decay:
            decay = torch.cat(self.decay[layer], dim=0).flatten().numpy()
            beta = torch.cat(self.beta[layer], dim=0).flatten().numpy()
            state_norm = torch.cat(self.state_norm_traj[layer], dim=0)
            out[layer] = {
                "decay_mean":  float(decay.mean()),
                "decay_std":   float(decay.std()),
                "decay_p05":   float(np.percentile(decay, 5)),
                "decay_p95":   float(np.percentile(decay, 95)),
                "beta_mean":   float(beta.mean()),
                "beta_std":    float(beta.std()),
                "beta_mid_frac": float(((beta > 0.3) & (beta < 0.7)).mean()),
                "state_norm_final_mean": float(state_norm[:, -1].mean().item()),
                "state_norm_final_std":  float(state_norm[:, -1].std().item()),
                "state_norm_traj_mean":  state_norm.mean(dim=0).numpy().tolist(),
            }
        return out


class LRUCapture:
    """Monkey-patch RealLRU.forward_chunk to capture decay alpha = exp(log_alpha),
    write gate gate_x, and the state norm trajectory ||h_t||.

    The trecvit recurrent block is RealLRU (one per LRUResidualBlock). For
    parity with GDNCapture's API:
      decay      <-> alpha = exp(log_alpha) ∈ (0, 1]
      beta       <-> gate_x ∈ (0, 1)        (the data-dependent input gate)
      state_norm <-> ||h_t|| over time
    """
    def __init__(self):
        self.decay = defaultdict(list)
        self.beta = defaultdict(list)
        self.state_norm_traj = defaultdict(list)
        self._orig_fwd = {}

    def attach(self, model):
        n_patched = 0
        for name, module in model.named_modules():
            if type(module).__name__ == "RealLRU":
                self._patch_layer(module, name)
                n_patched += 1
        print(f"  Patched {n_patched} LRU layers")
        return n_patched

    def _patch_layer(self, module, layer_name):
        orig_fwd = module.forward_chunk
        self._orig_fwd[layer_name] = orig_fwd
        decay_cap = self.decay
        beta_cap = self.beta
        state_cap = self.state_norm_traj

        def patched(x, lru_state=None, conv_state=None):
            # We replicate the forward sufficiently to capture state-norm trajectory
            # without altering the math. The output is what `orig_fwd` would have
            # produced (we call it at the end to avoid drift).
            bn, T, _ = x.shape
            K = module.conv1d_temporal_width

            # --- Compute the per-step gates exactly as in forward_chunk ---
            with torch.no_grad():
                u = module.linear_x(x)
                u_t_in = u.transpose(1, 2)
                cs = conv_state if conv_state is not None else torch.zeros(
                    bn, module.lru_width, K - 1, dtype=u_t_in.dtype, device=u_t_in.device)
                conv_input = torch.cat([cs, u_t_in], dim=-1)
                u_t = module.conv1d(conv_input)
                u = u_t.transpose(1, 2)

                gate_x = torch.sigmoid(module.input_gate(x))
                gate_a = torch.sigmoid(module.a_gate(x))
                a_real = -8.0 * F.softplus(module.log_a)
                log_alpha = a_real * gate_a
                alpha = torch.exp(log_alpha)
                multiplier = torch.sqrt(torch.clamp(
                    1.0 - torch.exp(2.0 * log_alpha), min=1e-6
                ))
                if lru_state is None and T > 0:
                    ones = torch.ones_like(multiplier[:, :1])
                    multiplier = torch.cat([ones, multiplier[:, 1:]], dim=1)
                gated_input = multiplier * gate_x * u

                # --- Recurrence to capture ||h_t|| ---
                if lru_state is None:
                    h = torch.zeros(
                        bn, module.lru_width, dtype=gated_input.dtype, device=gated_input.device)
                else:
                    h = lru_state
                norms = torch.zeros(bn, T)
                for t in range(T):
                    h = alpha[:, t] * h + gated_input[:, t]
                    norms[:, t] = h.norm(dim=-1).cpu()

            # Record analogues to GDN's decay/beta/state_norm
            decay_cap[layer_name].append(alpha.detach().cpu())
            beta_cap[layer_name].append(gate_x.detach().cpu())
            state_cap[layer_name].append(norms.detach())

            # Run the actual forward to keep model output correct
            return orig_fwd(x, lru_state, conv_state)

        module.forward_chunk = patched

    def restore(self, model):
        for name, module in model.named_modules():
            if name in self._orig_fwd:
                module.forward_chunk = self._orig_fwd[name]
        self._orig_fwd.clear()

    def clear(self):
        self.decay.clear(); self.beta.clear(); self.state_norm_traj.clear()

    def summarize_per_layer(self):
        out = {}
        for layer in self.decay:
            decay = torch.cat(self.decay[layer], dim=0).flatten().numpy()
            beta = torch.cat(self.beta[layer], dim=0).flatten().numpy()
            state_norm = torch.cat(self.state_norm_traj[layer], dim=0)
            out[layer] = {
                "decay_mean":  float(decay.mean()),
                "decay_std":   float(decay.std()),
                "decay_p05":   float(np.percentile(decay, 5)),
                "decay_p95":   float(np.percentile(decay, 95)),
                "beta_mean":   float(beta.mean()),
                "beta_std":    float(beta.std()),
                "beta_mid_frac": float(((beta > 0.3) & (beta < 0.7)).mean()),
                "state_norm_final_mean": float(state_norm[:, -1].mean().item()),
                "state_norm_final_std":  float(state_norm[:, -1].std().item()),
                "state_norm_traj_mean":  state_norm.mean(dim=0).numpy().tolist(),
            }
        return out


@torch.no_grad()
def measure_at_length(model, target_length, ssv2_root, ssv2_video, ssv2_anno,
                      n_samples=64, batch_size=2, device="cuda",
                      capture_recurrence=False,           # captures GDN or LRU
                      recurrence_kind="gdn",              # 'gdn' or 'lru'
                      capture_processor_attn=False,
                      n_encoder_layers=12):
    """Real variable-length sampling: build a fresh dataset with num_frames=target_length.

    Variant-aware:
      nope_gdn: capture_recurrence=True, recurrence_kind='gdn'
      rope:     capture_recurrence=False
      trecvit:  capture_recurrence=True, recurrence_kind='lru'
    """
    val_ds = SomethingSomethingV2(
        data_root=ssv2_root, split="validation",
        num_frames=target_length, img_size=224,
        video_dir=ssv2_video, anno_dir=ssv2_anno,
    )

    enc_cap = AttentionCapture(scope="encoder", n_encoder=n_encoder_layers)
    enc_cap.attach(model)

    proc_cap = None
    if capture_processor_attn:
        proc_cap = AttentionCapture(scope="processor", n_encoder=n_encoder_layers)
        proc_cap.attach(model)

    rec_cap = None
    if capture_recurrence:
        if recurrence_kind == "gdn":
            rec_cap = GDNCapture()
        elif recurrence_kind == "lru":
            rec_cap = LRUCapture()
        else:
            raise ValueError(f"Unknown recurrence_kind: {recurrence_kind}")
        rec_cap.attach(model)

    indices = np.linspace(0, len(val_ds) - 1, n_samples).astype(int)
    batch = []
    for i_idx, i in enumerate(indices):
        video, _ = val_ds[int(i)]
        batch.append(video)
        if len(batch) == batch_size or i_idx == len(indices) - 1:
            batch_t = torch.stack(batch).to(device)
            _ = model(batch_t)
            batch = []

    enc_metrics = enc_cap.compute_metrics_per_layer(); enc_cap.restore(model)
    proc_metrics = None
    if proc_cap is not None:
        proc_metrics = proc_cap.compute_metrics_per_layer(); proc_cap.restore(model)
    rec_metrics = None
    if rec_cap is not None:
        rec_metrics = rec_cap.summarize_per_layer(); rec_cap.restore(model)

    torch.cuda.empty_cache()
    return enc_metrics, proc_metrics, rec_metrics


def _ordered_layers(metrics_by_length):
    if not metrics_by_length:
        return []
    any_L = next(iter(metrics_by_length))
    names = list(metrics_by_length[any_L].keys())
    if not names:
        return []
    def key(n):
        parts = n.split('.')
        block_idx = next((int(p) for p in parts if p.isdigit()), -1)
        if any(tok in n for tok in ('encoder.spatial', 'backbone.encoder', 'encoder.blocks')):
            stage = 0
        elif any(tok in n for tok in ('encoder.temporal', 'processor', 'backbone.processor')):
            stage = 1
        else:
            stage = 2
        return (stage, block_idx)
    return sorted(names, key=key)


def plot_encoder_attention_3way(enc_nope, enc_rope, enc_trec, lengths, train_length, out_dir):
    """Plot encoder attention entropy across all available variants."""
    Path(out_dir).mkdir(exist_ok=True)

    def matrix(metrics, key):
        if not metrics:
            return None
        layers = _ordered_layers(metrics)
        if not layers:
            return None
        return np.array([[metrics[L][ln][key] for L in lengths] for ln in layers])

    Hn_n = matrix(enc_nope, "H_norm")
    Hn_r = matrix(enc_rope, "H_norm")
    Hn_t = matrix(enc_trec, "H_norm")
    Ne_n = matrix(enc_nope, "N_eff")
    Ne_r = matrix(enc_rope, "N_eff")
    Ne_t = matrix(enc_trec, "N_eff")

    if all(m is None for m in (Hn_n, Hn_r, Hn_t)):
        print("  WARNING: no encoder attention captured for any variant")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    if Hn_n is not None: ax.plot(lengths, Hn_n.mean(0), 'o-', label='NoPE+GDN encoder', linewidth=2)
    if Hn_r is not None: ax.plot(lengths, Hn_r.mean(0), 's-', label='RoPE encoder',     linewidth=2)
    if Hn_t is not None: ax.plot(lengths, Hn_t.mean(0), '^-', label='TRecViT spatial',  linewidth=2)
    ax.axvline(train_length, color='red', linestyle='--', alpha=0.7, label='train length')
    ax.set_xlabel("Sequence length (real-frame sampling)")
    ax.set_ylabel("Normalized attention entropy  H / log N")
    ax.set_title("Encoder attention concentration vs. length\n"
                 "(scale-free; 1.0 = uniform, 0.0 = delta)")
    ax.set_ylim(0, 1)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "encoder_attn_Hnorm_summary.pdf", bbox_inches='tight')
    plt.savefig(Path(out_dir) / "encoder_attn_Hnorm_summary.png", bbox_inches='tight', dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    if Ne_n is not None: ax.plot(lengths, Ne_n.mean(0), 'o-', label='NoPE+GDN encoder', linewidth=2)
    if Ne_r is not None: ax.plot(lengths, Ne_r.mean(0), 's-', label='RoPE encoder',     linewidth=2)
    if Ne_t is not None: ax.plot(lengths, Ne_t.mean(0), '^-', label='TRecViT spatial',  linewidth=2)
    ax.axvline(train_length, color='red', linestyle='--', alpha=0.7, label='train length')
    ax.set_xlabel("Sequence length"); ax.set_ylabel("Effective # attended tokens (exp H)")
    ax.set_title("Encoder effective attention spread vs. length")
    ax.set_yscale('log'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "encoder_attn_Neff_summary.pdf", bbox_inches='tight')
    plt.close()


def plot_recurrent_state(gdn_by_length, lru_by_length, lengths, train_length, out_dir):
    """Plot recurrent-state diagnostics for GDN (nope_gdn processor) and LRU (trecvit)."""
    Path(out_dir).mkdir(exist_ok=True)

    have_gdn = bool(gdn_by_length and any(gdn_by_length[L] for L in lengths))
    have_lru = bool(lru_by_length and any(lru_by_length[L] for L in lengths))

    if not (have_gdn or have_lru):
        print("  WARNING: no recurrent-state data captured")
        return

    def aggregate(by_length, field):
        layers = _ordered_layers(by_length)
        if not layers:
            return None
        return np.array([[by_length[L][ln][field] for L in lengths] for ln in layers])

    gdn_decay = aggregate(gdn_by_length, "decay_mean") if have_gdn else None
    gdn_beta  = aggregate(gdn_by_length, "beta_mean")  if have_gdn else None
    gdn_state = aggregate(gdn_by_length, "state_norm_final_mean") if have_gdn else None
    lru_decay = aggregate(lru_by_length, "decay_mean") if have_lru else None
    lru_beta  = aggregate(lru_by_length, "beta_mean")  if have_lru else None
    lru_state = aggregate(lru_by_length, "state_norm_final_mean") if have_lru else None

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    if gdn_decay is not None:
        ax.plot(lengths, gdn_decay.mean(0), 'o-', label='NoPE+GDN: decay (exp g)', linewidth=2)
    if lru_decay is not None:
        ax.plot(lengths, lru_decay.mean(0), '^-', label='TRecViT: decay (alpha)',  linewidth=2)
    ax.axvline(train_length, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel("Length"); ax.set_ylabel("Mean decay")
    ax.set_title("Recurrent decay rate")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    if gdn_beta is not None:
        ax.plot(lengths, gdn_beta.mean(0), 'o-', label='NoPE+GDN: write gate β', linewidth=2)
    if lru_beta is not None:
        ax.plot(lengths, lru_beta.mean(0), '^-', label='TRecViT: input gate x',  linewidth=2)
    ax.axvline(train_length, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel("Length"); ax.set_ylabel("Mean gate")
    ax.set_title("Recurrent write/input gate")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    if gdn_state is not None:
        ax.plot(lengths, gdn_state.mean(0), 'o-', label='NoPE+GDN: ||S_L||_F', linewidth=2)
    if lru_state is not None:
        ax.plot(lengths, lru_state.mean(0), '^-', label='TRecViT: ||h_L||',    linewidth=2)
    ax.axvline(train_length, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel("Length"); ax.set_ylabel("Final state norm")
    ax.set_title("Recurrent state norm at final step")
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle("Recurrent-mechanism comparison: GDN vs LRU", y=1.02)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "recurrent_comparison.pdf", bbox_inches='tight')
    plt.savefig(Path(out_dir) / "recurrent_comparison.png", bbox_inches='tight', dpi=150)
    plt.close()


def load_eval_model(variant, ckpt_path, num_classes=174, num_frames=32, device="cuda"):
    """Build a model variant, load weights (preferring EMA shadow via safe overlay),
    set eval mode, and return it. This is the variant-agnostic version of the
    bug-fixed loader."""
    cfg = get_config(variant=variant, size="base")
    cfg.model.num_classes = num_classes
    cfg.data.num_classes = num_classes
    cfg.data.num_frames = num_frames
    cfg.model.num_frames = num_frames
    model = build_model(variant, cfg.model).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ema = ckpt.get("ema_state")
    model_sd = model.state_dict()

    if ema and "shadow" in ema:
        shadow = ema["shadow"]
        loaded = 0
        for name in shadow:
            if name in model_sd:
                model_sd[name] = shadow[name]
                loaded += 1
        model.load_state_dict(model_sd)
        print(f"  [{variant}] EMA shadow: {loaded}/{len(model_sd)} tensors overlaid "
              f"(+ {len(model_sd)-loaded} buffers/params kept from fresh init)")
    else:
        model.load_state_dict(ckpt["model_state"])
        print(f"  [{variant}] Loaded raw model_state (no EMA found)")

    model.eval()                                                       # ← critical
    print(f"  [{variant}] epoch={ckpt.get('epoch','?')}, "
          f"best_acc={ckpt.get('best_acc','?')}, model.eval() set")
    return model


def count_attention(model, tag):
    attn = []
    for name, module in model.named_modules():
        if type(module).__name__ in _ATTN_CLASS_NAMES:
            attn.append((name, type(module).__name__))
    print(f"  {tag}: {len(attn)} total attention modules")
    if attn:
        print(f"    First: {attn[0][0]} ({attn[0][1]})")
        print(f"    Last:  {attn[-1][0]} ({attn[-1][1]})")
        types = defaultdict(int)
        for _, t in attn:
            types[t] += 1
        print(f"    Type counts: {dict(types)}")
