"""No-retrain ablations and loggers:

- processor_attn_temperature / eval_topk_at_length: is the processor-attention
  entropy collapse at long L causal? (soften SDPA logits, sweep the factor)
- ablate_gdn_recurrence / eval_topk_ablation: recurrence load-bearing test —
  frame-shuffle the input and zero the GDN blocks at training length.
- capture_gdn_decay / decay_histogram: where does per-channel decay live, and
  is the length-safe rotation band alpha in (0.37, 0.83) populated?
- ablate_temporal_rope / eval_acc: axial-RoPE temporal-PE-band ablation.
"""
import contextlib
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ..config import get_config
from ..data.ssv2 import SomethingSomethingV2, build_dataloaders


@contextlib.contextmanager
def processor_attn_temperature(model, factor, n_encoder=12):
    """Scale processor-attention SDPA logits by `factor` (factor<1 => softer).
    Only the processor NoPEMultiheadAttention block(s) are patched (the attn
    modules AFTER the first n_encoder=12 encoder ones). factor == 1.0 == baseline."""
    attn = [m for _, m in model.named_modules()
            if type(m).__name__ == "NoPEMultiheadAttention"]
    targets = attn[n_encoder:]                  # processor attention block(s) only
    saved = [(m, m.forward) for m in targets]

    def make_fwd(mod):
        bscale = mod.scale                      # default SDPA scale = head_dim ** -0.5

        def fwd(x, attn_mask=None):
            B, N, D = x.shape
            H, d = mod.num_heads, mod.head_dim
            qkv = mod.qkv_proj(x).reshape(B, N, 3, H, d)
            q, k, v = qkv.unbind(dim=2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            sdpa_mask = None
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    sdpa_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                    sdpa_mask = sdpa_mask.masked_fill(attn_mask, float("-inf"))
                else:
                    sdpa_mask = attn_mask
                if sdpa_mask.dim() == 2:
                    sdpa_mask = sdpa_mask[None, None, :, :]
                elif sdpa_mask.dim() == 3:
                    sdpa_mask = sdpa_mask[:, None, :, :]
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=sdpa_mask, dropout_p=0.0,
                is_causal=False, scale=bscale * factor)        # <-- temperature
            out = out.transpose(1, 2).reshape(B, N, D)
            out = mod.attn_dropout(out)
            return mod.out_proj(out)
        return fwd

    for m in targets:
        m.forward = make_fwd(m)
    try:
        yield len(targets)
    finally:
        for m, f in saved:
            m.forward = f


@torch.no_grad()
def eval_topk_at_length(model, L, ssv2_root, ssv2_video, ssv2_anno,
                        n_samples=2000, batch_size=4, num_workers=8, device=None):
    """Top-1/Top-5 on a fixed, evenly-spaced validation subset at num_frames=L.
    Same subset every call => apples-to-apples across temperature factors."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = SomethingSomethingV2(
        data_root=ssv2_root, split="validation",
        num_frames=L, img_size=224,
        video_dir=ssv2_video, anno_dir=ssv2_anno,
    )
    if n_samples and n_samples < len(ds):
        idx = np.linspace(0, len(ds) - 1, n_samples).astype(int)
        ds = Subset(ds, idx.tolist())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    model.eval()
    top1 = top5 = tot = 0
    use_amp = torch.cuda.is_available()
    for videos, targets in loader:
        videos = videos.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(videos)
        _, pred = logits.float().topk(5, dim=1)
        correct = pred.eq(targets[:, None])
        top1 += correct[:, :1].any(1).sum().item()
        top5 += correct.any(1).sum().item()
        tot += videos.size(0)
    return 100.0 * top1 / tot, 100.0 * top5 / tot, tot


@contextlib.contextmanager
def ablate_gdn_recurrence(model):
    """Zero every GDN temporal block's output (identity on its residual), removing
    recurrence + short-conv mixing while keeping the encoder, the processor attention
    block, all MLPs, and the head intact. Restores forwards on exit."""
    gdn_blocks = [b for b in model.backbone.processor_blocks
                  if getattr(b, "block_type", None) == "gdn"]

    def zero_fwd(x, *args, **kwargs):
        return torch.zeros_like(x), None

    saved = [(b, b.layer.forward) for b in gdn_blocks]
    for b in gdn_blocks:
        b.layer.forward = zero_fwd
    try:
        yield len(gdn_blocks)
    finally:
        for b, f in saved:
            b.layer.forward = f


@torch.no_grad()
def eval_topk_ablation(model, L, ssv2_root, ssv2_video, ssv2_anno,
                       n_samples=2000, batch_size=4, num_workers=8,
                       device=None, shuffle_frames=False, seed=0):
    """Top-1/5 on a fixed evenly-spaced validation subset at num_frames=L.
    shuffle_frames=True applies an independent temporal permutation per clip."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = SomethingSomethingV2(data_root=ssv2_root, split="validation",
                              num_frames=L, img_size=224,
                              video_dir=ssv2_video, anno_dir=ssv2_anno)
    if n_samples and n_samples < len(ds):
        idx = np.linspace(0, len(ds) - 1, n_samples).astype(int)
        ds = Subset(ds, idx.tolist())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    top1 = top5 = tot = 0
    use_amp = torch.cuda.is_available()
    for videos, targets in loader:
        if shuffle_frames:                       # videos: (B, C, T, H, W)
            T = videos.shape[2]
            for bi in range(videos.shape[0]):
                perm = torch.randperm(T, generator=gen)
                videos[bi] = videos[bi][:, perm]
        videos = videos.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(videos)
        _, pred = logits.float().topk(5, dim=1)
        correct = pred.eq(targets[:, None])
        top1 += correct[:, :1].any(1).sum().item()
        top5 += correct.any(1).sum().item()
        tot += videos.size(0)
    return 100.0 * top1 / tot, 100.0 * top5 / tot, tot


@contextlib.contextmanager
def capture_gdn_decay(model, max_per_call=20000):
    """Intercept g (log-decay) into every GatedDeltaLayer._chunkwise_channelwise and
    store alpha = exp(g), subsampled to keep memory bounded. Restores on exit."""
    layers = [(n, m) for n, m in model.named_modules()
              if type(m).__name__ == "GatedDeltaLayer"
              and hasattr(m, "_chunkwise_channelwise")]
    store = {n: [] for n, _ in layers}
    saved = []
    for n, m in layers:
        orig = m._chunkwise_channelwise
        saved.append((m, orig))
        def make(orig_fn, name):
            def wrap(q, k, v, g, beta, state):
                a = g.detach().float().exp().reshape(-1)
                if a.numel() > max_per_call:
                    sel = torch.randint(0, a.numel(), (max_per_call,), device=a.device)
                    a = a[sel]
                store[name].append(a.cpu())
                return orig_fn(q, k, v, g, beta, state)
            return wrap
        m._chunkwise_channelwise = make(orig, n)
    try:
        yield store
    finally:
        for m, orig in saved:
            m._chunkwise_channelwise = orig


def _horizon(a):
    a = float(min(max(a, 1e-6), 0.999999))
    return -1.0 / np.log(a)


def _bands(a):
    return ((a < 0.37).mean() * 100,
            (((a >= 0.37) & (a <= 0.83)).mean()) * 100,
            (a > 0.83).mean() * 100)


def _report_layer(name, a):
    p50 = np.percentile(a, 50)
    fast, band, slow = _bands(a)
    short = name.replace("backbone.", "")
    print(f"    {short:<40} mean={a.mean():.3f} p50={p50:.3f}  "
          f"fast<0.37={fast:4.1f}%  SAFE={band:4.1f}%  slow>0.83={slow:4.1f}%")


def _report_pooled(a, L):
    ps = np.percentile(a, [1, 5, 25, 50, 75, 95, 99])
    fast, band, slow = _bands(a)
    print(f"  POOLED @ L={L}  (n={a.size:,})")
    print(f"    percentiles  p01={ps[0]:.3f} p05={ps[1]:.3f} p25={ps[2]:.3f} "
          f"p50={ps[3]:.3f} p75={ps[4]:.3f} p95={ps[5]:.3f} p99={ps[6]:.3f}")
    print(f"    horizon(tok) p50={_horizon(ps[3]):.2f}  p95={_horizon(ps[5]):.2f}  "
          f"p99={_horizon(ps[6]):.2f}")
    print(f"    bands        fast a<0.37 = {fast:5.1f}%   "
          f"SAFE 0.37-0.83 = {band:5.1f}%   slow a>0.83 = {slow:5.1f}%")
    hist, _ = np.histogram(a, bins=20, range=(0.0, 1.0))
    mx = max(int(hist.max()), 1)
    print("    histogram (alpha 0 -> 1):")
    for bi in range(20):
        lo, hi = bi / 20.0, (bi + 1) / 20.0
        bar = "#" * int(38 * hist[bi] / mx)
        tag = "  <- safe band" if (lo >= 0.35 and hi <= 0.85) else ""
        print(f"      [{lo:.2f}-{hi:.2f}] {int(hist[bi]):9d} {bar}{tag}")


@torch.no_grad()
def decay_histogram(model, lengths=(16, 72), n_samples=96, batch_size=4,
                    ssv2_root=None, ssv2_video=None, ssv2_anno=None, device=None):
    """Per-channel decay alpha = exp(g) from every GatedDeltaLayer at the given
    lengths; reports percentiles, memory horizons, and fast / SAFE / slow band
    fractions + a pooled histogram.

    Band rationale (training length T'=16 tokens): alpha<0.37 -> no room for an
    oscillation (horizon<1 tok); alpha>0.83 -> kernel still alive past train
    length -> phase extrapolates (RoPE-like). Only (0.37, 0.83) is length-safe
    for rotation."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ssv2_root  = ssv2_root  or globals().get("ssv2_root",  "/content/ssv2")
    ssv2_video = ssv2_video or globals().get("ssv2_video", "20bn-something-something-v2")
    ssv2_anno  = ssv2_anno  or globals().get("ssv2_anno",  "annotations")
    model.eval()
    for L in lengths:
        ds = SomethingSomethingV2(data_root=ssv2_root, split="validation",
                                  num_frames=L, img_size=224,
                                  video_dir=ssv2_video, anno_dir=ssv2_anno)
        idx = np.linspace(0, len(ds) - 1, n_samples).astype(int)
        with capture_gdn_decay(model) as store:
            batch = []
            for j, i in enumerate(idx):
                v, _ = ds[int(i)]
                batch.append(v)
                if len(batch) == batch_size or j == len(idx) - 1:
                    model(torch.stack(batch).to(device))
                    batch = []
            per_layer = {n: torch.cat(v).numpy() for n, v in store.items() if v}
        print(f"\n================ decay alpha = exp(g) @ L={L} ================")
        if not per_layer:
            print("  (no decay captured -- model not using the channel-wise chunk path?)")
            continue
        for n in sorted(per_layer):
            _report_layer(n, per_layer[n])
        _report_pooled(np.concatenate(list(per_layer.values())), L)

    print("\nGate-2 read-out (rotary KDA):")
    print("  SAFE-band fraction sizeable (>~15-20%) -> real population to host length-safe")
    print("     rotation -> prototype rotary on those mid-decay channels (omega ~ -ln(alpha)).")
    print("  bimodal (big fast + big slow, thin middle) -> band thin; phase has little")
    print("     length-safe room -> reconsider before any kernel work.")
    print("  Also: compare L=16 vs L=72 -- the distribution should stay put (length-stable).")


# Defaults for eval_acc (EDIT DATA_ROOT to your SSv2 root)
DATA_ROOT = "/content/ssv2" if os.path.exists("/content/ssv2") else "/Users/hanan/Downloads/ssv2"
DEV       = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available() else "cpu")
N_BATCHES = 120


@contextlib.contextmanager
def ablate_temporal_rope(model):
    """Zero inv_freq_t on every AxialMixedRoPE3D -> temporal rotation = identity.
    (Spatial H/W RoPE intact; same weights, no retrain.)"""
    saved = []
    for m in model.modules():
        if type(m).__name__ == "AxialMixedRoPE3D" and hasattr(m, "inv_freq_t"):
            saved.append((m, m.inv_freq_t.clone())); m.inv_freq_t.zero_()
    try:
        yield len(saved)
    finally:
        for m, v in saved: m.inv_freq_t.copy_(v)

@torch.no_grad()
def eval_acc(model, L, n_batches=N_BATCHES, batch=8):
    cfg = get_config("axial_rope", "base"); cfg.data.data_root = DATA_ROOT
    cfg.data.num_frames = L; cfg.model.num_frames = L
    _, val_loader = build_dataloaders(cfg.data, batch_size=batch)   # val shuffle=False
    top1 = top5 = tot = 0
    for bi, (vids, tgts) in enumerate(val_loader):
        if bi >= n_batches: break
        vids, tgts = vids.to(DEV), tgts.to(DEV)
        logits = model(vids)
        t5 = logits.topk(5, dim=1).indices
        top1 += (t5[:, 0] == tgts).sum().item()
        top5 += (t5 == tgts[:, None]).any(1).sum().item()
        tot  += vids.size(0)
    return 100 * top1 / tot, 100 * top5 / tot, tot
