"""Speed benchmark — vanilla NoPE+GDN vs TRecViT vs Axial-RoPE vs VideoRoPE.

Pure wall-clock speed: parameters, latency, throughput and peak GPU memory.
No fvcore — its FlopCountAnalysis cannot trace FLA's custom Triton kernels, so
we keep FLA enabled and time real forward passes instead. Autocast (bf16/fp16)
on CUDA to reflect deployment; plain fp32 on CPU for a smoke test.
"""
import time
import gc

import torch

from ..config import get_config
from ..models.factory import build_model
from ..models.gated_delta import FLA_AVAILABLE

# label -> build_model variant string
BENCH_VARIANTS = {
    "nope_gdn (vanilla)": "nope_gdn",    # NoPE + Gated DeltaNet (config default: gdn_temporal_only)
    "trecvit":            "trecvit",     # TRecViT: LRU temporal mixer + factorized ViT
    "axial_rope":         "axial_rope",  # ViT + 3D axial RoPE
    "video_rope":         "rope",        # ViT + VideoRoPE (Wei et al., ICML 2025)
}


def _fwd(model, dummy, cuda, amp_dtype):
    if cuda:
        with torch.autocast("cuda", dtype=amp_dtype):
            return model(dummy)
    return model(dummy)


@torch.no_grad()
def _time_forward(model, dummy, cuda, amp_dtype, warmup, iters):
    """Mean seconds per forward pass (batch included)."""
    for _ in range(warmup):
        _fwd(model, dummy, cuda, amp_dtype)
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _fwd(model, dummy, cuda, amp_dtype)
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


@torch.no_grad()
def benchmark_speed(variants=BENCH_VARIANTS, size="base",
                    frame_counts=(16, 32, 64), batch_size=4,
                    img_size=None, warmup=3, iters=10,
                    amp_dtype=None, device=None, verbose=True):
    """Wall-clock speed benchmark across model variants and frame counts.

    A fresh model is built for every (variant, frame_count) — matching how the
    variable-frame benchmark rebuilds per length — so positional structures and
    token counts are correct at each length.

    Returns a list of per-(variant, frames) result dicts.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cuda = (device == "cuda")
    if amp_dtype is None:
        amp_dtype = torch.bfloat16 if (cuda and torch.cuda.is_bf16_supported()) else torch.float16

    print(f"Device={device}  size={size}  batch={batch_size}  "
          f"amp={amp_dtype if cuda else 'fp32 (cpu)'}  warmup={warmup} iters={iters}")

    rows = []
    for nf in frame_counts:
        for label, variant in variants.items():
            cfg = get_config(variant=variant, size=size)
            cfg.model.num_frames = nf
            if img_size is not None:
                cfg.model.img_size = img_size
            H = cfg.model.img_size
            model = dummy = None
            try:
                model = build_model(variant, cfg.model).to(device).eval()
                params_m = sum(p.numel() for p in model.parameters()) / 1e6
                dummy = torch.randn(batch_size, cfg.model.in_channels, nf, H, H,
                                    device=device)
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                lat = _time_forward(model, dummy, cuda, amp_dtype, warmup, iters)
                mem_gb = (torch.cuda.max_memory_allocated() / 1e9) if cuda else float("nan")
                rows.append(dict(frames=nf, variant=label, params_m=params_m,
                                 latency_ms=lat * 1e3, throughput=batch_size / lat,
                                 mem_gb=mem_gb, ok=True))
            except Exception as e:
                rows.append(dict(frames=nf, variant=label, ok=False,
                                 error=f"{type(e).__name__}: {e}"))
                if verbose:
                    print(f"  [skip] {label:20s} @ {nf:>3}f  ->  {type(e).__name__}: {e}")
            finally:
                del model, dummy
                gc.collect()
                if cuda:
                    torch.cuda.empty_cache()

    _print_speed_tables(rows, frame_counts, list(variants), cuda)
    return rows


def _print_speed_tables(rows, frame_counts, labels, cuda):
    memcol = "PeakMem(GB)" if cuda else "PeakMem"
    for nf in frame_counts:
        print("\n" + "=" * 84)
        print(f"Frames = {nf}")
        print("-" * 84)
        print(f"{'Variant':22} {'Params(M)':>10} {'Latency(ms)':>12} "
              f"{'Vids/s':>9} {memcol:>12} {'Speedup':>9}")
        print("-" * 84)
        ref = next((r for r in rows if r['frames'] == nf and r['ok']
                    and r['variant'].startswith('nope_gdn')), None)
        ref_lat = ref['latency_ms'] if ref else None
        for label in labels:
            r = next((x for x in rows if x['frames'] == nf and x['variant'] == label), None)
            if r is None:
                continue
            if not r['ok']:
                print(f"{label:22} {'—':>10} {'FAILED':>12} {'—':>9} {'—':>12} {'—':>9}")
                continue
            spd = (ref_lat / r['latency_ms']) if ref_lat else float('nan')
            mem = f"{r['mem_gb']:.2f}" if cuda else "n/a"
            print(f"{label:22} {r['params_m']:>10.2f} {r['latency_ms']:>12.1f} "
                  f"{r['throughput']:>9.1f} {mem:>12} {spd:>8.2f}x")
    print("=" * 84)
    print("Speedup = nope_gdn latency / variant latency  (>1 means faster than NoPE+GDN)")


if __name__ == "__main__":
    # Full benchmark on GPU, tiny smoke test on CPU.
    if torch.cuda.is_available():
        _ = benchmark_speed(size="base", frame_counts=(16, 32, 64),
                            batch_size=4, img_size=224, warmup=3, iters=10)
    elif globals().get("FLA_AVAILABLE", False):
        print("⚠️  FLA is installed but no CUDA device — skipping CPU smoke test "
              "(FLA's Triton kernels need a GPU).")
        print("   -> call benchmark_speed(size='base', img_size=224) on a GPU runtime.")
    else:
        print("⚠️  No CUDA detected — running a tiny CPU smoke test only "
              "(use a GPU with FLA installed for real numbers).")
        _ = benchmark_speed(size="tiny", frame_counts=(8, 16),
                            batch_size=1, img_size=64, warmup=1, iters=3)
