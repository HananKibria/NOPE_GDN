"""Streaming / cached online inference for the NoPE + GDN (vanilla) backbone.

Runs the model incrementally over temporal chunks, carrying each GDN block's
recurrent state AND its depthwise short-conv state across chunks. Memory is
O(1) in the number of chunks while remaining numerically faithful to the
full-clip forward logits = model(video). Requires gdn_temporal_only=True,
bidirectional=False and factorized_attention=True (the default config).
"""
import time

import torch


def _stream_preconditions(model):
    """Validate that the model supports faithful streaming and return the
    backbone. Raises AssertionError with an actionable message otherwise."""
    bb = model.backbone if hasattr(model, "backbone") else model
    assert getattr(bb, "gdn_temporal_only", False), (
        "streaming requires gdn_temporal_only=True (GDN scans time per spatial "
        "location). Build the model with gdn_temporal_only=True.")
    assert not getattr(bb, "bidirectional", False), (
        "streaming is not defined for bidirectional GDN (the backward branch "
        "needs future frames). Use bidirectional=False.")
    assert getattr(bb, "factorized_attention", True), (
        "streaming is only faithful with factorized (spatial-only) attention; "
        "a global-attention block would attend within the current chunk only.")
    return bb


def _temporal_tubelet(bb):
    return bb.encoder.tubelet_embed.tubelet_size[0]


@torch.no_grad()
def streaming_inference(model, video, chunk_frames=16, return_states=False):
    """Full-clip-faithful logits computed by streaming over temporal chunks.

    Args:
        model:        NoPEGDNClassifier (gdn_temporal_only=True, bidirectional=False)
        video:        (B, C, T, H, W)
        chunk_frames: frames per chunk. Must be a positive multiple of the
                      temporal tubelet size so chunk edges fall on tubelet
                      boundaries.
        return_states: if True, also return the final list of per-block
                      (recurrent_state, conv_state) tuples.

    Returns:
        logits (B, num_classes) — equal to model(video) up to numerical error.
        (optionally) the final streaming states.
    """
    model.eval()
    bb = _stream_preconditions(model)
    head = model.head
    B, C, T, H, W = video.shape
    t_tub = _temporal_tubelet(bb)
    assert chunk_frames > 0 and chunk_frames % t_tub == 0, (
        f"chunk_frames ({chunk_frames}) must be a positive multiple of the "
        f"temporal tubelet size ({t_tub}).")

    # Head-exact, constant-memory accumulation.
    #  VideoClassificationHead: logits = fc(dropout(mean_n LayerNorm(feat_n))).
    #  LayerNorm is per token, so the mean over all tokens equals
    #  (sum over chunks of LayerNorm(feat).sum(1)) / (total tokens).
    mean_pool = head.__class__.__name__ == "VideoClassificationHead"

    states = None
    run_sum, n_tok = None, 0     # mean-pool path
    feats = []                   # generic fallback path
    for s in range(0, T, chunk_frames):
        chunk = video[:, :, s:s + chunk_frames]
        if chunk.shape[2] < t_tub:          # drop trailing frames < one tubelet
            break
        f, states = bb.forward_chunk(chunk, states)     # (B, N_chunk, D)
        if mean_pool:
            normed = head.norm(f)
            run_sum = normed.sum(1) if run_sum is None else run_sum + normed.sum(1)
            n_tok += f.shape[1]
        else:
            feats.append(f)

    if mean_pool:
        logits = head.fc(head.dropout(run_sum / n_tok))
    else:
        # Any other head (e.g. TemporalPoolingHead): concatenate features and
        # apply the head once. Correct, but not constant-memory.
        allf = torch.cat(feats, dim=1)
        try:
            logits = head(allf, S=model.spatial_tokens)
        except TypeError:
            logits = head(allf)

    return (logits, states) if return_states else logits


@torch.no_grad()
def streaming_predictions(model, video, chunk_frames=16):
    """Online prediction trajectory: the model's logits after each chunk, using
    only the frames seen so far (constant memory, mean-pool head).

    Returns a list of (frames_seen, logits (B, num_classes)) — one entry per chunk.
    Mirrors a live/streaming deployment where the prediction is refined as new
    frames arrive. Requires the mean-pool VideoClassificationHead.
    """
    model.eval()
    bb = _stream_preconditions(model)
    head = model.head
    assert head.__class__.__name__ == "VideoClassificationHead", (
        "streaming_predictions needs the mean-pool VideoClassificationHead; "
        "for other heads use streaming_inference (full-clip logits).")
    B, C, T, H, W = video.shape
    t_tub = _temporal_tubelet(bb)
    assert chunk_frames > 0 and chunk_frames % t_tub == 0

    states, run_sum, n_tok, traj = None, None, 0, []
    for s in range(0, T, chunk_frames):
        chunk = video[:, :, s:s + chunk_frames]
        if chunk.shape[2] < t_tub:
            break
        f, states = bb.forward_chunk(chunk, states)
        normed = head.norm(f)
        run_sum = normed.sum(1) if run_sum is None else run_sum + normed.sum(1)
        n_tok += f.shape[1]
        traj.append((s + chunk.shape[2], head.fc(head.dropout(run_sum / n_tok))))
    return traj


@torch.no_grad()
def verify_streaming_fidelity(model, video, chunk_frames=16, verbose=True):
    """Streaming vs full-clip inference. Returns max/mean abs logit diff and
    top-1 agreement (%); the streaming result should match model(video)."""
    model.eval()
    full = model(video)
    stream = streaming_inference(model, video, chunk_frames)
    max_d = (full - stream).abs().max().item()
    mean_d = (full - stream).abs().mean().item()
    agree = (full.argmax(-1) == stream.argmax(-1)).float().mean().item() * 100.0
    res = {"chunk_frames": chunk_frames, "max_abs_logit_diff": max_d,
           "mean_abs_logit_diff": mean_d, "top1_agreement_pct": agree}
    if verbose:
        if max_d < 1e-3:
            verdict = "numerically exact (< 1e-3)"
        elif agree >= 100.0:
            verdict = "predictions match (diff within bf16 / FLA-kernel tolerance)"
        else:
            verdict = ("MISMATCH — check config "
                       "(gdn_temporal_only=True, bidirectional=False, factorized_attention=True)")
        print("=" * 60)
        print(f"Streaming vs Full  (chunk_frames={chunk_frames})")
        print("=" * 60)
        print(f"  Max  abs logit diff: {max_d:.3e}")
        print(f"  Mean abs logit diff: {mean_d:.3e}")
        print(f"  Top-1 agreement:     {agree:.1f}%")
        print(f"  Verdict:             {verdict}")
        print("=" * 60)
    return res


@torch.no_grad()
def benchmark_streaming(model, video, chunk_frames=16, warmup=1, iters=3):
    """Cached streaming (state carry) vs prefix-recompute (re-encode the whole
    prefix each step). Returns mean seconds for each, the speedup, and — on CUDA
    — peak memory for a single full forward vs the streaming pass."""
    model.eval()
    bb = _stream_preconditions(model)
    B, C, T, H, W = video.shape
    t_tub = _temporal_tubelet(bb)
    cuda = video.is_cuda

    def run_stream():
        states = None
        for s in range(0, T, chunk_frames):
            ch = video[:, :, s:s + chunk_frames]
            if ch.shape[2] < t_tub:
                break
            _, states = bb.forward_chunk(ch, states)

    def run_recompute():
        for e in range(chunk_frames, T + 1, chunk_frames):
            bb(video[:, :, :e])

    def _sync():
        if cuda:
            torch.cuda.synchronize()

    for _ in range(warmup):
        run_stream(); run_recompute()
    _sync()

    def _time(fn):
        _sync(); t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        _sync()
        return (time.perf_counter() - t0) / iters

    ts, tr = _time(run_stream), _time(run_recompute)
    out = {"streaming_s": ts, "recompute_s": tr,
           "speedup": (tr / ts) if ts > 0 else float("nan")}
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        model(video); torch.cuda.synchronize()
        out["mem_full_gb"] = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        run_stream(); torch.cuda.synchronize()
        out["mem_stream_gb"] = torch.cuda.max_memory_allocated() / 1e9
    print(f"Streaming: {ts*1e3:.1f} ms  |  Prefix-recompute: {tr*1e3:.1f} ms  "
          f"|  Speedup: {out['speedup']:.1f}x")
    if cuda:
        print(f"Peak mem  full: {out['mem_full_gb']:.2f} GB  |  "
              f"streaming: {out['mem_stream_gb']:.2f} GB")
    return out
