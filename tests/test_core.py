"""3D NoPE + Gated DeltaNet video backbone — core tests.

Runs under pytest (test_* functions) or standalone:
    python tests/test_core.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from nope_gdn.models.backbone import NoPEGDNVideoBackbone, NoPEMultiheadAttention
from nope_gdn.models.gated_delta import GatedDeltaLayer
from nope_gdn.models.trecvit import TRecViTClassifier, RealLRU
from nope_gdn.models.factory import build_model

torch.manual_seed(42)

passed = 0
failed = 0
total_tests = 0

def check(cond, name, detail=""):
    global passed, failed, total_tests
    total_tests += 1
    if cond:
        passed += 1
        print(f"  ✅ {name}" + (f" — {detail}" if detail else ""))
    else:
        failed += 1
        print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))


def test_full_backbone_forward_shapes():
    print(f"\n{'─'*60}")
    print("[Test 1] Full backbone — forward pass shapes")
    print(f"{'─'*60}")

    backbone = NoPEGDNVideoBackbone(
        img_size=224, num_frames=16, tubelet_size=(2, 16, 16),
        encoder_dim=128, encoder_depth=2, encoder_heads=4,
        processor_dim=128, processor_depth=4, processor_heads=4,
        chunk_size=32, dropout=0.0
    )
    backbone.eval()

    B = 2
    video = torch.randn(B, 3, 16, 224, 224)

    with torch.no_grad():
        features = backbone(video)
        features2, enc_features = backbone(video, return_encoder_features=True)

    N = backbone.encoder.tubelet_embed.num_patches
    check(features.shape == (B, N, 128), "output shape", f"{features.shape}")
    check(enc_features.shape == (B, N, 128), "encoder features shape", f"{enc_features.shape}")
    check(torch.allclose(features, features2, atol=1e-6),
          "return_encoder_features doesn't change main output")

    params = sum(p.numel() for p in backbone.parameters()) / 1e6
    print(f"  Model: {params:.1f}M params, {N} tokens")


def test_hybrid_3to1_ratio():
    print(f"\n{'─'*60}")
    print("[Test 2] 3:1 hybrid ratio in processor")
    print(f"{'─'*60}")

    backbone = NoPEGDNVideoBackbone(
        img_size=224, num_frames=16, tubelet_size=(2, 16, 16),
        encoder_dim=128, encoder_depth=2, encoder_heads=4,
        processor_dim=128, processor_depth=4, processor_heads=4,
        chunk_size=32, dropout=0.0
    )
    backbone.eval()

    counts = backbone.count_blocks_by_type()
    types = backbone.get_block_types()
    print(f"  Block pattern: {types}")
    check(counts['gdn'] == 3, "3 GDN blocks")
    check(counts['nope'] == 1, "1 NoPE block")
    check(types == ['gdn', 'gdn', 'gdn', 'nope'], "correct pattern [G,G,G,N]")


def test_encoder_zero_positional_encoding():
    print(f"\n{'─'*60}")
    print("[Test 3] Encoder has zero positional encoding")
    print(f"{'─'*60}")

    backbone = NoPEGDNVideoBackbone(
        img_size=224, num_frames=16, tubelet_size=(2, 16, 16),
        encoder_dim=128, encoder_depth=2, encoder_heads=4,
        processor_dim=128, processor_depth=4, processor_heads=4,
        chunk_size=32, dropout=0.0
    )
    backbone.eval()

    pe_kw = ['inv_freq', 'pos_embed', 'position', 'alibi', 'rel_pos', 'bias_table']
    enc_params_names = [n for n, _ in backbone.encoder.named_parameters()]
    enc_bufs = [n for n, _ in backbone.encoder.named_buffers()]
    pe_found = [n for n in enc_params_names + enc_bufs
                if any(kw in n.lower() for kw in pe_kw)]
    check(len(pe_found) == 0, "no PE params/buffers in encoder")

    embeddings = [n for n, m in backbone.encoder.named_modules()
                  if isinstance(m, nn.Embedding)]
    check(len(embeddings) == 0, "no nn.Embedding in encoder")

    # Check NoPEMultiheadAttention for rotation keywords
    # (Jupyter replacement for inspect.getsource)
    nope_attrs = dir(NoPEMultiheadAttention)
    check('rope' not in str(nope_attrs).lower(), "no rope references in NoPE attention")
    check(not hasattr(NoPEMultiheadAttention, 'apply_rotary_pos_emb'),
          "no rotary method in NoPE attention")


def test_gdn_sequential_chunkwise_agreement():
    print(f"\n{'─'*60}")
    print("[Test 4] GDN sequential ↔ chunkwise numerical agreement")
    print(f"{'─'*60}")

    gdn = GatedDeltaLayer(hidden_size=64, num_heads=2, head_dim=32, chunk_size=16)
    for L in [1, 7, 16, 37, 64, 128]:
        x_t = torch.randn(1, L, 64, dtype=torch.float64)
        gdn_d = gdn.double()
        with torch.no_grad():
            o_s, s_s = gdn_d(x_t, use_parallel=False)
            o_c, s_c = gdn_d(x_t, use_parallel=True)
        e_o = (o_s - o_c).abs().max().item()
        e_s = (s_s - s_c).abs().max().item()
        ok = e_o < 1e-5 and e_s < 1e-5
        check(ok, f"L={L}", f"out Δ={e_o:.2e}  state Δ={e_s:.2e}")
    gdn = gdn.float()


def test_gdn_breaks_permutation_symmetry():
    print(f"\n{'─'*60}")
    print("[Test 5] GDN breaks permutation symmetry (provides position)")
    print(f"{'─'*60}")

    gdn_test = GatedDeltaLayer(hidden_size=64, num_heads=2, head_dim=32)
    gdn_test.eval()
    tok = torch.randn(1, 1, 64)
    repeated = tok.expand(1, 8, 64).clone()

    with torch.no_grad():
        out_gdn, _ = gdn_test(repeated, use_parallel=False)

    diffs = [(out_gdn[0, i] - out_gdn[0, 0]).abs().max().item() for i in range(1, 8)]
    check(max(diffs) > 1e-4, "identical input → different outputs",
          f"max divergence: {max(diffs):.4e}")


def test_gradient_flow():
    print(f"\n{'─'*60}")
    print("[Test 6] Gradient flow — full backbone")
    print(f"{'─'*60}")

    bb_grad = NoPEGDNVideoBackbone(
        224, 16, (2, 16, 16), 3,
        64, 1, 2, 64, 4, 2, chunk_size=32, dropout=0.0)
    bb_grad.train()

    v_grad = torch.randn(1, 3, 16, 224, 224, requires_grad=True)
    out_grad = bb_grad(v_grad)
    loss = out_grad.sum()
    loss.backward()

    check(v_grad.grad is not None and v_grad.grad.abs().max() > 0,
          "input gradient flows")

    n_ok = sum(1 for p in bb_grad.parameters() if p.grad is not None and p.grad.abs().max() > 0)
    n_total = sum(1 for p in bb_grad.parameters() if p.requires_grad)
    check(n_ok == n_total, f"all {n_total} param grads non-zero",
          f"{n_ok}/{n_total}")


def test_trecvit_forward_backward():
    print(f"\n{'─'*60}")
    print("[Test TRecViT] forward + backward + α_logit grad")
    print(f"{'─'*60}")

    trec = TRecViTClassifier(
        img_size=64, num_frames=8, patch_size=(2, 16, 16),
        in_channels=3, width=48, depth=2, num_heads=4,
        mlp_ratio=2.0, num_classes=10, rep_size=128,
    )
    video = torch.randn(2, 3, 8, 64, 64)
    logits = trec(video)
    check(logits.shape == (2, 10), "trecvit logits shape", f"{logits.shape}")

    import torch.nn.functional as _F
    loss = _F.cross_entropy(logits, torch.tensor([3, 7]))
    loss.backward()

    # every param must have a finite gradient
    n_params = sum(1 for _ in trec.parameters())
    n_finite = sum(1 for p in trec.parameters()
                   if p.grad is not None and torch.isfinite(p.grad).all())
    check(n_finite == n_params, "trecvit all grads finite", f"{n_finite}/{n_params}")

    # RG-LRU: log_a, input_gate, a_gate must all receive non-zero gradient
    total_log_a_grad     = sum(m.log_a.grad.abs().sum().item()         for m in trec.modules() if isinstance(m, RealLRU))
    total_input_gate_grad = sum(m.input_gate.weight.grad.abs().sum().item() for m in trec.modules() if isinstance(m, RealLRU))
    total_a_gate_grad     = sum(m.a_gate.weight.grad.abs().sum().item()     for m in trec.modules() if isinstance(m, RealLRU))
    check(total_log_a_grad > 0,      "trecvit log_a grad nonzero (decay learns)",      f"|sum|={total_log_a_grad:.4e}")
    check(total_input_gate_grad > 0, "trecvit input_gate grad nonzero (gate_x learns)", f"|sum|={total_input_gate_grad:.4e}")
    check(total_a_gate_grad > 0,     "trecvit a_gate grad nonzero (gate_a learns)",     f"|sum|={total_a_gate_grad:.4e}")


if __name__ == "__main__":
    print("=" * 70)
    print("3D NoPE + GATED DELTANET VIDEO BACKBONE — Core Tests")
    print("=" * 70)

    test_full_backbone_forward_shapes()
    test_hybrid_3to1_ratio()
    test_encoder_zero_positional_encoding()
    test_gdn_sequential_chunkwise_agreement()
    test_gdn_breaks_permutation_symmetry()
    test_gradient_flow()
    test_trecvit_forward_backward()

    print(f"\n{'='*70}")
    if failed == 0:
        print(f"✅ ALL {passed}/{total_tests} TESTS PASSED")
    else:
        print(f"❌ {failed}/{total_tests} TESTS FAILED ({passed} passed)")
    print(f"{'='*70}")
    sys.exit(1 if failed else 0)
