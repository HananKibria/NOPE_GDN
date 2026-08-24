# NoPE-GDN: Gated Delta Networks for Causal, Streamable Video Understanding

Official code for the TCSVT paper *"Gated Delta Networks for Causal, Streamable Video Understanding"* (Kibria & Bibi).

NoPE-GDN is a fully causal, streamable video encoder: a position-encoding-free
spatial encoder (spatial-only attention) plus a temporal processor built from
Gated DeltaNet (KDA) linear recurrence. Temporal context lives in a fixed-size
recurrent state, so inference runs frame-by-frame at constant memory and there
is no position embedding to extrapolate.

## Models

| Model | Causal | Params | SSv2 Top-1 | Top-5 |
|---|---|---|---|---|
| Axial-RoPE ViT (baseline) | ✗ | 114.7M | 68.34 | 92.36 |
| **NoPE-GDN-T (ours)** | ✓ | 115.6M | 65.59 | 90.59 |
| TRecViT (baseline) | ✓ | 111.8M | 64.73 | 89.96 |
| **NoPE-GDN-F (ours)** | ✓ | 117.4M | 63.28 | 89.35 |
| VideoRoPE (baseline) | ✗ | 114.7M | 62.83 | 89.10 |

SSv2 at the 32-frame training length, 2-clip × 3-crop TTA. All models share the
tubelet embedding and the VideoMAE-Base-SSv2 initialization.

- **NoPE-GDN-T** (default): time-only scan — one causal GDN scan per spatial
  location; global average-pool head.
- **NoPE-GDN-F**: flat raster scan over the T′×S grid; temporal-attention head.

Highlights from the paper: +3.6 pp over VideoRoPE at 72-frame Diving48
evaluation (length generalization), wall-clock crossover past ~24 frames
(1.34× at 72 frames), and cached streaming inference that is numerically
faithful (max logit diff 2.3e-3) with a 3.7× end-to-end speedup at bounded
memory.

## Install

```bash
pip install -r requirements.txt
```

`fla-core` provides the fused Triton KDA kernel used in training; without it a
pure-PyTorch fallback runs (slower). `nvidia-dali-cuda12` (commented out) is
only needed for Kinetics-400 GPU decoding.

## Train

```bash
# SSv2 (VideoMAE-SSv2 init)
python scripts/train_ssv2.py --variant nope_gdn --size base \
    --data-root ./data/ssv2 --pretrained videomae

# Diving48 transfer from an SSv2 checkpoint
python scripts/train_diving48.py --variant nope_gdn --size base \
    --data-root ./data/diving48 --init-weights ./outputs/best_model.pt

# Kinetics-400
python scripts/train_kinetics400.py --variant nope_gdn --size base \
    --data-root ./data/kinetics400 --train-list train.txt --val-list val.txt
```

Variants: `nope_gdn`, `rope` (VideoRoPE), `axial_rope`, `mixed_rope`, `trecvit`.
Sizes: `tiny`, `small`, `base`. NoPE-GDN-T vs -F is selected with the
`gdn_temporal_only` flag in `ModelConfig` (`True` = -T, the default): -T pairs
the time-only scan with the global mean-pool head, -F pairs the flat raster
scan with the temporal-attention head (`TemporalPoolingHead`).

Datasets are expected in their standard layouts (SSv2: `20bn-something-something-v2/`
+ annotation JSONs; Diving48: `videos/` + V2 annotations; Kinetics-400: video
files + train/val list files). Download scripts are not included.

## Streaming inference

```python
import torch
from nope_gdn import build_model, get_config
from nope_gdn.streaming import streaming_predictions, verify_streaming_fidelity

cfg = get_config(variant="nope_gdn", size="base")
model = build_model("nope_gdn", cfg.model).eval()

video = torch.randn(1, 3, 64, 224, 224)          # a stream of 64 frames
logits_per_chunk = streaming_predictions(model, video, chunk_frames=2)
verify_streaming_fidelity(model, video, chunk_frames=2)  # vs stateless prefill
```

## Evaluation & analysis

Paper experiments live in `nope_gdn/analysis/` — each module is runnable
(`python -m nope_gdn.analysis.<module> --help`-style entry points where the
notebook had interactive drivers):

- `benchmark_speed` — wall-clock / GFLOPs / memory vs frame count (Table II)
- `prefix_eval` — true contiguous-prefix evaluation (Table III)
- `mechanism_ssv2`, `length_generalization`, `processor_mechanism` — attention
  entropy, GDN/LRU gate statistics vs length (Fig. 3)
- `weight_drift`, `encoder_freeze` — initialization-control analyses (§IV-G)
- `error_analysis` — per-class accuracy and confusions (§IV-H)
- `ablations` — frame-shuffle / recurrence / RoPE-band / temperature ablations

## Tests

```bash
python tests/test_core.py     # or: pytest tests/
```

Covers forward shapes, the 3:1 GDN:NoPE hybrid ratio, zero-PE check, GDN
sequential↔chunkwise numerical agreement, permutation sensitivity, gradient
flow, and the TRecViT baseline.

## Repository layout

```
nope_gdn/
├── config.py              # dataclass configs, get_config(variant, size)
├── models/
│   ├── gated_delta.py     # GatedDeltaLayer (KDA), BiGatedDeltaLayer
│   ├── backbone.py        # NoPE encoder + hybrid GDN/attention processor
│   ├── heads.py           # global-pool / temporal-attention heads
│   ├── classifier.py      # NoPEGDNClassifier (backbone + head)
│   ├── videorope.py       # VideoRoPE baseline
│   ├── axial_rope.py      # axial / mixed RoPE baseline
│   ├── trecvit.py         # TRecViT baseline (fused HGRN LRU)
│   ├── factory.py         # build_model(variant, cfg), count_params
│   └── pretrained.py      # VideoMAE / DeiT initialization loaders
├── data/                  # ssv2, diving48, kinetics (DALI+CPU), debug
├── training/              # engine, utils, LLRD optimizer, run_training*
├── streaming.py           # cached online inference
└── analysis/              # paper evaluation & mechanism analyses
tests/                     # core unit tests
scripts/                   # train_ssv2 / train_diving48 / train_kinetics400
```

## Citation

```
Kibria, H. & Bibi, I. "Gated Delta Networks for Causal, Streamable Video
Understanding." IEEE Transactions on Circuits and Systems for Video
Technology (TCSVT), submitted.
```
