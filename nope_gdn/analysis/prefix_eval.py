"""True contiguous prefix evaluation on SSv2 — only the first N *contiguous*
frames of each video are fed (the rest has not yet been observed), simulating
real-time inference. No test-time augmentation. The NoPE-GDN model is fully
causal (gdn_temporal_only scans time per spatial location) with a
global-average-pool head, so it accepts any prefix length without modification.
"""
import json
from pathlib import Path
from contextlib import nullcontext

import torch
import torchvision.transforms as T
try:
    import av
except Exception:
    av = None

from ..config import get_config
from ..data.common import read_video
from ..models.factory import build_model
from ..training.utils import accuracy


class PrefixDataset(torch.utils.data.Dataset):
    """SSv2 dataset that returns only the FIRST N contiguous frames of each video."""
    def __init__(self, data_root, prefix_frames=16, img_size=224,
                 video_dir="20bn-something-something-v2", anno_dir="annotations/"):
        self.prefix_frames = prefix_frames
        self.img_size = img_size
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        data_root = Path(data_root)
        self.video_root = data_root / video_dir
        anno_root = data_root / anno_dir
        with open(anno_root / "labels.json") as f:
            self.label_map = json.load(f)
        with open(anno_root / "validation.json") as f:
            annotations = json.load(f)
        self.samples = []
        for item in annotations:
            vid_id = item["id"]
            label_text = item["template"].replace("[", "").replace("]", "")
            label_id = int(self.label_map.get(label_text, -1))
            if label_id == -1:
                continue
            for ext in [".webm", ".mp4"]:
                p = self.video_root / f"{vid_id}{ext}"
                if p.exists():
                    self.samples.append((str(p), label_id)); break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        nf = self.prefix_frames
        try:
            container = av.open(video_path)
            total = container.streams.video[0].frames
            if total == 0:
                total = sum(1 for _ in container.decode(container.streams.video[0]))
            container.close()
            if total <= nf:                              # pad short clips by repeating last frame
                indices = list(range(total)) + [total - 1] * (nf - total)
            else:
                indices = list(range(nf))                # strictly the first nf frames
            video = read_video(video_path, indices)
            video = video.float().div(255.0).permute(0, 3, 1, 2)
            size = int(self.img_size * 256 / 224)
            video = torch.nn.functional.interpolate(
                video, size=(size, size), mode='bilinear', align_corners=False)
            off = (size - self.img_size) // 2
            video = video[:, :, off:off + self.img_size, off:off + self.img_size]
            video = torch.stack([self.normalize(f) for f in video]).permute(1, 0, 2, 3)
        except Exception as e:
            print(f"  Prefix warning: {video_path}: {e}")
            video = torch.zeros(3, nf, self.img_size, self.img_size)
        return video, label


def ssv2_prefix_loader(nf, data_root, img_size=224, batch_size=16, num_workers=8,
                       video_dir="20bn-something-something-v2", anno_dir="annotations/"):
    ds = PrefixDataset(data_root, prefix_frames=nf, img_size=img_size,
                       video_dir=video_dir, anno_dir=anno_dir)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def evaluate_prefix(model, make_loader,
                    prefix_frames_list=(4, 8, 12, 16, 20, 24, 28, 32),
                    device=None, amp_dtype=torch.bfloat16):
    """Evaluate `model` on contiguous prefixes of increasing length (no TTA).
    make_loader(nf) -> a DataLoader yielding (video[:, :, :nf], label)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cuda = (device == "cuda")
    model.eval()
    results = []
    for nf in prefix_frames_list:
        loader = make_loader(nf)
        c1 = c5 = total = 0
        for videos, labels in loader:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            ctx = torch.autocast("cuda", dtype=amp_dtype) if cuda else nullcontext()
            with ctx:
                logits = model(videos)
            a1, a5 = accuracy(logits, labels, topk=(1, 5))
            c1 += a1.item() * labels.size(0) / 100
            c5 += a5.item() * labels.size(0) / 100
            total += labels.size(0)
        top1, top5 = c1 / total * 100, c5 / total * 100
        print(f"  Prefix {nf:>2} frames:  Top-1={top1:5.2f}%   Top-5={top5:5.2f}%")
        results.append({"frames": nf, "top1": top1, "top5": top5})
    return results


_VARIANT_LABEL = {
    "nope_gdn": "NoPE-GDN (gdn_temporal_only=True, global-pool)",
    "rope": "VideoRoPE", "axial_rope": "Axial-RoPE",
    "trecvit": "TRecViT", "learned_pe": "LearnedPE",
}


def run_prefix_eval(ckpt, data_root, variant="nope_gdn", size="base", num_classes=174,
                    img_size=224, prefix_frames_list=(4, 8, 12, 16, 20, 24, 28, 32),
                    batch_size=16, num_workers=8, device=None,
                    video_dir="20bn-something-something-v2", anno_dir="annotations/",
                    label=None):
    """Build `variant`, load its checkpoint, and run true contiguous prefix evaluation
    on SSv2. nope_gdn is built as gdn_temporal_only=True + global-average-pool head.
    trecvit (causal via its RG-LRU) and axial_rope (bidirectional; recomputes rotary
    positions from the input) also accept any prefix length without modification."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_config(variant=variant, size=size)
    cfg.model.num_classes = num_classes
    cfg.model.img_size = img_size
    if variant == "nope_gdn":
        cfg.model.gdn_temporal_only = True              # time-only recurrence per spatial location
    model = build_model(variant, cfg.model).to(device)
    if variant == "nope_gdn":
        assert getattr(model.backbone, "gdn_temporal_only", False), "expected gdn_temporal_only=True"
        assert type(model.head).__name__ == "VideoClassificationHead", "expected global-pool head"
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    ema = ck.get("ema_state") if isinstance(ck, dict) else None
    if ema and "shadow" in ema:
        sd = model.state_dict()
        for n in ema["shadow"]:
            if n in sd: sd[n] = ema["shadow"][n]
        model.load_state_dict(sd)
        print(f"  loaded EMA weights, epoch {ck.get('epoch')}, best_acc={ck.get('best_acc')}")
    else:
        model.load_state_dict(ck.get("model_state", ck), strict=False)
    model.eval()
    print(f"{label or _VARIANT_LABEL.get(variant, variant)} — true contiguous prefix eval:")
    return evaluate_prefix(
        model,
        make_loader=lambda nf: ssv2_prefix_loader(
            nf, data_root, img_size=img_size, batch_size=batch_size,
            num_workers=num_workers, video_dir=video_dir, anno_dir=anno_dir),
        prefix_frames_list=prefix_frames_list, device=device)


def run_prefix_eval_all(variants, data_root,
                        prefix_frames_list=(4, 8, 12, 16, 20, 24, 28, 32), **kw):
    """variants: {label: {"variant": ..., "ckpt": ...}}. Runs each and prints a
    side-by-side Top-1 table across prefix lengths."""
    res = {}
    for lab, spec in variants.items():
        try:
            res[lab] = run_prefix_eval(spec["ckpt"], data_root, variant=spec["variant"],
                                       label=lab, prefix_frames_list=prefix_frames_list, **kw)
        except Exception as e:
            print(f"[skip] {lab}: {type(e).__name__}: {e}")
    if res:
        print("\n=== True contiguous prefix evaluation — Top-1 (%) ===")
        print(f"{'Prefix':>7} " + "".join(f"{lab[:13]:>15}" for lab in res))
        for i, nf in enumerate(prefix_frames_list):
            print(f"{nf:>7} " + "".join(f"{res[lab][i]['top1']:>15.2f}" for lab in res))
    return res
