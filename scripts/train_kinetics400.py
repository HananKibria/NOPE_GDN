"""Train on Kinetics-400 (DALI GPU decode with CPU PyAV fallback).

Example:
    python scripts/train_kinetics400.py --variant nope_gdn --size base \
        --data-root ./data/kinetics400 \
        --train-list ./data/kinetics400/train_list.txt \
        --val-list ./data/kinetics400/val_list.txt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nope_gdn.training.train import run_training_kinetics400


def main():
    p = argparse.ArgumentParser(description="Train on Kinetics-400")
    p.add_argument("--variant", default="nope_gdn",
                   choices=["nope_gdn", "rope", "videorope", "axial_rope", "mixed_rope", "trecvit"])
    p.add_argument("--size", default="base", choices=["tiny", "small", "base"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--num-frames", type=int, default=32)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--backend", default="auto", choices=["auto", "dali", "cpu"])
    p.add_argument("--data-root", default="./data/kinetics400")
    p.add_argument("--train-list", default=None, help="DALI/CPU train file list")
    p.add_argument("--val-list", default=None, help="DALI/CPU val file list")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--pretrained", default="videomae")
    p.add_argument("--init-weights", default=None)
    p.add_argument("--use-ema-weights", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    run_training_kinetics400(variant=args.variant, size=args.size,
                             epochs=args.epochs, batch_size=args.batch_size,
                             lr=args.lr, num_frames=args.num_frames,
                             stride=args.stride, backend=args.backend,
                             num_workers=args.num_workers,
                             pretrained=args.pretrained,
                             init_weights=args.init_weights,
                             use_ema_weights=args.use_ema_weights,
                             resume=args.resume, device=args.device,
                             data_root=args.data_root,
                             train_list=args.train_list, val_list=args.val_list)


if __name__ == "__main__":
    main()
