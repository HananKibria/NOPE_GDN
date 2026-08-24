"""Train on Something-Something V2.

Example:
    python scripts/train_ssv2.py --variant nope_gdn --size base \
        --data-root ./data/ssv2 --pretrained videomae --epochs 30
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nope_gdn.training.train import run_training


def main():
    p = argparse.ArgumentParser(description="Train on SSv2")
    p.add_argument("--variant", default="nope_gdn",
                   choices=["nope_gdn", "rope", "videorope", "axial_rope", "mixed_rope", "trecvit"])
    p.add_argument("--size", default="base", choices=["tiny", "small", "base"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--pretrained", default="videomae",
                   help="'videomae' | 'deit' | '' (none)")
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--debug", action="store_true",
                   help="synthetic DebugDataset pipeline test")
    args = p.parse_args()

    run_training(variant=args.variant, size=args.size, epochs=args.epochs,
                 batch_size=args.batch_size, lr=args.lr, debug=args.debug,
                 device=args.device, data_root=args.data_root,
                 num_workers=args.num_workers,
                 pretrained=args.pretrained or False, resume=args.resume)


if __name__ == "__main__":
    main()
