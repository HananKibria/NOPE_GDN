"""Train on Diving48 (length-generalization transfer).

Example:
    python scripts/train_diving48.py --variant nope_gdn --size base \
        --data-root ./data/diving48 --init-weights ./outputs/ssv2/best_model.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nope_gdn.training.train import run_training_diving48


def main():
    p = argparse.ArgumentParser(description="Train on Diving48")
    p.add_argument("--variant", default="nope_gdn",
                   choices=["nope_gdn", "rope", "videorope", "axial_rope", "mixed_rope", "trecvit"])
    p.add_argument("--size", default="base", choices=["tiny", "small", "base"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-frames", type=int, default=32)
    p.add_argument("--data-root", default="./data/diving48")
    p.add_argument("--video-dir", default="videos")
    p.add_argument("--anno-dir", default="annotations")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pretrained", default="videomae")
    p.add_argument("--init-weights", default=None,
                   help="SSv2 checkpoint to transfer from")
    p.add_argument("--use-ema-weights", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    run_training_diving48(variant=args.variant, size=args.size,
                          epochs=args.epochs, batch_size=args.batch_size,
                          lr=args.lr, num_frames=args.num_frames,
                          data_root=args.data_root, num_workers=args.num_workers,
                          pretrained=args.pretrained, init_weights=args.init_weights,
                          use_ema_weights=args.use_ema_weights, resume=args.resume,
                          device=args.device, video_dir=args.video_dir,
                          anno_dir=args.anno_dir)


if __name__ == "__main__":
    main()
