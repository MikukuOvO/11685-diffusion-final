#!/usr/bin/env python3
"""
Compute Inception Score (IS) for a directory of generated images.

Example:
    python compute_is.py \
        --image_dir output/report_eval/latent_epoch280_ddim250_cfg2p75_clipfalse_ema/generated_images \
        --output output/report_eval/latent_epoch280_ddim250_cfg2p75_clipfalse_ema/is.json
"""

import argparse
import json
import os

from fid_utils import compute_inception_score_from_dir


def main():
    parser = argparse.ArgumentParser(description="Compute Inception Score for generated images")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing generated images")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON output path")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    mean, std = compute_inception_score_from_dir(
        args.image_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        splits=args.splits,
    )

    result = {
        "image_dir": args.image_dir,
        "num_splits": args.splits,
        "inception_score_mean": mean,
        "inception_score_std": std,
    }

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(f"Saved IS result to {args.output}")


if __name__ == "__main__":
    main()
