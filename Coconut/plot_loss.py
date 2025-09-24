import argparse
import json
import os
import sys

import matplotlib.pyplot as plt

def read_jsonl(path):
    xs, ys = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Expect keys: train/step, train/loss
            if ("train/step" in obj) and ("train/loss" in obj):
                try:
                    xs.append(float(obj["train/step"]))
                    ys.append(float(obj["train/loss"]))
                except Exception:
                    pass
    return xs, ys

def main():
    parser = argparse.ArgumentParser(description="Plot training loss from JSONL and save to PNG.")
    parser.add_argument("jsonl", help="Input JSONL log file (one JSON per line).")
    parser.add_argument("--output", "-o", help="Output image path (default: <jsonl_basename>_loss.png).")
    parser.add_argument("--title", help="Optional title for the plot.", default=None)
    parser.add_argument("--xlabel", help="X label (default: train/step).", default="train/step")
    parser.add_argument("--ylabel", help="Y label (default: train/loss).", default="train/loss")
    args = parser.parse_args()

    if not os.path.exists(args.jsonl):
        print(f"File not found: {args.jsonl}", file=sys.stderr)
        sys.exit(1)

    x, y = read_jsonl(args.jsonl)
    if not x:
        print("No valid data points found in the JSONL file.", file=sys.stderr)
        sys.exit(2)

    # Sort by step to avoid zig-zag if out of order
    paired = sorted(zip(x, y), key=lambda t: t[0])
    x = [p[0] for p in paired]
    y = [p[1] for p in paired]

    plt.figure()
    plt.plot(x, y)
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    if args.title:
        plt.title(args.title)
    plt.tight_layout()

    out_path = args.output
    if not out_path:
        base = os.path.splitext(os.path.basename(args.jsonl))[0]
        save_dir = "/mnt/shared-storage-user/weixilin/MLLM/coconut_reproduce/visualize_loss"
        os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
        out_path = os.path.join(save_dir, f"{base}_loss.png")

        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()