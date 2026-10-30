"""Train FNO (single-shot or autoregressive) on Schrödinger data."""
import os
import argparse
import yaml
import torch

from src.train.loop import train_model


def main(config_path: str, data_dir: str, mode: str, seed: int):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["data_dir"] = data_dir
    cfg.setdefault("out_dir", "results")
    os.makedirs(cfg["out_dir"], exist_ok=True)
    train_model(cfg, data_dir=data_dir, mode=mode, seed=seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--mode", default="single", choices=["single", "autoregressive"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.config, args.data, args.mode, args.seed)