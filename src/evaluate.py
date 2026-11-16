import argparse, json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.common import load_config, make_model, device
from src.dataset import TileDataset
from src.train import scores_from_logits, metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", default=None, help="Explicit checkpoint path (default: <output_dir>/checkpoints/best.pt)")
    ap.add_argument("--output", default=None, help="Output metrics JSON path (default: <output_dir>/metrics.json)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_dir"])
    df = pd.read_csv(cfg.get("splits_csv") or (out / "splits.csv"))
    test = df[df.split == "test"]
    if test.empty:
        raise ValueError("No test split. Provide at least 3 independent geographic raster areas and rerun preparation, or implement spatial-block splitting.")
    dev = device()
    ckpt_path = Path(args.checkpoint) if args.checkpoint else out / "checkpoints" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    # Architektura z checkpointu (pozwala ocenić model z innym enkoderem na masach z tego configu)
    arch_cfg = ckpt.get("config", cfg) if isinstance(ckpt, dict) else cfg
    arch_cfg = {**arch_cfg, "training": {**arch_cfg.get("training", {}), "encoder_weights": None}}
    model = make_model(arch_cfg).to(dev)
    model.load_state_dict(ckpt["model"]); model.eval()
    loader = DataLoader(TileDataset(test), batch_size=int(cfg.get("training", {}).get("batch_size", 8)),
                        shuffle=False, num_workers=int(cfg.get("training", {}).get("num_workers", 0)))
    tp = fp = fn = 0
    threshold = float(cfg.get("training", {}).get("threshold", 0.5))
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(dev))
            a,b,c = scores_from_logits(logits.cpu(), y, threshold)
            tp += a; fp += b; fn += c
    result = metrics(tp, fp, fn)
    result.update({"test_tiles": int(len(test)), "threshold": threshold,
                   "checkpoint": str(ckpt_path),
                   "note": "Pixel-level metrics against ground-truth derived masks."})
    metrics_path = Path(args.output) if args.output else out / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved metrics to {metrics_path}:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
