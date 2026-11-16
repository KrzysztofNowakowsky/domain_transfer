"""Generowanie raportu wizualnego i paneli porównawczych (PNG) przy użyciu Pillow i NumPy.

Tworzy:
1. Panele porównawcze: Ortofoto RGB | Maska referencyjna (GT) | Maska predykowana | Nakładka TP/FP/FN.
2. Zestawienia dla Polski (baseline) oraz Los Llanos de Aridane (Zero-Shot).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from src.common import load_config, make_model, device


def create_header(width: int, text: str, height: int = 32) -> Image.Image:
    img = Image.new("RGB", (width, height), color=(240, 242, 245))
    draw = ImageDraw.Draw(img)
    draw.text((12, 8), text, fill=(30, 41, 59))
    return img


def generate_tile_comparison(cfg_path: str, checkpoint_path: str, split_name: str, out_img_path: Path, max_samples: int = 4):
    cfg = load_config(cfg_path)
    dev = device()
    model = make_model(cfg).to(dev)
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = Path(cfg["output_dir"])
    splits_csv = out_dir / "splits.csv"
    if not splits_csv.exists():
        print(f"Brak pliku {splits_csv}")
        return

    df = pd.read_csv(splits_csv)
    sub = df[(df["split"] == split_name) & (df["positive_pixels"] > 1000)]
    if sub.empty:
        sub = df[df["split"] == split_name]
    if sub.empty:
        print(f"Brak kafli dla splitu {split_name}")
        return

    samples = sub.sample(min(max_samples, len(sub)), random_state=42)
    threshold = float(cfg.get("prediction", {}).get("threshold", 0.5))

    tile_size = 512
    gap = 10
    header_h = 36
    row_h = tile_size + header_h + gap
    total_w = 4 * tile_size + 3 * gap
    total_h = len(samples) * row_h

    canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    headers = [
        "1. Ortofotomapa RGB",
        "2. Referencja (Ground Truth)",
        f"3. Predykcja AI (próg={threshold})",
        "4. Nakładka (Zielony: TP, Czerwony: FP, Niebieski: FN)"
    ]

    for row_idx, (_, row) in enumerate(samples.iterrows()):
        img_arr = np.load(row["image"])  # (H, W, 3), float32 [0, 1]
        gt_arr = np.load(row["mask"])    # (H, W), uint8 {0, 1}

        x = torch.from_numpy(np.moveaxis(img_arr, -1, 0)[None]).float().to(dev)
        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        pred_arr = (prob >= threshold).astype(np.uint8)

        # 1. RGB Image
        rgb_uint8 = (np.clip(img_arr, 0, 1) * 255).astype(np.uint8)
        img_pil = Image.fromarray(rgb_uint8)

        # 2. Ground Truth Mask
        gt_pil = Image.fromarray((gt_arr * 255).astype(np.uint8)).convert("RGB")

        # 3. Predicted Mask
        pred_pil = Image.fromarray((pred_arr * 255).astype(np.uint8)).convert("RGB")

        # 4. Overlay:
        # TP (Zielony): gt==1 & pred==1
        # FP (Czerwony): gt==0 & pred==1
        # FN (Niebieski): gt==1 & pred==0
        overlay = np.copy(img_arr)
        tp = (gt_arr == 1) & (pred_arr == 1)
        fp = (gt_arr == 0) & (pred_arr == 1)
        fn = (gt_arr == 1) & (pred_arr == 0)

        overlay[tp] = overlay[tp] * 0.35 + np.array([0.0, 0.95, 0.15]) * 0.65
        overlay[fp] = overlay[fp] * 0.35 + np.array([0.95, 0.15, 0.15]) * 0.65
        overlay[fn] = overlay[fn] * 0.35 + np.array([0.15, 0.45, 0.95]) * 0.65
        overlay_pil = Image.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8))

        panels = [img_pil, gt_pil, pred_pil, overlay_pil]
        y_top = row_idx * row_h

        for col_idx, (panel, title) in enumerate(zip(panels, headers)):
            x_left = col_idx * (tile_size + gap)
            
            # Nagłówek dla kolumny (tylko w pierwszym wierszu lub każdym)
            draw.rectangle([x_left, y_top, x_left + tile_size, y_top + header_h], fill=(241, 245, 249))
            lbl = f"{title} (Kafel {row_idx + 1})"
            draw.text((x_left + 12, y_top + 10), lbl, fill=(15, 23, 42))

            # Obrazek
            canvas.paste(panel, (x_left, y_top + header_h))

    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_img_path, quality=92)
    print(f"Pomyślnie zapisano panel porównawczy: {out_img_path} ({canvas.size[0]}x{canvas.size[1]} px)")

