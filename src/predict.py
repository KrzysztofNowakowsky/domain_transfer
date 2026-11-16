"""Predykcja na całym ortofotomapie i eksport poligonów do GeoPackage.

Zmiany względem pierwszej wersji (kafle bez zakładki, wektoryzacja kafel po kaflu):
- kafle 512x512 z zakładką: z każdego kafla bierzemy tylko środek (margines --margin px odrzucamy),
  więc każdy piksel jest liczony z kontekstem i nie ma "szwów" na granicach kafli;
- maska jest składana dla całego rastra i wektoryzowana raz — budynki nie są cięte liniami kafli;
- małe dziury wewnątrz dachów (kominy, zbiorniki, anteny) są wypełniane (--fill-holes-m2),
  małe wyspy usuwane (min_polygon_area_m2).
Metryki pikselowe z evaluate.py nie zależą od tego skryptu.
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
from tqdm import tqdm
import torch
from src.common import load_config, make_model, device, normalize_image


def remove_small(mask, min_px, value):
    """Usuwa (value=0) lub wypełnia (value=1) spójne składowe mniejsze niż min_px.
    Dla wypełniania dziur podaj odwróconą maskę; składowe dotykające krawędzi rastra są pomijane."""
    if min_px <= 1:
        return mask
    target = (mask == 0) if value == 1 else (mask == 1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(target.astype(np.uint8), connectivity=4)
    h, w = mask.shape
    small = np.zeros(n, dtype=bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area >= min_px:
            continue
        if value == 1 and (x == 0 or y == 0 or x + bw == w or y + bh == h):
            continue  # "dziura" na krawędzi rastra to tło, nie dziura w dachu
        small[i] = True
    mask[small[labels]] = value
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", default=None, help="Explicit checkpoint path (default: <output_dir>/checkpoints/best.pt)")
    ap.add_argument("--input", required=True, help="Input georeferenced orthophoto GeoTIFF")
    ap.add_argument("--output", required=True, help="Output GeoPackage path")
    ap.add_argument("--margin", type=int, default=None, help="Margines kafla odrzucany przy składaniu (px), domyślnie 64")
    ap.add_argument("--fill-holes-m2", type=float, default=None, help="Wypełnij dziury mniejsze niż (m²), domyślnie 4")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    dev = device()
    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path(cfg["output_dir"]) / "checkpoints" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    arch_cfg = ckpt.get("config", cfg)
    arch_cfg = {**arch_cfg, "training": {**arch_cfg.get("training", {}), "encoder_weights": None}}
    model = make_model(arch_cfg).to(dev)
    model.load_state_dict(ckpt["model"]); model.eval()

    bands = [int(x) for x in cfg.get("bands", [1, 2, 3])]
    tile = int(cfg.get("tile_size", 512))
    pred_cfg = cfg.get("prediction", {})
    margin = int(args.margin if args.margin is not None else pred_cfg.get("margin_px", 64))
    core = tile - 2 * margin
    if core <= 0:
        raise ValueError("margin musi być mniejszy niż połowa tile_size")
    threshold = float(pred_cfg.get("threshold", 0.5))
    min_area = float(pred_cfg.get("min_polygon_area_m2", 4.0))
    simplify = float(pred_cfg.get("simplify_tolerance_m", 0.1))
    fill_holes = float(args.fill_holes_m2 if args.fill_holes_m2 is not None else pred_cfg.get("fill_holes_m2", 4.0))
    batch_size = int(pred_cfg.get("batch_size", 8))

    with rasterio.open(args.input) as src:
        if src.crs is None:
            raise ValueError("Input GeoTIFF has no CRS.")
        if max(bands) > src.count:
            raise ValueError("Configured band index exceeds raster band count.")
        if not src.crs.is_projected and (min_area > 0 or simplify > 0 or fill_holes > 0):
            raise ValueError("Area filtering, hole filling and simplification require projected CRS in metres.")
        H, W = src.height, src.width
        px_area = abs(src.transform.a * src.transform.e)
        prob = np.zeros((H, W), dtype=np.uint8)  # prawdopodobieństwo * 255 dla całego rastra

        jobs = [(r0, c0) for r0 in range(0, H, core) for c0 in range(0, W, core)]

        def run_batch(batch):
            xs = []
            for r0, c0 in batch:
                tr0, tc0 = r0 - margin, c0 - margin          # lewy górny róg kafla (może wychodzić poza raster)
                rr0, cc0 = max(0, tr0), max(0, tc0)
                rr1, cc1 = min(H, tr0 + tile), min(W, tc0 + tile)
                raw = src.read(bands, window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0))
                img = normalize_image(np.moveaxis(raw, 0, -1))
                pad = np.zeros((tile, tile, img.shape[2]), dtype=np.float32)
                pad[rr0 - tr0: rr1 - tr0, cc0 - tc0: cc1 - tc0] = img
                xs.append(np.moveaxis(pad, -1, 0))
            x = torch.from_numpy(np.stack(xs)).float().to(dev)
            with torch.no_grad():
                p = torch.sigmoid(model(x))[:, 0].cpu().numpy()
            for (r0, c0), pi in zip(batch, p):
                r1, c1 = min(H, r0 + core), min(W, c0 + core)
                prob[r0:r1, c0:c1] = np.round(pi[margin:margin + (r1 - r0), margin:margin + (c1 - c0)] * 255).astype(np.uint8)

        for i in tqdm(range(0, len(jobs), batch_size), desc="Predicting"):
            run_batch(jobs[i:i + batch_size])

        mask = (prob >= int(round(threshold * 255))).astype(np.uint8)
        del prob
        mask = remove_small(mask, int(fill_holes / px_area), value=1)   # wypełnij małe dziury w dachach
        mask = remove_small(mask, int(min_area / px_area), value=0)     # usuń drobne plamki

        geometries = []
        for geom, val in shapes(mask, mask=mask.astype(bool), transform=src.transform):
            if val != 1:
                continue
            poly = shape(geom)
            if poly.is_empty:
                continue
            if simplify > 0:
                poly = poly.simplify(simplify, preserve_topology=True)
            if poly.area >= min_area:
                geometries.append(poly)
        crs = src.crs

    if not geometries:
        print("No polygons detected at the current threshold.")
        gdf = gpd.GeoDataFrame({"class": [], "area": []}, geometry=[], crs=crs)
    else:
        gdf = gpd.GeoDataFrame({"class": ["building_roof"] * len(geometries),
                                "area": [float(g.area) for g in geometries]},
                               geometry=geometries, crs=crs)
        gdf.geometry = gdf.geometry.make_valid()
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf.to_file(out_path, layer="roof_polygons", driver="GPKG")
    print(f"Saved {len(gdf)} polygons to {out_path} (margin={margin}px, fill_holes={fill_holes} m²)")


if __name__ == "__main__":
    main()
