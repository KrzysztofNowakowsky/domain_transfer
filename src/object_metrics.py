"""Metryki obiektowe (per budynek) w obszarze testowym — do porównania 10 cm vs 20 cm.

Budynek referencyjny uznajemy za wykryty, gdy predykcje pokrywają >= --min-overlap jego powierzchni.
Wynik: recall w przedziałach powierzchni (małe obiekty: altanki, garaże, dobudówki) + liczba
fałszywych poligonów (predykcja bez żadnego przecięcia z referencją).

Przykład:
python -m src.object_metrics --config configs/config.yaml \
  --predictions outputs/poland_10cm/pred_north.gpkg outputs/poland_10cm/pred_south.gpkg \
  --output outputs/poland_10cm/object_metrics.json
"""
import argparse, json
from pathlib import Path
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window, bounds as win_bounds
from shapely.geometry import box
from shapely.ops import unary_union
from src.common import load_config

BINS = [(0, 25, "<25 m²"), (25, 60, "25–60 m²"), (60, 150, "60–150 m²"), (150, float("inf"), ">150 m²")]

def test_area(cfg):
    out = Path(cfg["output_dir"])
    df = pd.read_csv(cfg.get("splits_csv") or (out / "splits.csv"))
    test = df[df.split == "test"]
    tile = int(cfg.get("tile_size", 512))
    boxes, crs = [], None
    for raster, grp in test.groupby("raster"):
        with rasterio.open(raster) as src:
            crs = src.crs
            for r, c in zip(grp["row"], grp["col"]):
                boxes.append(box(*win_bounds(Window(int(c), int(r), tile, tile), src.transform)))
    return unary_union(boxes), crs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Config wyznaczający obszar testowy i referencję")
    ap.add_argument("--predictions", nargs="+", required=True, help="Jeden lub więcej plików .gpkg z predict.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-overlap", type=float, default=0.5)
    args = ap.parse_args()
    cfg = load_config(args.config)
    area, crs = test_area(cfg)

    ref = gpd.read_file(cfg["buildings_vector"], layer=cfg.get("buildings_layer"))
    ref = ref[ref.geometry.notna() & ~ref.geometry.is_empty].to_crs(crs)
    ref = ref[ref.geometry.centroid.within(area)].copy()
    ref["area_m2"] = ref.geometry.area

    preds = pd.concat([gpd.read_file(p).to_crs(crs) for p in args.predictions], ignore_index=True)
    preds = gpd.GeoDataFrame(preds, geometry="geometry", crs=crs)
    preds = preds[preds.geometry.intersects(area)].copy()
    pred_union = unary_union(list(preds.geometry)) if len(preds) else None

    ref["covered"] = [
        (g.intersection(pred_union).area / g.area) if (pred_union is not None and g.area > 0) else 0.0
        for g in ref.geometry
    ]
    ref["detected"] = ref["covered"] >= args.min_overlap

    by_size = []
    for lo, hi, label in BINS:
        sub = ref[(ref.area_m2 >= lo) & (ref.area_m2 < hi)]
        by_size.append({"bin": label, "n_reference": int(len(sub)),
                        "n_detected": int(sub.detected.sum()),
                        "recall": float(sub.detected.mean()) if len(sub) else None})
    ref_union = unary_union(list(ref.geometry)) if len(ref) else None
    false_polys = int(sum(1 for g in preds.geometry if ref_union is None or not g.intersects(ref_union)))

    result = {
        "config": args.config, "predictions": args.predictions, "min_overlap": args.min_overlap,
        "n_reference_in_test": int(len(ref)), "n_detected": int(ref.detected.sum()),
        "object_recall": float(ref.detected.mean()) if len(ref) else None,
        "n_predicted_polygons_in_test": int(len(preds)), "n_false_polygons": false_polys,
        "recall_by_size": by_size,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
