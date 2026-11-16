"""Download an RGB WMS image and save it as a georeferenced GeoTIFF.

Example for the official GRAFCAN orthophoto service:
python -m src.download_wms \
  --service 'https://idecan1.grafcan.es/ServicioWMS/Ortofoto' \
  --layer ortoterritorial \
  --bbox 214800 3173500 215200 3173900 \
  --resolution 0.2 \
  --output data/los_llanos_sample_20cm.tif
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", required=True, help="WMS base URL, without query parameters")
    parser.add_argument("--layer", required=True, help="WMS layer name")
    parser.add_argument("--bbox", required=True, nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    parser.add_argument("--crs", default="EPSG:32628", help="Projected WMS CRS; default: EPSG:32628")
    parser.add_argument("--resolution", type=float, default=0.2, help="Output pixel size in CRS units; default: 0.2 m")
    parser.add_argument("--output", required=True, type=Path, help="Output RGB GeoTIFF path")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xmin, ymin, xmax, ymax = args.bbox
    if not (xmin < xmax and ymin < ymax):
        raise ValueError("bbox must satisfy xmin < xmax and ymin < ymax")
    if args.resolution <= 0:
        raise ValueError("resolution must be positive")

    width = round((xmax - xmin) / args.resolution)
    height = round((ymax - ymin) / args.resolution)
    if width <= 0 or height <= 0:
        raise ValueError("bbox is smaller than one output pixel")
    if width * height > 100_000_000:
        raise ValueError("Requested image exceeds 100 million pixels. Download smaller chunks.")

    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": args.layer,
        "STYLES": "",
        "CRS": args.crs,
        "BBOX": f"{xmin},{ymin},{xmax},{ymax}",
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": "image/jpeg",
        "TRANSPARENT": "FALSE",
    }
    url = f"{args.service}?{urlencode(params)}"
    print(f"Requesting {width} x {height} px from {args.layer}")
    with urlopen(url, timeout=args.timeout) as response:
        image = Image.open(io.BytesIO(response.read())).convert("RGB")

    array = np.asarray(image)
    if array.shape[:2] != (height, width):
        raise RuntimeError(f"WMS returned {array.shape[1]} x {array.shape[0]} px, expected {width} x {height} px")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": "uint8",
        "crs": args.crs,
        "transform": from_bounds(xmin, ymin, xmax, ymax, width, height),
        "compress": "deflate",
        "photometric": "RGB",
    }
    with rasterio.open(args.output, "w", **profile) as dst:
        dst.write(np.moveaxis(array, -1, 0))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
