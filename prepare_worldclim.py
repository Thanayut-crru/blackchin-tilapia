"""Download and crop the WorldClim 2.1 predictors used by Paper 1.

WorldClim files are intentionally not redistributed in this release. Running
this script means you accept the provider's terms at https://worldclim.org/about.html.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "environmental" / "clipped"
CACHE = ROOT / "external_cache" / "worldclim"
BIO_URL = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_2.5m_bio.zip"
ELEV_URL = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_2.5m_elev.zip"
SELECTED = [1, 4, 7, 10, 11, 12, 15, 16, 17]
BOUNDS = (96.0, 5.0, 106.0, 23.0)


def fetch(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"Using cached {path}")
        return
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, path.open("wb") as target:
        shutil.copyfileobj(response, target)


def crop(src_path: Path, dst_path: Path) -> None:
    with rasterio.open(src_path) as src:
        window = from_bounds(*BOUNDS, transform=src.transform)
        window = window.round_offsets().round_lengths()
        arr = src.read(1, window=window)
        transform = src.window_transform(window)
        meta = src.meta.copy()
        meta.update(width=arr.shape[1], height=arr.shape[0], transform=transform, compress="lzw")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **meta) as dst:
            dst.write(arr, 1)
    print(f"Created {dst_path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download archives from WorldClim")
    parser.add_argument("--bio-zip", type=Path, help="Use an existing WorldClim BIO archive")
    parser.add_argument("--elev-zip", type=Path, help="Use an existing WorldClim elevation archive")
    args = parser.parse_args()

    bio_zip = args.bio_zip or CACHE / "wc2.1_2.5m_bio.zip"
    elev_zip = args.elev_zip or CACHE / "wc2.1_2.5m_elev.zip"
    if args.download:
        fetch(BIO_URL, bio_zip)
        fetch(ELEV_URL, elev_zip)
    if not bio_zip.exists() or not elev_zip.exists():
        raise SystemExit("Provide --download or both --bio-zip and --elev-zip.")

    extract = CACHE / "extracted"
    extract.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bio_zip) as zf:
        zf.extractall(extract / "bio")
    with zipfile.ZipFile(elev_zip) as zf:
        zf.extractall(extract / "elev")

    for index in SELECTED:
        source = extract / "bio" / f"wc2.1_2.5m_bio_{index}.tif"
        crop(source, OUT / f"bio{index:02d}.tif")
    crop(extract / "elev" / "wc2.1_2.5m_elev.tif", OUT / "elevation.tif")
    print("WorldClim preparation complete. Run: python verify_inputs.py")


if __name__ == "__main__":
    main()
