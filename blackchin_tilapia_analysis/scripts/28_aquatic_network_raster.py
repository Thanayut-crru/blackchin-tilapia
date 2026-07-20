"""Mask the full-country best-model raster with OSM water polygons and waterways."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT.parent
OUT = ROOT / "output"
RASTER_DIR = OUT / "best_model_rasters"
DATA = PROJECT / "data"
OSM = DATA / "osm" / "thailand_shp"

MEAN_IN = RASTER_DIR / "Paper1_best_RF_MSAP_uncertainty_mean_FullCountry_Thailand.tif"
SD_IN = RASTER_DIR / "Paper1_best_RF_MSAP_uncertainty_sd_FullCountry_Thailand.tif"
WATER_POLYGONS = OSM / "gis_osm_water_a_free_1.shp"
WATERWAYS = OSM / "gis_osm_waterways_free_1.shp"
BUFFER_M = 500
NODATA = -9999.0


def load_water_geometries(target_crs):
    polygons = gpd.read_file(WATER_POLYGONS).to_crs("EPSG:3857")
    waterways = gpd.read_file(WATERWAYS).to_crs("EPSG:3857")
    polygons = polygons[polygons.geometry.notna() & ~polygons.geometry.is_empty]
    waterways = waterways[waterways.geometry.notna() & ~waterways.geometry.is_empty]
    if "fclass" in waterways.columns:
        waterways = waterways[
            waterways["fclass"].isin(["river", "canal", "stream"])]
    buffered_lines = waterways.geometry.buffer(BUFFER_M)
    combined = gpd.GeoSeries(
        list(polygons.geometry) + list(buffered_lines), crs="EPSG:3857")
    return combined.to_crs(target_crs)


def mask_raster(input_path, output_path, water_mask):
    with rasterio.open(input_path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        valid = np.isfinite(arr) & (arr != src.nodata)
        arr[~(valid & water_mask)] = NODATA
        profile.update(nodata=NODATA, compress="deflate", tiled=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(arr, 1)


def main():
    if not MEAN_IN.exists():
        raise FileNotFoundError(MEAN_IN)
    with rasterio.open(MEAN_IN) as src:
        geometries = load_water_geometries(src.crs)
        water_mask = rasterize(
            ((geom, 1) for geom in geometries if geom is not None and not geom.is_empty),
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        ).astype(bool)

    mean_out = RASTER_DIR / "Paper1_best_RF_MSAP_mean_AquaticNetwork_Thailand.tif"
    sd_out = RASTER_DIR / "Paper1_best_RF_MSAP_sd_AquaticNetwork_Thailand.tif"
    mask_raster(MEAN_IN, mean_out, water_mask)
    mask_raster(SD_IN, sd_out, water_mask)
    plot_map(mean_out)

    metadata = {
        "water_polygon_source": str(WATER_POLYGONS),
        "waterway_source": str(WATERWAYS),
        "waterway_buffer_m": BUFFER_M,
        "rasterization": "all_touched=True",
        "interpretation": (
            "Recommended ecological display mask. Values remain relative habitat "
            "suitability, not calibrated invasion probability."
        ),
    }
    (RASTER_DIR / "AquaticNetwork_mask_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Aquatic cells: {int(water_mask.sum())}")
    print(mean_out)
    print(sd_out)


def plot_map(raster_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    boundary = gpd.read_file(DATA / "thailand_boundary.geojson").to_crs("EPSG:4326")
    with rasterio.open(raster_path) as src:
        arr = src.read(1).astype(float)
        arr[arr == src.nodata] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    fig, ax = plt.subplots(figsize=(6.2, 8.2))
    image = ax.imshow(
        arr, extent=extent, origin="upper", cmap="viridis",
        norm=Normalize(vmin=0, vmax=1), interpolation="nearest")
    boundary.boundary.plot(ax=ax, color="black", linewidth=0.8)
    fig.colorbar(image, ax=ax, shrink=0.65, label="Relative habitat suitability")
    ax.set_title("Blackchin tilapia relative habitat suitability\nOSM aquatic-network mask")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    figure_dir = OUT / "factor_explainability" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"Fig_aquatic_suitability_map.{suffix}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
