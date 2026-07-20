"""Export Thailand-clipped rasters for the best validated Paper 1 configuration."""

import importlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

exp = importlib.import_module("blackchin_tilapia_analysis.scripts.20_full_experiment_v2")
cfg = importlib.import_module("blackchin_tilapia_analysis.scripts.s00_config")

OUT_DIR = cfg.OUT5 / "best_model_rasters"
BOUNDARY = cfg.DATA / "thailand_boundary.geojson"
NODATA_FLOAT = -9999.0
NODATA_BYTE = 255


def _load_boundary():
    with open(BOUNDARY, encoding="utf-8") as handle:
        data = json.load(handle)
    if data["type"] == "FeatureCollection":
        return [feature["geometry"] for feature in data["features"]]
    if data["type"] == "Feature":
        return [data["geometry"]]
    return [data]


def _write_clipped(array, transform, crs, output_path, nodata, dtype, geometries):
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 2 if dtype != "uint8" else 1,
        "tiled": True,
    }
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(array.astype(dtype), 1)
        with rasterio.open(tmp_path) as src:
            clipped, clipped_transform = rio_mask(
                src,
                geometries,
                crop=True,
                filled=True,
                nodata=nodata,
                all_touched=True,
            )
            clipped_profile = src.profile.copy()
            clipped_profile.update(
                height=clipped.shape[1],
                width=clipped.shape[2],
                transform=clipped_transform,
                compress="deflate",
                tiled=True,
            )
        with rasterio.open(output_path, "w", **clipped_profile) as dst:
            dst.write(clipped)
    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pooled = pd.read_csv(cfg.OUT5 / "pooled_oof_metrics.csv")
    folds = pd.read_csv(cfg.OUT5 / "experiment_fold_metrics.csv")

    best = pooled.sort_values("auc_roc", ascending=False).iloc[0]
    algo = str(best["algo"])
    feat_set = str(best["feat_set"])
    pa_method = str(best["pa_method"])
    if algo != "RF":
        raise ValueError(f"Exporter currently expects RF; best configuration is {algo}")

    selected_folds = folds[
        (folds["algo"] == algo)
        & (folds["feat_set"] == feat_set)
        & (folds["pa_method"] == pa_method)
    ].copy()
    if len(selected_folds) != 15:
        raise ValueError(f"Expected 15 validated fold specifications, found {len(selected_folds)}")

    arrays, transform, crs = exp.load_all_rasters()
    feature_names = exp.MODEL_VARIANTS[feat_set]
    height, width = next(iter(arrays.values())).shape

    full_valid = np.ones((height, width), dtype=bool)
    for name in feature_names:
        full_valid &= np.isfinite(arrays[name])

    valid = full_valid.copy()
    if exp.WATER_MASK.exists():
        with rasterio.open(exp.WATER_MASK) as src:
            valid &= src.read(1) == 1

    rows, cols = np.where(valid)
    cell_lons = transform.c + (cols + 0.5) * transform.a
    cell_lats = transform.f + (rows + 0.5) * transform.e
    cell_x = np.column_stack([arrays[name][valid] for name in feature_names])
    full_x = np.column_stack([arrays[name][full_valid] for name in feature_names])

    occurrences = pd.read_csv(cfg.OCC_CSV)
    occurrences["year"] = pd.to_numeric(occurrences["year"], errors="coerce")
    occurrences = occurrences[occurrences["year"] <= 2024].copy().reset_index(drop=True)
    pres_lons = occurrences["longitude"].to_numpy(float)
    pres_lats = occurrences["latitude"].to_numpy(float)
    pres_x = exp.sample_at_coords(
        arrays, feature_names, pres_lons, pres_lats, transform)
    if not np.isfinite(pres_x).all():
        raise ValueError("At least one occurrence has missing predictor values")

    predictions = []
    full_predictions = []
    model_records = []
    for _, row in selected_folds.sort_values(["rep", "fold"]).iterrows():
        selected_names = [name for name in str(row["selected"]).split(",") if name]
        selected_idx = [feature_names.index(name) for name in selected_names]
        hp = json.loads(row["hp_str"])
        ratio_key = str(row["pa_ratio"])
        multiplier = exp.PA_RATIO_MULT[ratio_key]
        n_pa = multiplier * len(occurrences)
        seed = cfg.RANDOM_STATE + int(row["rep"]) * 1000 + int(row["fold"]) * 17

        bg_lons, bg_lats = exp.get_training_pa(
            pa_method,
            cell_lons,
            cell_lats,
            pres_lons,
            pres_lats,
            cell_x,
            n_pa,
            seed,
        )
        bg_x = exp._sample_cell_X(
            bg_lons, bg_lats, cell_lons, cell_lats, cell_x)[:, selected_idx]
        bg_x = bg_x[np.isfinite(bg_x).all(axis=1)]
        x_pres = pres_x[:, selected_idx]
        x_train = np.vstack([x_pres, bg_x])
        y_train = np.concatenate([np.ones(len(x_pres)), np.zeros(len(bg_x))])

        model = exp.fit_rf(x_train, y_train, hp, seed)
        prediction = model.predict_proba(cell_x[:, selected_idx])[:, 1]
        predictions.append(prediction)
        full_predictions.append(
            model.predict_proba(full_x[:, selected_idx])[:, 1])
        model_records.append(
            {
                "rep": int(row["rep"]),
                "fold": int(row["fold"]),
                "outer_auc": float(row["auc_roc"]),
                "features": selected_names,
                "hyperparameters": hp,
                "pa_ratio": ratio_key,
                "n_pseudo_absence": int(len(bg_x)),
                "inner_oof_threshold": float(row["inner_threshold"]),
                "seed": seed,
            }
        )

    prediction_stack = np.vstack(predictions)
    full_prediction_stack = np.vstack(full_predictions)
    mean_values = prediction_stack.mean(axis=0)
    sd_values = prediction_stack.std(axis=0, ddof=1)
    full_mean_values = full_prediction_stack.mean(axis=0)
    full_sd_values = full_prediction_stack.std(axis=0, ddof=1)
    threshold = float(selected_folds["inner_threshold"].median())

    mean_raster = np.full((height, width), NODATA_FLOAT, dtype=np.float32)
    sd_raster = np.full((height, width), NODATA_FLOAT, dtype=np.float32)
    binary_raster = np.full((height, width), NODATA_BYTE, dtype=np.uint8)
    mean_raster[valid] = mean_values.astype(np.float32)
    sd_raster[valid] = sd_values.astype(np.float32)
    binary_raster[valid] = (mean_values >= threshold).astype(np.uint8)

    geometries = _load_boundary()
    mean_path = OUT_DIR / "Paper1_best_RF_MSAP_uncertainty_mean_Thailand.tif"
    sd_path = OUT_DIR / "Paper1_best_RF_MSAP_uncertainty_sd_Thailand.tif"
    binary_path = OUT_DIR / "Paper1_best_RF_MSAP_uncertainty_binary_Thailand.tif"
    _write_clipped(mean_raster, transform, crs, mean_path, NODATA_FLOAT, "float32", geometries)
    _write_clipped(sd_raster, transform, crs, sd_path, NODATA_FLOAT, "float32", geometries)
    _write_clipped(binary_raster, transform, crs, binary_path, NODATA_BYTE, "uint8", geometries)

    full_mean_raster = np.full((height, width), NODATA_FLOAT, dtype=np.float32)
    full_sd_raster = np.full((height, width), NODATA_FLOAT, dtype=np.float32)
    full_binary_raster = np.full((height, width), NODATA_BYTE, dtype=np.uint8)
    full_mean_raster[full_valid] = full_mean_values.astype(np.float32)
    full_sd_raster[full_valid] = full_sd_values.astype(np.float32)
    full_binary_raster[full_valid] = (full_mean_values >= threshold).astype(np.uint8)

    full_mean_path = OUT_DIR / "Paper1_best_RF_MSAP_uncertainty_mean_FullCountry_Thailand.tif"
    full_sd_path = OUT_DIR / "Paper1_best_RF_MSAP_uncertainty_sd_FullCountry_Thailand.tif"
    full_binary_path = OUT_DIR / "Paper1_best_RF_MSAP_uncertainty_binary_FullCountry_Thailand.tif"
    _write_clipped(
        full_mean_raster, transform, crs, full_mean_path,
        NODATA_FLOAT, "float32", geometries)
    _write_clipped(
        full_sd_raster, transform, crs, full_sd_path,
        NODATA_FLOAT, "float32", geometries)
    _write_clipped(
        full_binary_raster, transform, crs, full_binary_path,
        NODATA_BYTE, "uint8", geometries)

    metadata = {
        "configuration": {
            "algorithm": algo,
            "feature_set": feat_set,
            "pseudo_absence_method": pa_method,
            "pooled_oof_auc": float(best["auc_roc"]),
            "pooled_oof_auc_ci_95": [
                float(best["auc_roc_ci_lo"]),
                float(best["auc_roc_ci_hi"]),
            ],
        },
        "occurrence_count": int(len(occurrences)),
        "year_cutoff": 2024,
        "ensemble_members": int(len(predictions)),
        "binary_threshold": threshold,
        "threshold_source": "median of 15 inner-OOF Youden thresholds",
        "prediction_domains": {
            "recommended": "common valid aquatic cells within Thailand boundary",
            "full_country_context": (
                "all predictor-complete cells within Thailand; terrestrial predictions "
                "are extrapolations and are not directly interpretable as fish habitat"
            ),
        },
        "crs": str(crs),
        "nodata_float": NODATA_FLOAT,
        "nodata_binary": NODATA_BYTE,
        "models": model_records,
    }
    metadata_path = OUT_DIR / "Paper1_best_model_raster_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Best configuration: {algo}/{feat_set}/{pa_method}")
    print(f"Occurrences: {len(occurrences)}")
    print(f"Ensemble members: {len(predictions)}")
    print(f"Binary threshold: {threshold:.4f}")
    print(f"Valid aquatic cells: {valid.sum()}")
    print(f"Valid full-country cells: {full_valid.sum()}")
    for path in [
        mean_path, sd_path, binary_path,
        full_mean_path, full_sd_path, full_binary_path,
        metadata_path,
    ]:
        print(path)


if __name__ == "__main__":
    main()
