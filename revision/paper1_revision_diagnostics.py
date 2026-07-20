"""Revision diagnostics for the blackchin tilapia Paper 1 manuscript.

This script does not overwrite the original experiment. It derives reviewer-
requested diagnostics from the existing leakage-aware outputs and reconstructs
the 15 selected outer-fold RF models for grouped permutation analysis.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject
from scipy.ndimage import binary_closing, binary_dilation
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors


RANDOM_STATE = 42
N_GROUP_PERM = 100
N_BOOT = 5000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def bootstrap_mean_ci(values, rng, n_boot=N_BOOT):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return np.quantile(draws, [0.025, 0.975])


def exact_sign_flip_pvalue(differences):
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    observed = abs(differences.mean())
    if len(differences) > 20:
        rng = np.random.default_rng(RANDOM_STATE)
        signs = rng.choice([-1.0, 1.0], size=(200000, len(differences)))
    else:
        signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=len(differences))))
    null = np.abs((signs * differences).mean(axis=1))
    return (np.count_nonzero(null >= observed) + 1) / (len(null) + 1)


def load_occurrences(project):
    path = project.parent / "audit" / "output_audit" / "occurrence_merged_lit.csv"
    occurrences = pd.read_csv(path)
    occurrences["year"] = pd.to_numeric(occurrences["year"], errors="coerce")
    occurrences = occurrences[occurrences["year"] <= 2024].copy().reset_index(drop=True)
    occurrences["occ_id"] = occurrences.index
    occurrences["source_group"] = np.where(
        occurrences["source"].eq("Literature"), "Literature", "Citizen science"
    )
    return occurrences


def build_provenance_tables(project, out, occurrences):
    audit = project.parent / "audit" / "output_audit"
    summary = (
        occurrences.groupby(["source_group", "source"], dropna=False)
        .agg(
            n_records=("occ_id", "size"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            median_uncertainty_m=("coord_uncertainty_m", "median"),
        )
        .reset_index()
    )
    summary.to_csv(out / "occurrence_provenance_summary.csv", index=False)
    occurrences.to_csv(out / "occurrence_model_records.csv", index=False)

    qc_path = audit / "occurrence_qc_log.csv"
    if qc_path.exists():
        pd.read_csv(qc_path).to_csv(out / "occurrence_qc_log.csv", index=False)


def summarize_performance(project, out):
    original = project / "output"
    pooled = pd.read_csv(original / "pooled_oof_metrics.csv")
    folds = pd.read_csv(original / "experiment_fold_metrics.csv")
    oof = pd.read_csv(original / "experiment_oof_predictions.csv")
    best = pooled.sort_values("auc_roc", ascending=False).iloc[0]
    key = (best["algo"], best["feat_set"], best["pa_method"])
    selected_folds = folds[
        folds["algo"].eq(key[0])
        & folds["feat_set"].eq(key[1])
        & folds["pa_method"].eq(key[2])
    ].copy()
    selected_oof = oof[
        oof["algo"].eq(key[0])
        & oof["feat_set"].eq(key[1])
        & oof["pa_method"].eq(key[2])
    ].copy()

    prevalence = selected_oof["label"].mean()
    rng = np.random.default_rng(RANDOM_STATE)
    null_pr = []
    y = selected_oof["label"].to_numpy()
    for _ in range(5000):
        null_pr.append(average_precision_score(y, rng.random(len(y))))
    null_lo, null_hi = np.quantile(null_pr, [0.025, 0.975])

    row = {
        "algorithm": key[0],
        "feature_set": key[1],
        "pseudo_absence_method": key[2],
        "conditional_post_selection_auc": best["auc_roc"],
        "conditional_auc_ci_lo": best["auc_roc_ci_lo"],
        "conditional_auc_ci_hi": best["auc_roc_ci_hi"],
        "pr_auc": best["pr_auc"],
        "pr_auc_ci_lo": best["pr_auc_ci_lo"],
        "pr_auc_ci_hi": best["pr_auc_ci_hi"],
        "constructed_evaluation_prevalence": prevalence,
        "random_pr_auc_expectation": prevalence,
        "random_pr_auc_sim_ci_lo": null_lo,
        "random_pr_auc_sim_ci_hi": null_hi,
        "fold_auc_min": selected_folds["auc_roc"].min(),
        "fold_auc_median": selected_folds["auc_roc"].median(),
        "fold_auc_max": selected_folds["auc_roc"].max(),
        "fold_auc_sd": selected_folds["auc_roc"].std(ddof=1),
        "interpretation": (
            "Performance is conditional on selecting the highest pooled AUC among "
            "48 configurations and is not an unbiased final-test estimate."
        ),
    }
    pd.DataFrame([row]).to_csv(out / "selected_performance_context.csv", index=False)
    selected_folds.to_csv(out / "selected_fold_metrics.csv", index=False)
    return best, selected_folds, selected_oof


def partition_level_feature_comparisons(project, out):
    folds = pd.read_csv(project / "output" / "experiment_fold_metrics.csv")
    block = (
        folds.groupby(["rep", "fold", "feat_set"], as_index=False)["auc_roc"]
        .mean()
        .pivot(index=["rep", "fold"], columns="feat_set", values="auc_roc")
        .reset_index()
    )
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for target in ["M_SA", "M_SP", "M_SAP"]:
        diff = block[target] - block["M_S"]
        lo, hi = bootstrap_mean_ci(diff, rng)
        rows.append(
            {
                "contrast": f"{target} - M_S",
                "independent_analysis_unit": "complete outer spatial partition",
                "n_partitions": len(diff),
                "mean_auc_difference": diff.mean(),
                "ci_lo": lo,
                "ci_hi": hi,
                "exact_sign_flip_p": exact_sign_flip_pvalue(diff),
            }
        )
    pd.DataFrame(rows).to_csv(out / "partition_level_feature_contrasts.csv", index=False)
    block.to_csv(out / "partition_level_feature_auc.csv", index=False)


def import_experiment(project):
    sys.path.insert(0, str(project.parent))
    return importlib.import_module("blackchin_tilapia_analysis.scripts.20_full_experiment_v2")


def reconstruct_selected_models(project, selected_folds, occurrences, exp):
    pooled = pd.read_csv(project / "output" / "pooled_oof_metrics.csv")
    best = pooled.sort_values("auc_roc", ascending=False).iloc[0]
    key = (str(best["algo"]), str(best["feat_set"]), str(best["pa_method"]))
    arrays, transform, _ = exp.load_all_rasters()
    feature_names = exp.MODEL_VARIANTS[key[1]]
    common_lons, common_lats = exp.get_common_valid_cells(arrays, transform)
    cell_x = exp.sample_at_coords(arrays, feature_names, common_lons, common_lats, transform)
    models = []

    for _, spec in selected_folds.sort_values(["rep", "fold"]).iterrows():
        rep = int(spec["rep"])
        fold = int(spec["fold"])
        rep_seed = RANDOM_STATE + (rep - 1) * 100
        blocked, _, partition = exp.assign_spatial_folds_stable(
            occurrences, k=3, base_seed=rep_seed
        )
        cell_block = exp.assign_cells_to_blocks(common_lons, common_lats, partition)
        train_presence = blocked[blocked["spatial_fold"] != fold]
        test_presence = blocked[blocked["spatial_fold"] == fold]
        train_cells = cell_block != (fold - 1)
        test_cells = cell_block == (fold - 1)

        selected_names = [x for x in str(spec["selected"]).split(",") if x]
        selected_idx = [feature_names.index(x) for x in selected_names]
        hp = json.loads(spec["hp_str"])
        n_pa = int(spec["n_pa_actual"])

        train_lons = common_lons[train_cells]
        train_lats = common_lats[train_cells]
        train_cell_x = cell_x[train_cells]
        bg_lons, bg_lats = exp.get_training_pa(
            key[2],
            train_lons,
            train_lats,
            train_presence["longitude"].to_numpy(),
            train_presence["latitude"].to_numpy(),
            train_cell_x,
            n_pa,
            rep_seed + fold,
        )
        bg_x = exp._sample_cell_X(
            bg_lons, bg_lats, train_lons, train_lats, train_cell_x
        )[:, selected_idx]
        bg_x = bg_x[np.isfinite(bg_x).all(axis=1)]
        presence_train_x = exp.sample_at_coords(
            arrays,
            feature_names,
            train_presence["longitude"].to_numpy(),
            train_presence["latitude"].to_numpy(),
            transform,
        )[:, selected_idx]
        presence_train_x = presence_train_x[np.isfinite(presence_train_x).all(axis=1)]
        x_train = np.vstack([presence_train_x, bg_x])
        y_train = np.concatenate([np.ones(len(presence_train_x)), np.zeros(len(bg_x))])
        model = exp.fit_rf(x_train, y_train, hp, rep_seed)

        test_lons = common_lons[test_cells]
        test_lats = common_lats[test_cells]
        n_eval = min(exp.N_EVAL_BG, len(test_lons))
        eval_rng = np.random.default_rng(rep_seed + fold * 97 + 1000)
        eval_idx = eval_rng.choice(len(test_lons), n_eval, replace=False)
        presence_test_x = exp.sample_at_coords(
            arrays,
            feature_names,
            test_presence["longitude"].to_numpy(),
            test_presence["latitude"].to_numpy(),
            transform,
        )[:, selected_idx]
        background_test_x = exp.sample_at_coords(
            arrays,
            feature_names,
            test_lons[eval_idx],
            test_lats[eval_idx],
            transform,
        )[:, selected_idx]
        x_test = np.vstack([presence_test_x, background_test_x])
        y_test = np.concatenate(
            [np.ones(len(presence_test_x)), np.zeros(len(background_test_x))]
        )
        valid = np.isfinite(x_test).all(axis=1)
        x_test = x_test[valid]
        y_test = y_test[valid]
        baseline = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
        models.append(
            {
                "rep": rep,
                "fold": fold,
                "model": model,
                "x_test": x_test,
                "y_test": y_test,
                "selected_names": selected_names,
                "baseline_auc": baseline,
                "reported_auc": float(spec["auc_roc"]),
                "x_train": x_train,
                "selected_idx": selected_idx,
            }
        )
    return models, common_lons, common_lats, cell_x, transform


def grouped_permutation(out, models):
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for item in models:
        group_columns = {
            "S_environment": [
                i
                for i, name in enumerate(item["selected_names"])
                if not name.startswith(("A", "P"))
            ],
            "A_waterway_context": [
                i for i, name in enumerate(item["selected_names"]) if name.startswith("A")
            ],
            "P_human_activity": [
                i for i, name in enumerate(item["selected_names"]) if name.startswith("P")
            ],
        }
        for group, columns in group_columns.items():
            decreases = []
            if columns:
                for _ in range(N_GROUP_PERM):
                    x_perm = item["x_test"].copy()
                    order = rng.permutation(len(x_perm))
                    x_perm[:, columns] = x_perm[order][:, columns]
                    auc = roc_auc_score(
                        item["y_test"], item["model"].predict_proba(x_perm)[:, 1]
                    )
                    decreases.append(item["baseline_auc"] - auc)
            else:
                decreases = [0.0]
            rows.append(
                {
                    "rep": item["rep"],
                    "fold": item["fold"],
                    "group": group,
                    "n_selected_features": len(columns),
                    "baseline_auc_reconstructed": item["baseline_auc"],
                    "reported_auc": item["reported_auc"],
                    "grouped_permutation_auc_decrease": np.mean(decreases),
                }
            )
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(out / "grouped_permutation_per_fold.csv", index=False)
    summary = []
    for group, sub in per_fold.groupby("group"):
        lo, hi = bootstrap_mean_ci(
            sub["grouped_permutation_auc_decrease"],
            np.random.default_rng(RANDOM_STATE),
        )
        summary.append(
            {
                "group": group,
                "mean_grouped_auc_decrease": sub[
                    "grouped_permutation_auc_decrease"
                ].mean(),
                "ci_lo": lo,
                "ci_hi": hi,
                "mean_selected_features": sub["n_selected_features"].mean(),
                "n_outer_models": len(sub),
            }
        )
    pd.DataFrame(summary).sort_values(
        "mean_grouped_auc_decrease", ascending=False
    ).to_csv(out / "grouped_permutation_summary.csv", index=False)


def area_of_applicability(project, out, models, common_lons, common_lats, cell_x, transform):
    in_aoa = np.zeros((len(models), len(common_lons)), dtype=np.uint8)
    dissimilarity = np.full((len(models), len(common_lons)), np.nan, dtype=np.float32)
    for model_idx, item in enumerate(models):
        train = np.asarray(item["x_train"], dtype=float)
        pred = np.asarray(cell_x[:, item["selected_idx"]], dtype=float)
        mean = np.nanmean(train, axis=0)
        sd = np.nanstd(train, axis=0)
        sd[sd == 0] = 1.0
        train_z = (train - mean) / sd
        pred_z = (pred - mean) / sd
        train_nn = NearestNeighbors(n_neighbors=min(2, len(train))).fit(train_z)
        train_dist = train_nn.kneighbors(train_z, return_distance=True)[0]
        reference_dist = train_dist[:, -1]
        threshold = np.quantile(reference_dist, 0.95)
        pred_nn = NearestNeighbors(n_neighbors=1).fit(train_z)
        pred_dist = pred_nn.kneighbors(pred_z, return_distance=True)[0][:, 0]
        dissimilarity[model_idx] = pred_dist
        in_aoa[model_idx] = pred_dist <= threshold

    aoa_fraction = in_aoa.mean(axis=0)
    mean_di = np.nanmean(dissimilarity, axis=0)
    with rasterio.open(project.parent / "data" / "environmental" / "clipped" / "bio01.tif") as ref:
        profile = ref.profile.copy()
        height, width = ref.height, ref.width
        ref_crs = ref.crs
    boundary = gpd.read_file(project.parent / "data" / "thailand_boundary.geojson")
    boundary = boundary.to_crs(ref_crs)
    country_mask = geometry_mask(
        boundary.geometry,
        out_shape=(height, width),
        transform=transform,
        invert=True,
    )
    profile.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")
    rows = ((common_lats - transform.f) / transform.e).astype(int)
    cols = ((common_lons - transform.c) / transform.a).astype(int)
    aoa_grid = np.full((height, width), -9999.0, dtype=np.float32)
    di_grid = np.full((height, width), -9999.0, dtype=np.float32)
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    inside_thailand = np.zeros(len(valid), dtype=bool)
    inside_thailand[valid] = country_mask[rows[valid], cols[valid]]
    valid_country = valid & inside_thailand
    aoa_grid[rows[valid_country], cols[valid_country]] = aoa_fraction[valid_country]
    di_grid[rows[valid_country], cols[valid_country]] = mean_di[valid_country]
    with rasterio.open(out / "selected_model_aoa_fraction.tif", "w", **profile) as dst:
        dst.write(aoa_grid, 1)
    with rasterio.open(out / "selected_model_mean_dissimilarity.tif", "w", **profile) as dst:
        dst.write(di_grid, 1)

    extent = [
        transform.c,
        transform.c + width * transform.a,
        transform.f + height * transform.e,
        transform.f,
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax in axes:
        boundary.plot(
            ax=ax,
            facecolor="#f2f2f2",
            edgecolor="black",
            linewidth=0.7,
            zorder=0,
        )
    im0 = axes[0].imshow(
        np.where(aoa_grid == -9999, np.nan, aoa_grid),
        extent=extent,
        origin="upper",
        vmin=0,
        vmax=1,
        cmap="viridis",
        zorder=1,
    )
    boundary.boundary.plot(ax=axes[0], color="black", linewidth=0.7, zorder=2)
    axes[0].set_title("Fraction of 15 outer models within AOA")
    fig.colorbar(im0, ax=axes[0], fraction=0.035)
    shown_di = np.where(di_grid == -9999, np.nan, di_grid)
    vmax = np.nanquantile(shown_di, 0.98)
    im1 = axes[1].imshow(
        shown_di,
        extent=extent,
        origin="upper",
        vmin=0,
        vmax=vmax,
        cmap="magma",
        zorder=1,
    )
    boundary.boundary.plot(ax=axes[1], color="black", linewidth=0.7, zorder=2)
    axes[1].set_title("Mean predictor-space dissimilarity")
    fig.colorbar(im1, ax=axes[1], fraction=0.035)
    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        minx, miny, maxx, maxy = boundary.total_bounds
        ax.set_xlim(minx - 0.2, maxx + 0.2)
        ax.set_ylim(miny - 0.2, maxy + 0.2)
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out / "Fig_revision_area_of_applicability.png", dpi=300)
    fig.savefig(out / "Fig_revision_area_of_applicability.pdf")
    plt.close(fig)
    pd.DataFrame(
        [
            {
                "common_prediction_cells": int(valid_country.sum()),
                "fraction_cells_supported_by_all_15_models": np.mean(
                    aoa_fraction[valid_country] == 1
                ),
                "fraction_cells_supported_by_at_least_12_models": np.mean(
                    aoa_fraction[valid_country] >= 0.8
                ),
                "fraction_cells_supported_by_fewer_than_8_models": np.mean(
                    aoa_fraction[valid_country] < (8 / 15)
                ),
                "method": (
                    "Per-fold standardized nearest-neighbour dissimilarity; AOA threshold "
                    "is the 95th percentile of second-neighbour distances within each "
                    "fold-specific training set. Summary and rasters are clipped to the "
                    "Thailand national boundary."
                ),
            }
        ]
    ).to_csv(out / "area_of_applicability_summary.csv", index=False)


def source_sensitivity(out, selected_oof, occurrences):
    source = occurrences[["occ_id", "source_group", "source"]]
    merged = selected_oof.merge(source, on="occ_id", how="left")
    backgrounds = merged[merged["label"].eq(0)].copy()
    rows = []
    for group in ["All presences", "Citizen science", "Literature"]:
        if group == "All presences":
            pres = merged[merged["label"].eq(1)]
        else:
            pres = merged[
                merged["label"].eq(1) & merged["source_group"].eq(group)
            ]
        combined = pd.concat([pres, backgrounds], ignore_index=True)
        if pres["occ_id"].nunique() < 2:
            auc = pr = np.nan
        else:
            auc = roc_auc_score(combined["label"], combined["suit_oof"])
            pr = average_precision_score(combined["label"], combined["suit_oof"])
        rows.append(
            {
                "presence_subset": group,
                "unique_occurrences": pres["occ_id"].nunique(),
                "repeated_presence_rows": len(pres),
                "background_rows": len(backgrounds),
                "auc": auc,
                "pr_auc": pr,
                "constructed_prevalence": combined["label"].mean(),
            }
        )
    pd.DataFrame(rows).to_csv(out / "observation_source_sensitivity.csv", index=False)


def morans_i_presence_residuals(out, selected_oof, occurrences, threshold_km=100):
    pres = selected_oof[selected_oof["label"].eq(1)].copy()
    averaged = (
        pres.groupby("occ_id", as_index=False)["suit_oof"]
        .mean()
        .merge(occurrences[["occ_id", "longitude", "latitude"]], on="occ_id")
    )
    averaged["residual"] = 1.0 - averaged["suit_oof"]
    lon = np.radians(averaged["longitude"].to_numpy())
    lat = np.radians(averaged["latitude"].to_numpy())
    dlon = lon[:, None] - lon[None, :]
    dlat = lat[:, None] - lat[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(
        dlon / 2
    ) ** 2
    distance = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    weights = ((distance > 0) & (distance <= threshold_km)).astype(float)
    x = averaged["residual"].to_numpy()
    z = x - x.mean()
    s0 = weights.sum()
    observed = (
        len(x) / s0 * np.sum(weights * z[:, None] * z[None, :]) / np.sum(z**2)
        if s0 > 0
        else np.nan
    )
    rng = np.random.default_rng(RANDOM_STATE)
    null = []
    for _ in range(9999):
        zp = rng.permutation(z)
        null.append(
            len(x)
            / s0
            * np.sum(weights * zp[:, None] * zp[None, :])
            / np.sum(zp**2)
        )
    null = np.asarray(null)
    p = (np.count_nonzero(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1)
    pd.DataFrame(
        [
            {
                "n_unique_occurrences": len(x),
                "distance_threshold_km": threshold_km,
                "morans_i": observed,
                "permutation_p_two_sided": p,
                "expected_i": -1 / (len(x) - 1),
                "weight_links": int(s0),
            }
        ]
    ).to_csv(out / "residual_morans_i.csv", index=False)
    averaged.to_csv(out / "presence_oof_residuals.csv", index=False)


def province_summary(project, out):
    raster = (
        project
        / "output"
        / "best_model_rasters"
        / "Paper1_best_RF_MSAP_mean_AquaticNetwork_Thailand.tif"
    )
    provinces_path = project.parent / "data" / "thailand_provinces.geojson"
    provinces = gpd.read_file(provinces_path)
    with rasterio.open(raster) as src:
        provinces = provinces.to_crs(src.crs)
        data = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        rows = []
        name_col = next(
            (
                x
                for x in ["name_en", "NAME_1", "shapeName", "name", "ADM1_EN"]
                if x in provinces.columns
            ),
            None,
        )
        for idx, feature in provinces.iterrows():
            mask = geometry_mask(
                [feature.geometry],
                out_shape=data.shape,
                transform=src.transform,
                invert=True,
            )
            values = data[mask & np.isfinite(data)]
            if len(values) == 0:
                continue
            rows.append(
                {
                    "province": feature[name_col] if name_col else str(idx),
                    "aquatic_cells": len(values),
                    "mean_relative_suitability": values.mean(),
                    "median_relative_suitability": np.median(values),
                    "p90_relative_suitability": np.quantile(values, 0.9),
                    "fraction_cells_ge_0_5": np.mean(values >= 0.5),
                }
            )
    frame = pd.DataFrame(rows).sort_values(
        ["mean_relative_suitability", "p90_relative_suitability"], ascending=False
    )
    frame.to_csv(out / "province_relative_suitability.csv", index=False)
    frame.head(15).to_csv(out / "province_relative_suitability_top15.csv", index=False)


def plot_suitability_uncertainty(project, out):
    raster_dir = project / "output" / "best_model_rasters"
    mean_path = raster_dir / "Paper1_best_RF_MSAP_mean_AquaticNetwork_Thailand.tif"
    sd_path = raster_dir / "Paper1_best_RF_MSAP_sd_AquaticNetwork_Thailand.tif"
    arrays = []
    extent = None
    raster_transform = None
    raster_crs = None
    for path in [mean_path, sd_path]:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(float)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            arrays.append(arr)
            bounds = src.bounds
            extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
            raster_transform = src.transform
            raster_crs = src.crs
    boundary = gpd.read_file(project.parent / "data" / "thailand_boundary.geojson")
    boundary = boundary.to_crs(raster_crs)
    with rasterio.open(out / "selected_model_aoa_fraction.tif") as src:
        aoa = np.full(arrays[0].shape, -9999.0, dtype=np.float32)
        reproject(
            source=src.read(1),
            destination=aoa,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=raster_transform,
            dst_crs=raster_crs,
            dst_nodata=-9999.0,
            resampling=Resampling.nearest,
        )
        aoa[aoa == -9999.0] = np.nan
    support = np.isfinite(aoa) & (aoa >= (8 / 15)) & np.isfinite(arrays[0])
    generalized_support = binary_closing(
        binary_dilation(support, iterations=3),
        iterations=2,
    )
    country_mask = geometry_mask(
        boundary.geometry,
        out_shape=generalized_support.shape,
        transform=raster_transform,
        invert=True,
    )
    generalized_support &= country_mask
    fig, axes = plt.subplots(1, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax in axes:
        boundary.plot(
            ax=ax,
            facecolor="#f2f2f2",
            edgecolor="black",
            linewidth=0.7,
            zorder=0,
        )
    im0 = axes[0].imshow(
        arrays[0],
        extent=extent,
        origin="upper",
        vmin=0,
        vmax=1,
        cmap="viridis",
        zorder=1,
    )
    axes[0].contour(
        generalized_support.astype(float),
        levels=[0.5],
        extent=extent,
        origin="upper",
        colors="#FF0000",
        linewidths=1.2,
        linestyles="--",
        zorder=2,
    )
    boundary.boundary.plot(ax=axes[0], color="black", linewidth=0.7, zorder=3)
    axes[0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#FF0000",
                linestyle="--",
                linewidth=1.5,
                label="Generalized AOA support (>=8/15 models)",
            )
        ],
        loc="lower left",
        fontsize=8,
        framealpha=0.9,
    )
    axes[0].set_title("Ensemble mean relative suitability")
    fig.colorbar(im0, ax=axes[0], fraction=0.035)
    im1 = axes[1].imshow(
        arrays[1],
        extent=extent,
        origin="upper",
        vmin=0,
        vmax=np.nanquantile(arrays[1], 0.99),
        cmap="magma",
        zorder=1,
    )
    boundary.boundary.plot(ax=axes[1], color="black", linewidth=0.7, zorder=2)
    axes[1].set_title("Between-model standard deviation")
    fig.colorbar(im1, ax=axes[1], fraction=0.035)
    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        minx, miny, maxx, maxy = boundary.total_bounds
        ax.set_xlim(minx - 0.2, maxx + 0.2)
        ax.set_ylim(miny - 0.2, maxy + 0.2)
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out / "Fig_revision_suitability_uncertainty.png", dpi=300)
    fig.savefig(out / "Fig_revision_suitability_uncertainty.pdf")
    plt.close(fig)


def plot_occurrences_and_folds(project, out, occurrences, exp):
    boundary = gpd.read_file(project.parent / "data" / "thailand_boundary.geojson")
    points = gpd.GeoDataFrame(
        occurrences,
        geometry=gpd.points_from_xy(occurrences.longitude, occurrences.latitude),
        crs="EPSG:4326",
    )
    blocked, _, _ = exp.assign_spatial_folds_stable(
        occurrences, k=3, base_seed=RANDOM_STATE
    )
    points["fold_example"] = blocked["spatial_fold"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 8), sharex=True, sharey=True)
    boundary.plot(ax=axes[0], facecolor="#f5f5f5", edgecolor="black", linewidth=0.6)
    for label, color, marker in [
        ("Citizen science", "#2166ac", "o"),
        ("Literature", "#b2182b", "^"),
    ]:
        sub = points[points["source_group"].eq(label)]
        sub.plot(ax=axes[0], color=color, marker=marker, markersize=28, label=label)
    axes[0].legend()
    axes[0].set_title("Modelled occurrences by provenance")
    boundary.plot(ax=axes[1], facecolor="#f5f5f5", edgecolor="black", linewidth=0.6)
    points.plot(
        ax=axes[1],
        column="fold_example",
        categorical=True,
        cmap="Set1",
        markersize=28,
        legend=True,
    )
    axes[1].set_title("Example outer spatial partition (repeat 1)")
    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(out / "Fig_revision_occurrence_folds.png", dpi=300)
    fig.savefig(out / "Fig_revision_occurrence_folds.pdf")
    plt.close(fig)


def plot_ale_with_support(project, out):
    xai = project / "output" / "factor_explainability"
    ale = pd.read_csv(xai / "ale_curves.csv")
    shap = pd.read_csv(xai / "shap_values_long.csv")
    order = [
        "bio07",
        "elevation",
        "bio10",
        "bio04",
        "P1_pop_density",
        "A2_waterway_order",
    ]
    available = [x for x in order if x in set(ale["feature"])]
    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
    for ax, feature in zip(axes.ravel(), available):
        sub = ale[ale["feature"].eq(feature)].sort_values("x")
        vals = shap.loc[shap["feature"].eq(feature), "value"].dropna().to_numpy()
        ax.plot(sub["x"], sub["ale_mean"], color="#2166ac", linewidth=2)
        ax.fill_between(
            sub["x"],
            sub["ale_ci_lo"],
            sub["ale_ci_hi"],
            color="#92c5de",
            alpha=0.45,
        )
        if len(vals):
            ymin, ymax = ax.get_ylim()
            rug_y = ymin + 0.02 * (ymax - ymin)
            ax.plot(vals, np.full(len(vals), rug_y), "|", color="black", alpha=0.08)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(sub["label"].iloc[0])
        ax.set_xlabel("Predictor value; rug shows held-out support")
        ax.set_ylabel("Accumulated local effect")
    fig.tight_layout()
    fig.savefig(out / "Fig_revision_ALE_with_support.png", dpi=300)
    fig.savefig(out / "Fig_revision_ALE_with_support.pdf")
    plt.close(fig)


def write_environment(out):
    modules = [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "rasterio",
        "geopandas",
        "shapely",
        "matplotlib",
        "xgboost",
        "shap",
        "docx",
    ]
    rows = []
    for name in modules:
        module = importlib.import_module(name)
        rows.append({"package": name, "version": getattr(module, "__version__", "")})
    pd.DataFrame(rows).to_csv(out / "software_environment.csv", index=False)


def plot_uncertainty_zone_workflow(out):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    boxes = [
        (0.4, 5.3, 3.0, 1.0, "1. Training split only\npresences + eligible cells", "#DCEAF7"),
        (4.5, 5.3, 3.0, 1.0, "2. Exclude cells <10 km\nfrom training presences", "#DCEAF7"),
        (8.6, 5.3, 3.0, 1.0, "3. Draw <=200 pilot\nbackground cells", "#DCEAF7"),
        (8.6, 3.1, 3.0, 1.0, "4. Fit pilot RF\n50 trees; split-specific seed", "#FFF2CC"),
        (4.5, 3.1, 3.0, 1.0, "5. Score every eligible cell\nmissing predictor score = 0.4", "#FFF2CC"),
        (0.4, 3.1, 3.0, 1.0, "6. Candidate zone\npilot score 0.20-0.60", "#E2F0D9"),
        (0.4, 0.8, 3.0, 1.0, "7a. If candidates >= n/2\nsample requested n", "#E2F0D9"),
        (4.5, 0.8, 3.0, 1.0, "7b. Otherwise use all cells\noutside the 10-km buffer", "#FCE4D6"),
        (8.6, 0.8, 3.0, 1.0, "8. Sample without replacement\nup to n pseudo-absences", "#D9EAD3"),
    ]
    for x, y, w, h, text, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=color,
            edgecolor="#1F4E79",
            linewidth=1.2,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    arrows = [
        ((3.4, 5.8), (4.5, 5.8)),
        ((7.5, 5.8), (8.6, 5.8)),
        ((10.1, 5.3), (10.1, 4.1)),
        ((8.6, 3.6), (7.5, 3.6)),
        ((4.5, 3.6), (3.4, 3.6)),
        ((1.9, 3.1), (1.9, 1.8)),
        ((7.5, 1.3), (8.6, 1.3)),
    ]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="-|>", color="#1F4E79", lw=1.4),
        )
    ax.annotate(
        "",
        xy=(8.6, 1.55),
        xytext=(3.4, 1.55),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#548235",
            lw=1.2,
            connectionstyle="arc3,rad=-0.18",
        ),
    )
    ax.annotate(
        "",
        xy=(6.0, 1.8),
        xytext=(2.8, 3.1),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#C65911",
            lw=1.2,
            connectionstyle="arc3,rad=-0.18",
        ),
    )
    ax.text(1.55, 2.4, "Yes", fontsize=9, color="#548235")
    ax.text(4.2, 2.35, "No", fontsize=9, color="#C65911")
    ax.text(
        6,
        6.75,
        "Training-only uncertainty-zone pseudo-absence algorithm",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        6,
        0.2,
        "n is selected inside inner cross-validation from 1:1, 1:3, 1:5, or 1:10 "
        "times the number of training presences. Held-out presences and fixed evaluation "
        "backgrounds never enter this workflow.",
        ha="center",
        va="center",
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout()
    fig.savefig(out / "Fig_method_uncertainty_zone_workflow.png", dpi=300)
    fig.savefig(out / "Fig_method_uncertainty_zone_workflow.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    project = args.project.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    occurrences = load_occurrences(project)
    build_provenance_tables(project, out, occurrences)
    _, selected_folds, selected_oof = summarize_performance(project, out)
    partition_level_feature_comparisons(project, out)
    source_sensitivity(out, selected_oof, occurrences)
    morans_i_presence_residuals(out, selected_oof, occurrences)
    province_summary(project, out)
    exp = import_experiment(project)
    models, common_lons, common_lats, cell_x, transform = reconstruct_selected_models(
        project, selected_folds, occurrences, exp
    )
    grouped_permutation(out, models)
    area_of_applicability(
        project, out, models, common_lons, common_lats, cell_x, transform
    )
    plot_suitability_uncertainty(project, out)
    plot_occurrences_and_folds(project, out, occurrences, exp)
    plot_ale_with_support(project, out)
    plot_uncertainty_zone_workflow(out)
    write_environment(out)
    print(out)


if __name__ == "__main__":
    main()
