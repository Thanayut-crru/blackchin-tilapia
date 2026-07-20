"""Leakage-aware factor interpretation for the best validated Paper 1 model."""

import importlib
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

exp = importlib.import_module("blackchin_tilapia_analysis.scripts.20_full_experiment_v2")

OUT = ROOT / "output"
OUT_XAI = OUT / "factor_explainability"
FIG_DIR = OUT_XAI / "figures"
OCC_CSV = PROJECT / "audit" / "output_audit" / "occurrence_merged_lit.csv"
FOLD_CSV = OUT / "experiment_fold_metrics.csv"
POOL_CSV = OUT / "pooled_oof_metrics.csv"

N_PERM = 30
N_BOOT = 2000
N_SHAP_PER_FOLD = 300
N_ALE_SAMPLE = 2500
N_ALE_BINS = 10
RANDOM_STATE = 42

LABELS = {
    "bio01": "Annual mean temperature",
    "bio04": "Temperature seasonality",
    "bio07": "Annual temperature range",
    "bio10": "Mean temperature, warmest quarter",
    "bio11": "Mean temperature, coldest quarter",
    "bio12": "Annual precipitation",
    "bio15": "Precipitation seasonality",
    "bio16": "Precipitation, wettest quarter",
    "bio17": "Precipitation, driest quarter",
    "elevation": "Elevation",
    "A1_dist_waterway": "Distance to waterway",
    "A2_waterway_order": "Waterway order",
    "P1_pop_density": "Population density",
    "P2_dist_road": "Distance to road",
    "P3_road_density": "Road density",
    "P4_dist_urban": "Distance to urban area",
}


def feature_group(name):
    if name.startswith("A"):
        return "A: waterway accessibility"
    if name.startswith("P"):
        return "P: human-pressure proxy"
    return "S: environmental suitability"


def bootstrap_ci(values, rng):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    draws = rng.choice(values, size=(N_BOOT, len(values)), replace=True).mean(axis=1)
    return np.quantile(draws, [0.025, 0.975])


def extract_shap_class1(model, x):
    import shap
    values = shap.TreeExplainer(model).shap_values(x)
    if isinstance(values, list):
        return np.asarray(values[-1])
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, -1]
    return values


def ale_curve(model, x_common, selected_idx, feature_idx, edges):
    local_position = selected_idx.index(feature_idx)
    x_selected = x_common[:, selected_idx].copy()
    original = x_common[:, feature_idx]
    bins = np.clip(np.digitize(original, edges[1:-1]), 0, len(edges) - 2)
    effects = np.full(len(edges) - 1, np.nan)
    weights = np.zeros(len(edges) - 1, dtype=int)

    for bin_id in range(len(effects)):
        idx = np.where(bins == bin_id)[0]
        if len(idx) == 0:
            continue
        low = x_selected[idx].copy()
        high = x_selected[idx].copy()
        low[:, local_position] = edges[bin_id]
        high[:, local_position] = edges[bin_id + 1]
        effects[bin_id] = np.mean(
            model.predict_proba(high)[:, 1] - model.predict_proba(low)[:, 1])
        weights[bin_id] = len(idx)

    effects = np.nan_to_num(effects, nan=0.0)
    accumulated = np.cumsum(effects)
    if weights.sum() > 0:
        accumulated -= np.average(accumulated, weights=np.maximum(weights, 1))
    return accumulated


def main():
    OUT_XAI.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    pooled = pd.read_csv(POOL_CSV)
    best = pooled.sort_values("auc_roc", ascending=False).iloc[0]
    key = (str(best["algo"]), str(best["feat_set"]), str(best["pa_method"]))
    if key[0] != "RF":
        raise ValueError(f"Expected RF as best algorithm, found {key[0]}")

    folds = pd.read_csv(FOLD_CSV)
    specs = folds[
        (folds["algo"] == key[0])
        & (folds["feat_set"] == key[1])
        & (folds["pa_method"] == key[2])
    ].sort_values(["rep", "fold"])
    if len(specs) != 15:
        raise ValueError(f"Expected 15 fold specifications, found {len(specs)}")

    occurrences = pd.read_csv(OCC_CSV)
    occurrences["year"] = pd.to_numeric(occurrences["year"], errors="coerce")
    occurrences = occurrences[occurrences["year"] <= 2024].copy().reset_index(drop=True)
    occurrences["occ_id"] = occurrences.index

    arrays, transform, _ = exp.load_all_rasters()
    feature_names = exp.MODEL_VARIANTS[key[1]]
    common_lons, common_lats = exp.get_common_valid_cells(arrays, transform)
    cell_x = exp.sample_at_coords(
        arrays, feature_names, common_lons, common_lats, transform)

    fold_rows = []
    shap_rows = []
    models = []

    for _, spec in specs.iterrows():
        rep = int(spec["rep"])
        fold = int(spec["fold"])
        rep_seed = RANDOM_STATE + (rep - 1) * 100
        blocked, _, partition = exp.assign_spatial_folds_stable(
            occurrences, k=3, base_seed=rep_seed)
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
            bg_lons, bg_lats, train_lons, train_lats, train_cell_x)[:, selected_idx]
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
        y_train = np.concatenate([
            np.ones(len(presence_train_x)), np.zeros(len(bg_x))])
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
        y_test = np.concatenate([
            np.ones(len(presence_test_x)), np.zeros(len(background_test_x))])
        valid_test = np.isfinite(x_test).all(axis=1)
        x_test = x_test[valid_test]
        y_test = y_test[valid_test]
        baseline_auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])

        importance = {name: 0.0 for name in feature_names}
        for local_idx, name in enumerate(selected_names):
            decreases = []
            for perm in range(N_PERM):
                x_perm = x_test.copy()
                order = np.random.default_rng(
                    rep_seed + fold * 1000 + local_idx * 100 + perm
                ).permutation(len(x_perm))
                x_perm[:, local_idx] = x_perm[order, local_idx]
                decreases.append(
                    baseline_auc
                    - roc_auc_score(y_test, model.predict_proba(x_perm)[:, 1])
                )
            importance[name] = float(np.mean(decreases))

        shap_n = min(N_SHAP_PER_FOLD, len(x_test))
        shap_idx = rng.choice(len(x_test), shap_n, replace=False)
        shap_x = x_test[shap_idx]
        shap_values = extract_shap_class1(model, shap_x)
        shap_common = np.zeros((shap_n, len(feature_names)))
        x_common = np.full((shap_n, len(feature_names)), np.nan)
        shap_common[:, selected_idx] = shap_values
        x_common[:, selected_idx] = shap_x

        for feature_idx, name in enumerate(feature_names):
            selected = name in selected_names
            vals = shap_common[:, feature_idx]
            xs = x_common[:, feature_idx]
            rho = np.nan
            if selected and np.nanstd(xs) > 0 and np.nanstd(vals) > 0:
                rho = spearmanr(xs, vals, nan_policy="omit").statistic
            fold_rows.append({
                "rep": rep,
                "fold": fold,
                "feature": name,
                "label": LABELS[name],
                "group": feature_group(name),
                "selected": int(selected),
                "baseline_auc_reconstructed": baseline_auc,
                "reported_auc": float(spec["auc_roc"]),
                "permutation_auc_decrease": importance[name],
                "mean_abs_shap": float(np.mean(np.abs(vals))),
                "shap_direction_rho": rho,
            })
            if selected:
                for value, shap_value in zip(xs, vals):
                    shap_rows.append({
                        "rep": rep,
                        "fold": fold,
                        "feature": name,
                        "value": float(value),
                        "shap_value": float(shap_value),
                    })

        models.append({
            "rep": rep,
            "fold": fold,
            "model": model,
            "selected_names": selected_names,
            "selected_idx": selected_idx,
            "reported_auc": float(spec["auc_roc"]),
        })
        print(
            f"rep={rep} fold={fold}: reported AUC={float(spec['auc_roc']):.3f}, "
            f"reconstructed={baseline_auc:.3f}, features={len(selected_names)}"
        )

    per_fold = pd.DataFrame(fold_rows)
    per_fold.to_csv(OUT_XAI / "feature_importance_per_fold.csv", index=False)
    pd.DataFrame(shap_rows).to_csv(OUT_XAI / "shap_values_long.csv", index=False)

    summary_rows = []
    for name in feature_names:
        sub = per_fold[per_fold["feature"] == name]
        perm_lo, perm_hi = bootstrap_ci(sub["permutation_auc_decrease"], rng)
        shap_lo, shap_hi = bootstrap_ci(sub["mean_abs_shap"], rng)
        selected_rho = sub.loc[sub["selected"] == 1, "shap_direction_rho"].dropna()
        summary_rows.append({
            "feature": name,
            "label": LABELS[name],
            "group": feature_group(name),
            "selection_frequency": sub["selected"].mean(),
            "permutation_auc_decrease_mean": sub["permutation_auc_decrease"].mean(),
            "permutation_ci_lo": perm_lo,
            "permutation_ci_hi": perm_hi,
            "mean_abs_shap": sub["mean_abs_shap"].mean(),
            "shap_ci_lo": shap_lo,
            "shap_ci_hi": shap_hi,
            "shap_direction_rho": selected_rho.mean() if len(selected_rho) else np.nan,
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        "permutation_auc_decrease_mean", ascending=False)
    summary.to_csv(OUT_XAI / "feature_importance_summary.csv", index=False)

    group_summary = summary.groupby("group", as_index=False).agg(
        permutation_auc_decrease=("permutation_auc_decrease_mean", "sum"),
        mean_abs_shap=("mean_abs_shap", "sum"),
        mean_selection_frequency=("selection_frequency", "mean"),
    )
    group_summary.to_csv(OUT_XAI / "factor_group_summary.csv", index=False)

    top_features = summary.head(6)["feature"].tolist()
    sample_idx = rng.choice(
        len(cell_x), min(N_ALE_SAMPLE, len(cell_x)), replace=False)
    ale_x = cell_x[sample_idx]
    ale_rows = []
    for name in top_features:
        feature_idx = feature_names.index(name)
        edges = np.unique(np.quantile(
            ale_x[:, feature_idx], np.linspace(0, 1, N_ALE_BINS + 1)))
        if len(edges) < 4:
            continue
        curves = []
        for item in models:
            if feature_idx not in item["selected_idx"]:
                continue
            curves.append(ale_curve(
                item["model"], ale_x, item["selected_idx"], feature_idx, edges))
        if not curves:
            continue
        curves = np.vstack(curves)
        centers = (edges[:-1] + edges[1:]) / 2
        for i, center in enumerate(centers):
            ale_rows.append({
                "feature": name,
                "label": LABELS[name],
                "x": center,
                "ale_mean": curves[:, i].mean(),
                "ale_ci_lo": np.quantile(curves[:, i], 0.025),
                "ale_ci_hi": np.quantile(curves[:, i], 0.975),
                "n_models": len(curves),
            })
    ale = pd.DataFrame(ale_rows)
    ale.to_csv(OUT_XAI / "ale_curves.csv", index=False)

    plot_importance(summary)
    plot_ale(ale, top_features)
    plot_shap_beeswarm(pd.DataFrame(shap_rows), top_features)

    manifest = {
        "best_configuration": {
            "algorithm": key[0],
            "feature_set": key[1],
            "pseudo_absence_method": key[2],
            "pooled_oof_auc": float(best["auc_roc"]),
            "auc_ci_95": [
                float(best["auc_roc_ci_lo"]), float(best["auc_roc_ci_hi"])],
        },
        "occurrences": len(occurrences),
        "fold_models": len(models),
        "permutation_repeats": N_PERM,
        "bootstrap_repeats": N_BOOT,
        "interpretation": (
            "Permutation importance was computed only on held-out outer-fold data. "
            "SHAP and ALE summarize fitted associations and do not establish causality."
        ),
    }
    (OUT_XAI / "explainability_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved explainability outputs to {OUT_XAI}")


def plot_importance(summary):
    shown = summary.head(12).sort_values(
        "permutation_auc_decrease_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    means = shown["permutation_auc_decrease_mean"].to_numpy()
    lo = means - shown["permutation_ci_lo"].to_numpy()
    hi = shown["permutation_ci_hi"].to_numpy() - means
    colors = shown["group"].map({
        "S: environmental suitability": "#2166ac",
        "A: waterway accessibility": "#4dac26",
        "P: human-pressure proxy": "#d6604d",
    })
    ax.barh(shown["label"], means, color=colors)
    ax.errorbar(means, shown["label"], xerr=[lo, hi], fmt="none", color="black")
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Held-out AUC decrease after permutation")
    ax.set_title("Factor importance in the best validated configuration")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"Fig_factor_permutation_importance.{suffix}", dpi=300)
    plt.close(fig)


def plot_ale(ale, top_features):
    available = [x for x in top_features if x in set(ale["feature"])]
    if not available:
        return
    ncols = 2
    nrows = int(np.ceil(len(available) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, available):
        sub = ale[ale["feature"] == name].sort_values("x")
        ax.plot(sub["x"], sub["ale_mean"], color="#2b6cb0", linewidth=2)
        ax.fill_between(
            sub["x"], sub["ale_ci_lo"], sub["ale_ci_hi"],
            color="#90cdf4", alpha=0.45)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(LABELS[name])
        ax.set_ylabel("Accumulated local effect")
    for ax in axes[len(available):]:
        ax.axis("off")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"Fig_ALE_top_factors.{suffix}", dpi=300)
    plt.close(fig)


def plot_shap_beeswarm(shap_long, top_features):
    if shap_long.empty:
        return
    shown = [x for x in top_features if x in set(shap_long["feature"])]
    fig, axes = plt.subplots(len(shown), 1, figsize=(8, max(4, len(shown) * 1.25)))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, shown):
        sub = shap_long[shap_long["feature"] == name]
        values = sub["value"].to_numpy()
        scaled = (values - np.nanmin(values)) / (
            np.nanmax(values) - np.nanmin(values) + 1e-12)
        jitter = np.random.default_rng(42).normal(0, 0.08, len(sub))
        ax.scatter(
            sub["shap_value"], jitter, c=scaled, cmap="coolwarm",
            s=8, alpha=0.45, edgecolors="none")
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_yticks([])
        ax.set_ylabel(LABELS[name], rotation=0, ha="right", va="center")
    axes[-1].set_xlabel("SHAP contribution to predicted suitability")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"Fig_SHAP_top_factors.{suffix}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
