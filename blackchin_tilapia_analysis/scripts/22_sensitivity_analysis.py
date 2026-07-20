"""
Step 22 — Sensitivity Analysis (output5)

Tests how results change when varying:
  1. Thinning distance: 5, 10, 20 km
  2. Water mask threshold: 50%, 75% occurrence frequency
  3. Occurrence subset: GBIF/iNat only vs full dataset (with field reports)
  4. Spatial partition: k=2, 3, 4 folds

For each sensitivity factor, run the best model configuration from Step 20
(algo=RF, feat_set=M_SAP, pa_method=random, pa_ratio=1:5) across 3 repeats.

Outputs (output4/sensitivity/):
  sensitivity_thinning.csv
  sensitivity_watermask.csv
  sensitivity_occsubset.csv
  sensitivity_kfolds.csv
  sensitivity_summary.csv
"""

import sys, os, warnings
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from sklearn.linear_model import LogisticRegression

from blackchin_tilapia_analysis.scripts.s00_config import (
    DATA, ENV_DIR, A_DIR, P_DIR, OUT3, OUT5, OCC_CSV as OCC_SOURCE,
    RANDOM_STATE, PA_BUFFER_KM,
)

OUT_SENS = OUT5 / "sensitivity"
OUT_SENS.mkdir(parents=True, exist_ok=True)

OCC_CSV   = OCC_SOURCE
OCC_RAW   = OCC_SOURCE
WATER_TIF = DATA / "water" / "water_mask_envgrid.tif"
JRC_TIF   = DATA / "water" / "jrc_occurrence_envgrid.tif"

S_FEATURES = ["bio01","bio04","bio07","bio10","bio11",
               "bio12","bio15","bio16","bio17","elevation"]
A_FEATURES = ["A1_dist_waterway","A2_waterway_order"]
P_FEATURES = ["P1_pop_density","P2_dist_road","P3_road_density","P4_dist_urban"]
ALL_FEATS  = S_FEATURES + A_FEATURES + P_FEATURES

FEAT_PATHS = {
    **{f: Path(ENV_DIR) / f"{f}.tif" for f in S_FEATURES},
    **{f: Path(A_DIR)   / f"{f}.tif" for f in A_FEATURES},
    **{f: Path(P_DIR)   / f"{f}.tif" for f in P_FEATURES},
}

N_REPS = 3    # reduced reps for sensitivity (speed)
N_FOLDS = 3
N_PA = 200


# ── Shared utilities ──────────────────────────────────────────────────────────

def load_arrays():
    arrays = {}; transform = crs = None
    for name, path in FEAT_PATHS.items():
        if not path.exists():
            continue
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float64)
            nd  = src.nodata
            if nd is not None:
                arr[np.abs(arr - nd) < max(abs(nd)*1e-4, 1e-3)] = np.nan
            arr[arr < -1e30] = np.nan
            arrays[name] = arr
            if transform is None:
                transform = src.transform; crs = src.crs

    if "P1_pop_density" in arrays and transform is not None:
        p1 = arrays["P1_pop_density"]
        rows = np.arange(p1.shape[0])
        lats = transform.f + (rows + 0.5) * transform.e
        dy = abs(transform.e) * np.pi / 180 * 6371
        dx = abs(transform.a) * np.pi / 180 * 6371 * np.abs(np.cos(np.radians(lats)))
        area_km2 = dy * dx[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            p1 = np.where(area_km2 > 0, p1 / area_km2, np.nan)
        if WATER_TIF.exists():
            with rasterio.open(WATER_TIF) as wm:
                water = wm.read(1) == 1
            p1 = np.where(water & ~np.isfinite(p1), 0.0, p1)
        arrays["P1_pop_density"] = p1
    return arrays, transform, crs


def load_occurrences():
    df = pd.read_csv(OCC_CSV, encoding="utf-8-sig")
    if "year" in df.columns:
        year = pd.to_numeric(df["year"], errors="coerce")
        df = df[year.between(1980, 2024) | year.isna()].copy()
    return df.reset_index(drop=True)


def get_valid_cells(arrays, water_mask_arr):
    H, W = next(iter(arrays.values())).shape
    valid = (water_mask_arr == 1)
    for arr in arrays.values():
        valid &= np.isfinite(arr)
    rows, cols = np.where(valid)
    transform = None
    # get transform from first available tif
    for p in FEAT_PATHS.values():
        if p.exists():
            with rasterio.open(p) as src:
                transform = src.transform
            break
    lons = transform.c + (cols + 0.5) * transform.a
    lats = transform.f + (rows + 0.5) * transform.e
    return lons, lats, transform


def sample_features(arrays, feature_names, lons, lats, transform):
    H, W = next(iter(arrays.values())).shape
    X = np.full((len(lons), len(feature_names)), np.nan)
    for j, name in enumerate(feature_names):
        arr = arrays[name]
        for i, (lo, la) in enumerate(zip(lons, lats)):
            c = int((lo - transform.c) / transform.a)
            r = int((la - transform.f) / transform.e)
            if 0 <= r < H and 0 <= c < W:
                X[i, j] = arr[r, c]
    return X


def spatial_thin(lons, lats, min_km, seed):
    np.random.seed(seed)
    n   = len(lons)
    idx = np.random.permutation(n)
    kept = [idx[0]]; klo = [lons[idx[0]]]; kla = [lats[idx[0]]]
    for i in idx[1:]:
        dlo = np.radians(lons[i]  - np.array(klo))
        dla = np.radians(lats[i]  - np.array(kla))
        a   = (np.sin(dla/2)**2 + np.cos(np.radians(lats[i]))
               * np.cos(np.radians(kla)) * np.sin(dlo/2)**2)
        d   = 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        if d.min() >= min_km:
            kept.append(i); klo.append(lons[i]); kla.append(lats[i])
    return np.array(kept)


def balanced_spatial_blocks(lons, lats, k, seed):
    """Balanced geographic projection blocks shared by presences and cells."""
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    lon_scale = max(float(np.cos(np.radians(np.nanmean(lats)))), 0.2)
    center = np.array([np.nanmean(lons), np.nanmean(lats)], dtype=float)
    xy = np.column_stack([(lons - center[0]) * lon_scale, lats - center[1]])
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0.0, np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)])
    scores = xy @ direction
    order = np.argsort(scores, kind="mergesort")
    groups = np.array_split(order, k)
    labels = np.empty(len(lons), dtype=int)
    for fold, idx in enumerate(groups):
        labels[idx] = fold
    sorted_scores = scores[order]
    cumulative = np.cumsum([len(idx) for idx in groups])[:-1]
    cuts = np.array([(sorted_scores[p-1] + sorted_scores[p]) / 2 for p in cumulative])
    return labels, (center, lon_scale, direction, cuts)


def assign_block_cells(lons, lats, partition):
    center, lon_scale, direction, cuts = partition
    xy = np.column_stack([
        (np.asarray(lons) - center[0]) * lon_scale,
        np.asarray(lats) - center[1],
    ])
    return np.digitize(xy @ direction, cuts).astype(int)


def hav_min_dist(cell_lons, cell_lats, pres_lons, pres_lats):
    min_d = np.full(len(cell_lons), np.inf)
    for plo, pla in zip(pres_lons, pres_lats):
        dlo = np.radians(cell_lons - plo); dla = np.radians(cell_lats - pla)
        a   = (np.sin(dla/2)**2 + np.cos(np.radians(pla))
               * np.cos(np.radians(cell_lats)) * np.sin(dlo/2)**2)
        d   = 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        min_d = np.minimum(min_d, d)
    return min_d


def sample_random_bg(cell_lons, cell_lats, pres_lons, pres_lats, n, seed):
    np.random.seed(seed)
    d    = hav_min_dist(cell_lons, cell_lats, pres_lons, pres_lats)
    cand = np.where(d >= PA_BUFFER_KM)[0]
    if len(cand) == 0: cand = np.arange(len(cell_lons))
    chosen = np.random.choice(cand, size=min(n, len(cand)), replace=False)
    return cell_lons[chosen], cell_lats[chosen]


def run_cv(occ, arrays, cell_lons, cell_lats, transform, feat_names,
           n_folds=3, n_reps=N_REPS, seed=RANDOM_STATE):
    """Run leakage-free nested spatial CV and return per-fold AUC + PR-AUC."""
    rows = []
    for rep in range(n_reps):
        rep_seed = seed + rep * 100
        k = n_folds
        while k >= 2 and len(occ) // k < 2:
            k -= 1
        if k < 2:
            continue
        lbl, partition = balanced_spatial_blocks(
            occ["longitude"].values, occ["latitude"].values, k, rep_seed)
        occ["_fold"] = lbl + 1
        cell_block = assign_block_cells(cell_lons, cell_lats, partition)

        for fold in range(1, k + 1):
            tm = (occ["_fold"] != fold); vm = ~tm
            ptr_lo = occ[tm]["longitude"].values; ptr_la = occ[tm]["latitude"].values
            pte_lo = occ[vm]["longitude"].values; pte_la = occ[vm]["latitude"].values
            if len(pte_lo) == 0:
                continue
            tr_cells = (cell_block != (fold - 1))
            te_cells = (cell_block == (fold - 1))
            clo_tr = cell_lons[tr_cells]; cla_tr = cell_lats[tr_cells]
            clo_te = cell_lons[te_cells]; cla_te = cell_lats[te_cells]
            if len(clo_te) < 5:
                continue

            X_ptr = sample_features(arrays, feat_names, ptr_lo, ptr_la, transform)
            X_pte = sample_features(arrays, feat_names, pte_lo, pte_la, transform)
            btr_lo, btr_la = sample_random_bg(clo_tr, cla_tr, ptr_lo, ptr_la, N_PA, rep_seed + fold)
            bte_lo, bte_la = sample_random_bg(clo_te, cla_te, pte_lo, pte_la, N_PA, rep_seed + fold + 100)
            X_btr = sample_features(arrays, feat_names, btr_lo, btr_la, transform)
            X_bte = sample_features(arrays, feat_names, bte_lo, bte_la, transform)

            X_tr = np.vstack([X_ptr, X_btr]); y_tr = np.concatenate([np.ones(len(X_ptr)), np.zeros(len(X_btr))])
            X_te = np.vstack([X_pte, X_bte]); y_te = np.concatenate([np.ones(len(X_pte)), np.zeros(len(X_bte))])

            ok_tr = np.isfinite(X_tr).all(axis=1); ok_te = np.isfinite(X_te).all(axis=1)
            X_tr = X_tr[ok_tr]; y_tr = y_tr[ok_tr]
            X_te = X_te[ok_te]; y_te = y_te[ok_te]

            if y_te.sum() == 0 or (y_te == 0).sum() == 0 or X_tr.shape[0] < 5:
                continue

            rf   = RandomForestClassifier(n_estimators=200, random_state=rep_seed, n_jobs=1)
            rf.fit(X_tr, y_tr)
            p_te = rf.predict_proba(X_te)[:, 1]
            rows.append({
                "rep": rep+1, "fold": fold,
                "auc_roc": roc_auc_score(y_te, p_te),
                "pr_auc":  average_precision_score(y_te, p_te),
                "brier":   brier_score_loss(y_te, p_te),
            })
    return pd.DataFrame(rows)


def summarise(df, label, value):
    if len(df) == 0:
        return {}
    return {
        "sensitivity_factor": label,
        "value": value,
        "n_folds": len(df),
        "auc_mean": round(df["auc_roc"].mean(), 4),
        "auc_sd":   round(df["auc_roc"].std(),  4),
        "prauc_mean": round(df["pr_auc"].mean(), 4),
        "prauc_sd":   round(df["pr_auc"].std(),  4),
        "brier_mean": round(df["brier"].mean(), 4),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Step 22: Sensitivity Analysis")
    print("=" * 65)

    arrays, transform, crs = load_arrays()
    feat_names = [f for f in ALL_FEATS if f in arrays]

    # Default water mask
    if WATER_TIF.exists():
        with rasterio.open(WATER_TIF) as wm:
            wm_arr = wm.read(1)
    else:
        H, W = next(iter(arrays.values())).shape
        wm_arr = np.ones((H, W), dtype=np.uint8)

    all_summary = []

    # ── 1. Thinning distance ──────────────────────────────────────────────────
    print("\n[1] Sensitivity to thinning distance (5, 10, 20 km)...")
    raw_occ = load_occurrences()
    thin_rows = []
    for thin_km in [5, 10, 20]:
        lons = raw_occ["longitude"].values; lats = raw_occ["latitude"].values
        # Use seed 0 for reproducibility
        counts = [len(spatial_thin(lons, lats, thin_km, s)) for s in range(5)]
        med_n  = float(np.median(counts))
        best_s = min(range(5), key=lambda s: abs(counts[s] - med_n))
        idx    = spatial_thin(lons, lats, thin_km, best_s)
        occ_t  = raw_occ.iloc[idx].copy().reset_index(drop=True)
        if len(occ_t) < 6:
            print(f"  thin={thin_km}km: only {len(occ_t)} occ, skip")
            continue
        print(f"  thin={thin_km}km: {len(occ_t)} occurrences")
        cell_lons, cell_lats, tr = get_valid_cells(arrays, wm_arr)
        df_cv = run_cv(occ_t, arrays, cell_lons, cell_lats, tr, feat_names)
        thin_rows.append(summarise(df_cv, "thinning_km", thin_km))

    df_thin = pd.DataFrame([r for r in thin_rows if r])
    df_thin.to_csv(OUT_SENS / "sensitivity_thinning.csv", index=False)
    all_summary.extend(thin_rows)
    print(df_thin[["value","auc_mean","auc_sd","prauc_mean"]].to_string(index=False))

    # ── 2. Water mask threshold ───────────────────────────────────────────────
    print("\n[2] Sensitivity to water mask threshold (50%, 75% JRC occurrence)...")
    jrc_rows = []
    if JRC_TIF.exists():
        with rasterio.open(JRC_TIF) as jrc:
            jrc_arr = jrc.read(1).astype(np.float32)
        for thr in [50, 75]:
            wm_thr = (jrc_arr >= thr).astype(np.uint8)
            cell_lons_t, cell_lats_t, tr_t = get_valid_cells(arrays, wm_thr)
            print(f"  JRC threshold={thr}%: {len(cell_lons_t)} valid cells")
            occ_base = load_occurrences()
            if len(occ_base) < 6: continue
            df_cv = run_cv(occ_base, arrays, cell_lons_t, cell_lats_t, tr_t, feat_names)
            jrc_rows.append(summarise(df_cv, "jrc_threshold_pct", thr))
    else:
        # Fallback: use proportion of water cells
        for pct in [50, 75]:
            cell_lons_t, cell_lats_t, tr_t = get_valid_cells(arrays, wm_arr)
            jrc_rows.append(summarise(pd.DataFrame(), "jrc_threshold_pct (JRC not found)", pct))

    df_jrc = pd.DataFrame([r for r in jrc_rows if r])
    df_jrc.to_csv(OUT_SENS / "sensitivity_watermask.csv", index=False)
    all_summary.extend(jrc_rows)
    if len(df_jrc) > 0:
        print(df_jrc[["value","auc_mean","auc_sd","prauc_mean"]].to_string(index=False))

    # ── 3. Occurrence subset: GBIF/iNat only vs full ──────────────────────────
    print("\n[3] Sensitivity to occurrence subset (GBIF/iNat vs full)...")
    subset_rows = []
    if OCC_RAW.exists():
        raw_full = load_occurrences()
    else:
        raw_full = load_occurrences()

    cell_lons_s, cell_lats_s, tr_s = get_valid_cells(arrays, wm_arr)

    for subset_name, subset_filter in [
        ("full", None),
        ("gbif_inat_only", ["gbif","inat","iNaturalist","GBIF"]),
    ]:
        if subset_filter is not None and "source" in raw_full.columns:
            sub = raw_full[raw_full["source"].str.lower().isin(
                [s.lower() for s in subset_filter]
            )].copy()
        else:
            sub = raw_full.copy()

        if len(sub) < 6:
            print(f"  {subset_name}: only {len(sub)} occ, skip")
            subset_rows.append({"sensitivity_factor": "occ_subset", "value": subset_name,
                                 "n_occ": len(sub), "auc_mean": np.nan})
            continue

        # Apply same QC as step 01 (minimal: coords, bbox)
        sub = sub.dropna(subset=["latitude","longitude"])
        sub = sub[(sub["latitude"].between(-90,90)) & (sub["longitude"].between(-180,180))]
        sub = sub[(sub["longitude"].between(97.3,105.7)) & (sub["latitude"].between(5.5,20.6))]
        yr  = pd.to_numeric(sub.get("year", pd.Series(2020, index=sub.index)), errors="coerce")
        sub = sub[yr.between(1980,2024) | yr.isna()].reset_index(drop=True)

        if len(sub) < 6:
            print(f"  {subset_name}: {len(sub)} occ after QC, skip")
            continue

        # Thin at 10 km
        lons_s = sub["longitude"].values; lats_s = sub["latitude"].values
        counts_s = [len(spatial_thin(lons_s, lats_s, 10.0, s)) for s in range(5)]
        med_s = float(np.median(counts_s))
        best_s2 = min(range(5), key=lambda s: abs(counts_s[s] - med_s))
        idx_s = spatial_thin(lons_s, lats_s, 10.0, best_s2)
        occ_s = sub.iloc[idx_s].copy().reset_index(drop=True)
        print(f"  {subset_name}: {len(occ_s)} occurrences")

        df_cv = run_cv(occ_s, arrays, cell_lons_s, cell_lats_s, tr_s, feat_names)
        r = summarise(df_cv, "occ_subset", subset_name)
        r["n_occ"] = len(occ_s)
        subset_rows.append(r)

    df_sub = pd.DataFrame([r for r in subset_rows if r])
    df_sub.to_csv(OUT_SENS / "sensitivity_occsubset.csv", index=False)
    all_summary.extend(subset_rows)
    if len(df_sub) > 0 and "auc_mean" in df_sub.columns:
        print(df_sub[["value","n_occ" if "n_occ" in df_sub else "value",
                       "auc_mean","auc_sd","prauc_mean"]].to_string(index=False))

    # ── 4. Number of spatial folds (k=2, 3, 4) ───────────────────────────────
    print("\n[4] Sensitivity to number of spatial folds (k=2, 3, 4)...")
    kfold_rows = []
    occ_base = load_occurrences()
    cell_lons_k, cell_lats_k, tr_k = get_valid_cells(arrays, wm_arr)
    for k in [2, 3, 4]:
        print(f"  k={k} folds")
        df_cv = run_cv(occ_base, arrays, cell_lons_k, cell_lats_k, tr_k, feat_names,
                       n_folds=k, n_reps=N_REPS)
        kfold_rows.append(summarise(df_cv, "n_folds", k))

    df_kf = pd.DataFrame([r for r in kfold_rows if r])
    df_kf.to_csv(OUT_SENS / "sensitivity_kfolds.csv", index=False)
    all_summary.extend(kfold_rows)
    print(df_kf[["value","n_folds","auc_mean","auc_sd","prauc_mean"]].to_string(index=False))

    # Combined summary
    df_all = pd.DataFrame([r for r in all_summary if r and isinstance(r, dict)])
    df_all.to_csv(OUT_SENS / "sensitivity_summary.csv", index=False)
    print(f"\n  Combined summary: {len(df_all)} rows -> output4/sensitivity/sensitivity_summary.csv")

    print("\n  Key question: Do AUC values remain stable across these conditions?")
    print("  If SD_across_conditions > SD_within_CV: results are sensitivity-dependent.")


if __name__ == "__main__":
    main()
