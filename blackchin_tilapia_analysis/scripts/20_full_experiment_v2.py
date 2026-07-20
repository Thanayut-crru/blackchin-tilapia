"""
Step 20 v2 — Full factorial experiment (output5)

FIXES vs v1 (output4):
  1. LEAKAGE-FREE EVAL BACKGROUND: pre-generated once per (rep, fold) from
     test block cells using pure random sampling. Test presence locations
     NEVER used for background selection. Same background shared by ALL 48
     (algo, feat, pa_method) combos.
  2. BALANCED SPATIAL FOLDS: repeated geographic projection blocks differ
     in size by at most one occurrence. The same projection boundaries are
     used for presence and background cells.
  3. TSS THRESHOLD FROM INNER OOF: Youden J derived from pooled inner-fold
     out-of-fold predictions (not from the test set). Prevents test-set
     optimism in threshold selection.
  4. PA RATIO RELATIVE TO n_pres_tr: PA_RATIO_MULT gives multipliers (1,3,5,10)
     applied to actual n_training_presences per fold — not fixed absolute counts.
  5. INNER SPATIAL CV: inner folds assigned by spatial k-means on training
     presences (same approach as outer). Falls back to random if <3 pres.
  6. MXN USES SELECTED PA BACKGROUND: MXN no longer overrides the PA background
     with a random 5000-pt pool. All algorithms see the same training data.
  7. YEAR FILTER ENFORCED AT LOAD TIME: occ filtered to year<=2024 in main(),
     independent of whether 01_occurrence_prep.py was run correctly.
  8. occ_id COLUMN in OOF for clustered bootstrap.
  9. ATTRITION TRACKING: n_pres_dropped logged per fold.
 10. NAMING: uncertainty_zone (was shap_fuzzy).

Output: output5/
"""

import sys, os, json, time, warnings
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import StandardScaler

from blackchin_tilapia_analysis.scripts.s00_config import (
    DATA, ENV_DIR, A_DIR, P_DIR, OUT3, OUT5, OCC_CSV,
    N_OUTER_FOLDS, N_OUTER_REPS, N_INNER_FOLDS, N_HP_CANDIDATES,
    PA_BUFFER_KM, RANDOM_STATE,
)

OUT5.mkdir(exist_ok=True)
OCC_FINAL  = OCC_CSV
WATER_MASK = DATA / "water" / "water_mask_envgrid.tif"

S_FEATURES = ["bio01","bio04","bio07","bio10","bio11",
               "bio12","bio15","bio16","bio17","elevation"]
# A3_dist_coast EXCLUDED: source raster has p50=0, indicating it measures distance
# to nearest water body (not sea coast) — redundant with A1 and likely mis-specified.
A_FEATURES = ["A1_dist_waterway","A2_waterway_order"]
P_FEATURES = ["P1_pop_density","P2_dist_road","P3_road_density","P4_dist_urban"]

MODEL_VARIANTS = {
    "M_S":   S_FEATURES,
    "M_SA":  S_FEATURES + A_FEATURES,
    "M_SP":  S_FEATURES + P_FEATURES,
    "M_SAP": S_FEATURES + A_FEATURES + P_FEATURES,
}

FEAT_PATHS = {
    **{f: Path(ENV_DIR) / f"{f}.tif" for f in S_FEATURES},
    **{f: Path(A_DIR)   / f"{f}.tif" for f in A_FEATURES},
    **{f: Path(P_DIR)   / f"{f}.tif" for f in P_FEATURES},
}

# PA ratio multipliers relative to actual n_training_presences
PA_RATIO_MULT = {"1:1": 1, "1:3": 3, "1:5": 5, "1:10": 10}

# Fixed evaluation background per (rep, fold) — not tuned, not PA-method-dependent
N_EVAL_BG = 400

RF_HP_SPACE = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [None, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features":     ["sqrt", 0.5, 0.7],
    "_pa_ratio":        list(PA_RATIO_MULT.keys()),
}

XGB_HP_SPACE = {
    "n_estimators":    [100, 200, 300],
    "max_depth":       [3, 5, 7],
    "learning_rate":   [0.05, 0.1, 0.2],
    "subsample":       [0.7, 0.9, 1.0],
    "colsample_bytree":[0.6, 0.8, 1.0],
    "min_child_weight":[1, 5, 10],
    "_pa_ratio":       list(PA_RATIO_MULT.keys()),
}

MXN_HP_SPACE = {
    "C":        [0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
    "_pa_ratio": list(PA_RATIO_MULT.keys()),   # also tune PA ratio for MXN
}

PA_METHODS = ["random", "spatial_constrained", "two_step", "uncertainty_zone"]


def _check_xgb():
    try:
        import xgboost
        return True
    except ImportError:
        return False

XGB_AVAILABLE = _check_xgb()


# ── Raster loading ─────────────────────────────────────────────────────────────

def _cell_area_km2(transform, shape):
    """Approximate cell area in km² for each row of a geographic raster."""
    H, W = shape
    dlat_deg = abs(transform.e)
    dlon_deg = abs(transform.a)
    rows = np.arange(H)
    lats = transform.f + (rows + 0.5) * transform.e
    area = (dlon_deg * np.pi/180 * 6371) * (dlat_deg * np.pi/180 * 6371) * np.abs(np.cos(np.radians(lats)))
    return area[:, None] * np.ones((1, W))


def load_all_rasters():
    arrays = {}
    ref_transform = ref_crs = None
    for name, path in FEAT_PATHS.items():
        if not path.exists():
            print(f"  WARNING: {name} not found at {path}")
            continue
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float64)
            nd  = src.nodata
            if nd is not None:
                arr[np.abs(arr - nd) < max(abs(nd)*1e-4, 1e-3)] = np.nan
            arr[arr < -1e30] = np.nan
            arrays[name] = arr
            if ref_transform is None:
                ref_transform = src.transform
                ref_crs       = src.crs

    # P1 post-processing:
    #   1. The raster is sum-resampled from WorldPop 100m → population COUNT per cell.
    #      Convert to density (people/km²) by dividing by cell area.
    #   2. Zero-fill NaN where the water mask has valid cells but WorldPop has no data
    #      (water bodies with no permanent population → 0 is correct).
    if "P1_pop_density" in arrays and ref_transform is not None:
        p1 = arrays["P1_pop_density"]
        area_km2 = _cell_area_km2(ref_transform, p1.shape)
        # Convert count → density (guard against zero area)
        with np.errstate(divide="ignore", invalid="ignore"):
            p1_density = np.where(area_km2 > 0, p1 / area_km2, np.nan)
        # Zero-fill NaN where water mask is 1 (valid study cell, just no WorldPop coverage)
        if WATER_MASK.exists():
            with rasterio.open(WATER_MASK) as wm:
                wmask = (wm.read(1) == 1)
            p1_density = np.where(wmask & ~np.isfinite(p1_density), 0.0, p1_density)
        arrays["P1_pop_density"] = p1_density
        n_fill = np.isfinite(p1_density).sum()
        print(f"  P1 post-proc: converted count to density; valid cells after zero-fill = {n_fill}")

    return arrays, ref_transform, ref_crs


def get_common_valid_cells(arrays, transform):
    H, W = next(iter(arrays.values())).shape
    if WATER_MASK.exists():
        with rasterio.open(WATER_MASK) as wm:
            wmask = (wm.read(1) == 1)
    else:
        wmask = np.ones((H, W), dtype=bool)
    valid = wmask.copy()
    for arr in arrays.values():
        valid &= np.isfinite(arr)
    rows, cols = np.where(valid)
    lons = transform.c + (cols + 0.5) * transform.a
    lats = transform.f + (rows + 0.5) * transform.e
    return lons, lats


def sample_at_coords(arrays, feature_names, lons, lats, transform):
    n_pts = len(lons)
    X = np.full((n_pts, len(feature_names)), np.nan)
    H, W = next(iter(arrays.values())).shape
    for j, name in enumerate(feature_names):
        arr = arrays[name]
        for i, (lo, la) in enumerate(zip(lons, lats)):
            col = int((lo - transform.c) / transform.a)
            row = int((la - transform.f) / transform.e)
            if 0 <= row < H and 0 <= col < W:
                X[i, j] = arr[row, col]
    return X


def _sample_cell_X(lons, lats, all_lons, all_lats, all_X):
    all_c = np.column_stack([all_lons, all_lats])
    pts   = np.column_stack([lons, lats])
    X_out = np.full((len(pts), all_X.shape[1]), np.nan)
    for i, pt in enumerate(pts):
        X_out[i] = all_X[np.argmin(np.sum((all_c - pt)**2, axis=1))]
    return X_out


# ── Spatial partitioning ──────────────────────────────────────────────────────

def _balanced_projection_folds(lons, lats, k, seed):
    """Create reproducible, spatially contiguous folds with sizes differing by <=1."""
    n = len(lons)
    if k < 2 or n < k:
        raise ValueError(f"Cannot split n={n} occurrences into k={k} folds")

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

    labels = np.empty(n, dtype=int)
    for fold, idx in enumerate(groups):
        labels[idx] = fold

    sorted_scores = scores[order]
    cumulative = np.cumsum([len(idx) for idx in groups])[:-1]
    cuts = np.array([
        (sorted_scores[pos - 1] + sorted_scores[pos]) / 2.0
        for pos in cumulative
    ])
    partition = {
        "center": center,
        "lon_scale": lon_scale,
        "direction": direction,
        "cuts": cuts,
    }
    return labels, partition


def assign_spatial_folds_stable(df, k=N_OUTER_FOLDS, base_seed=0, max_tries=20):
    """Balanced repeated geographic blocks; retains legacy return signature."""
    del max_tries
    labels, partition = _balanced_projection_folds(
        df["longitude"].values, df["latitude"].values, k, base_seed)
    df_out = df.copy()
    df_out["spatial_fold"] = labels + 1
    return df_out, k, partition


def assign_inner_spatial_folds(lons, lats, k, seed):
    """
    Balanced geographic fold assignment for inner CV.
    Returns (fold_labels, partition, fold_type).
    Projection boundaries are also used for inner background cells.
    Falls back to random (with None centroids) if too few points.
    """
    n = len(lons)
    if n < k * 2:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        folds = np.zeros(n, dtype=int)
        for ki in range(k):
            folds[idx[ki::k]] = ki
        return folds, None, "random_fallback"
    lbl, partition = _balanced_projection_folds(lons, lats, k, seed)
    return lbl, partition, "balanced_spatial"


def assign_cells_to_blocks(lons, lats, partition):
    xy = np.column_stack([
        (np.asarray(lons) - partition["center"][0]) * partition["lon_scale"],
        np.asarray(lats) - partition["center"][1],
    ])
    scores = xy @ partition["direction"]
    return np.digitize(scores, partition["cuts"]).astype(int)


# ── Distance ──────────────────────────────────────────────────────────────────

def hav_dist_to_nearest(cell_lons, cell_lats, pres_lons, pres_lats):
    min_d = np.full(len(cell_lons), np.inf)
    for plo, pla in zip(pres_lons, pres_lats):
        dlo = np.radians(cell_lons - plo)
        dla = np.radians(cell_lats - pla)
        a   = (np.sin(dla/2)**2
               + np.cos(np.radians(pla)) * np.cos(np.radians(cell_lats))
               * np.sin(dlo/2)**2)
        d   = 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        min_d = np.minimum(min_d, d)
    return min_d


# ── PA sampling (TRAINING only) ───────────────────────────────────────────────

def sample_at_coords_direct(pres_lons, pres_lats, all_lons, all_lats, all_X):
    pts       = np.column_stack([pres_lons, pres_lats])
    all_coords = np.column_stack([all_lons, all_lats])
    X = np.full((len(pts), all_X.shape[1]), np.nan)
    for i, pt in enumerate(pts):
        X[i] = all_X[np.argmin(np.sum((all_coords - pt)**2, axis=1))]
    return X


def sample_random_pa(cell_lons, cell_lats, pres_lons, pres_lats,
                     n, buffer_km=PA_BUFFER_KM, seed=0):
    np.random.seed(seed)
    d    = hav_dist_to_nearest(cell_lons, cell_lats, pres_lons, pres_lats)
    cand = np.where(d >= buffer_km)[0]
    if len(cand) == 0:
        cand = np.arange(len(cell_lons))
    chosen = np.random.choice(cand, size=min(n, len(cand)), replace=False)
    return cell_lons[chosen], cell_lats[chosen]


def sample_spatial_constrained_pa(cell_lons, cell_lats, pres_lons, pres_lats,
                                   n, buffer_km=PA_BUFFER_KM, max_range_km=200.0, seed=0):
    np.random.seed(seed)
    d    = hav_dist_to_nearest(cell_lons, cell_lats, pres_lons, pres_lats)
    cand = np.where((d >= buffer_km) & (d <= max_range_km))[0]
    if len(cand) < n:
        cand = np.where(d >= buffer_km)[0]
    if len(cand) == 0:
        cand = np.arange(len(cell_lons))
    chosen = np.random.choice(cand, size=min(n, len(cand)), replace=False)
    return cell_lons[chosen], cell_lats[chosen]


def sample_two_step_pa(cell_lons, cell_lats, pres_lons, pres_lats,
                        cell_X, n, buffer_km=PA_BUFFER_KM, seed=0):
    np.random.seed(seed)
    d       = hav_dist_to_nearest(cell_lons, cell_lats, pres_lons, pres_lats)
    outside = np.where(d >= buffer_km)[0]
    if len(outside) == 0:
        outside = np.arange(len(cell_lons))
    pres_X  = sample_at_coords_direct(pres_lons, pres_lats, cell_lons, cell_lats, cell_X)
    pres_ok = np.isfinite(pres_X).all(axis=1)
    bg_idx  = np.random.choice(outside, size=min(200, len(outside)), replace=False)
    bg_X    = cell_X[bg_idx]; bg_ok = np.isfinite(bg_X).all(axis=1)
    X_tr = np.vstack([pres_X[pres_ok], bg_X[bg_ok]])
    y_tr = np.concatenate([np.ones(pres_ok.sum()), np.zeros(bg_ok.sum())])
    if y_tr.sum() == 0 or (y_tr == 0).sum() == 0 or len(X_tr) < 5:
        return sample_random_pa(cell_lons, cell_lats, pres_lons, pres_lats, n, seed=seed)
    rf = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=1)
    rf.fit(X_tr, y_tr)
    out_X = cell_X[outside]; ok = np.isfinite(out_X).all(axis=1)
    sp    = np.full(len(outside), 0.5)
    if ok.sum() > 0:
        sp[ok] = rf.predict_proba(out_X[ok])[:, 1]
    low = outside[sp <= np.percentile(sp, 33)]
    if len(low) < n:
        low = outside
    chosen = np.random.choice(low, size=min(n, len(low)), replace=False)
    return cell_lons[chosen], cell_lats[chosen]


def sample_uncertainty_zone_pa(cell_lons, cell_lats, pres_lons, pres_lats,
                                cell_X, n, buffer_km=PA_BUFFER_KM,
                                fuzzy_lo=0.2, fuzzy_hi=0.6, seed=0):
    """Background from pilot-RF uncertainty zone [lo,hi]. Renamed from shap_fuzzy."""
    np.random.seed(seed)
    d       = hav_dist_to_nearest(cell_lons, cell_lats, pres_lons, pres_lats)
    outside = np.where(d >= buffer_km)[0]
    if len(outside) == 0:
        outside = np.arange(len(cell_lons))
    pres_X  = sample_at_coords_direct(pres_lons, pres_lats, cell_lons, cell_lats, cell_X)
    pres_ok = np.isfinite(pres_X).all(axis=1)
    bg_idx  = np.random.choice(outside, size=min(200, len(outside)), replace=False)
    bg_X    = cell_X[bg_idx]; bg_ok = np.isfinite(bg_X).all(axis=1)
    X_tr    = np.vstack([pres_X[pres_ok], bg_X[bg_ok]])
    y_tr    = np.concatenate([np.ones(pres_ok.sum()), np.zeros(bg_ok.sum())])
    if y_tr.sum() == 0 or len(X_tr) < 5:
        return sample_random_pa(cell_lons, cell_lats, pres_lons, pres_lats, n, seed=seed)
    rf = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=1)
    rf.fit(X_tr, y_tr)
    out_X = cell_X[outside]; ok = np.isfinite(out_X).all(axis=1)
    sp    = np.full(len(outside), 0.4)
    if ok.sum() > 0:
        sp[ok] = rf.predict_proba(out_X[ok])[:, 1]
    fuzzy = outside[(sp >= fuzzy_lo) & (sp <= fuzzy_hi)]
    if len(fuzzy) < n // 2:
        fuzzy = outside
    chosen = np.random.choice(fuzzy, size=min(n, len(fuzzy)), replace=False)
    return cell_lons[chosen], cell_lats[chosen]


def get_training_pa(method, cell_lons, cell_lats, pres_lons, pres_lats,
                    cell_X, n, seed):
    if method == "random":
        return sample_random_pa(cell_lons, cell_lats, pres_lons, pres_lats, n, seed=seed)
    elif method == "spatial_constrained":
        return sample_spatial_constrained_pa(cell_lons, cell_lats, pres_lons, pres_lats, n, seed=seed)
    elif method == "two_step":
        return sample_two_step_pa(cell_lons, cell_lats, pres_lons, pres_lats, cell_X, n, seed=seed)
    elif method == "uncertainty_zone":
        return sample_uncertainty_zone_pa(cell_lons, cell_lats, pres_lons, pres_lats, cell_X, n, seed=seed)
    return sample_random_pa(cell_lons, cell_lats, pres_lons, pres_lats, n, seed=seed)


# ── VIF selection ─────────────────────────────────────────────────────────────

def vif_selection(X, names, threshold=10.0):
    remaining = list(range(X.shape[1]))
    while len(remaining) > 1:
        vifs = []
        for i, idx in enumerate(remaining):
            others = [remaining[j] for j in range(len(remaining)) if j != i]
            Xo = X[:, others]; Xt = X[:, idx]
            mask = np.isfinite(Xo).all(axis=1) & np.isfinite(Xt)
            if mask.sum() < 5:
                vifs.append(0.0); continue
            r2 = LinearRegression().fit(Xo[mask], Xt[mask]).score(Xo[mask], Xt[mask])
            vifs.append(1.0 / (1.0 - r2 + 1e-8))
        if max(vifs) < threshold:
            break
        remaining.pop(int(np.argmax(vifs)))
    return remaining, [names[i] for i in remaining]


# ── Algorithm implementations ─────────────────────────────────────────────────

def _clean_rf_hp(hp):
    hp = {k: v for k, v in hp.items() if not k.startswith("_")}
    if isinstance(hp.get("max_features"), str):
        try:
            hp["max_features"] = float(hp["max_features"])
        except ValueError:
            pass
    return hp


def fit_rf(X_tr, y_tr, hp, seed):
    hp_c = _clean_rf_hp(hp)
    rf   = RandomForestClassifier(random_state=seed, n_jobs=1, **hp_c)
    rf.fit(X_tr, y_tr)
    return rf


def fit_xgb(X_tr, y_tr, hp, seed):
    from xgboost import XGBClassifier
    hp_c  = {k: v for k, v in hp.items() if not k.startswith("_")}
    ratio = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    xgb   = XGBClassifier(scale_pos_weight=ratio, eval_metric="logloss",
                           random_state=seed, n_jobs=1, verbosity=0, **hp_c)
    xgb.fit(X_tr, y_tr)
    return xgb


class MaxNetModel:
    """Elastic-net logistic regression with linear + hinge features."""
    def __init__(self, C=1.0, l1_ratio=0.5, n_hinge=10, seed=42):
        self.C=C; self.l1_ratio=l1_ratio; self.n_hinge=n_hinge; self.seed=seed
        self.scaler_=self.model_=self.thresholds_=None

    def _hinge(self, X, fit=False):
        if fit:
            self.thresholds_ = np.array([
                np.percentile(X[:, j], np.linspace(10, 90, self.n_hinge))
                for j in range(X.shape[1])])
        H = [np.maximum(0.0, X[:, j] - t)
             for j in range(X.shape[1]) for t in self.thresholds_[j]]
        return np.column_stack([X] + H) if H else X

    def fit(self, X_pres, X_bg, hp=None):
        if hp:
            self.C = hp.get("C", self.C)
            self.l1_ratio = hp.get("l1_ratio", self.l1_ratio)
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(np.vstack([X_pres, X_bg]))
        Xh = self._hinge(Xs, fit=True)
        y  = np.concatenate([np.ones(len(X_pres)), np.zeros(len(X_bg))])
        self.model_ = LogisticRegression(
            C=self.C, penalty="elasticnet", l1_ratio=self.l1_ratio,
            solver="saga", max_iter=500, random_state=self.seed,
            class_weight="balanced")
        self.model_.fit(Xh, y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self._hinge(self.scaler_.transform(X)))


def fit_maxnet(X_pres, X_bg, hp, seed):
    return MaxNetModel(seed=seed).fit(X_pres, X_bg, hp=hp)


# ── Metrics ───────────────────────────────────────────────────────────────────

def youden_threshold(y_true, y_prob):
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(thr[np.argmax(tpr - fpr)])


def calibration_slope(y_true, y_prob):
    try:
        eps = 1e-6
        lo  = np.log(np.clip(y_prob, eps, 1-eps) / (1-np.clip(y_prob, eps, 1-eps)))
        lr  = LogisticRegression(C=1e9, solver="lbfgs", max_iter=200)
        lr.fit(lo.reshape(-1,1), y_true)
        return round(float(lr.coef_[0][0]), 4), round(float(lr.intercept_[0]), 4)
    except Exception:
        return np.nan, np.nan


def compute_metrics(y_true, y_prob, threshold=None):
    """
    Compute evaluation metrics.
    threshold: if provided (from inner-OOF Youden J), used for TSS.
               if None, Youden J derived from y_true/y_prob (legacy, test-set optimism).
    """
    if y_true.sum() == 0 or (y_true == 0).sum() == 0:
        return {}
    # Use pre-computed threshold (from inner OOF) if available
    if threshold is not None and np.isfinite(threshold):
        thr = threshold
    else:
        thr = youden_threshold(y_true, y_prob)
    yp   = (y_prob >= thr).astype(int)
    tp   = int(((yp==1)&(y_true==1)).sum())
    tn   = int(((yp==0)&(y_true==0)).sum())
    fp   = int(((yp==1)&(y_true==0)).sum())
    fn   = int(((yp==0)&(y_true==1)).sum())
    sens = tp / (tp+fn+1e-9)
    spec = tn / (tn+fp+1e-9)
    cal_sl, cal_int = calibration_slope(y_true, y_prob)
    return {
        "auc_roc":    float(roc_auc_score(y_true, y_prob)),
        "pr_auc":     float(average_precision_score(y_true, y_prob)),
        "tss":        float(sens+spec-1.0),
        "tss_thr":    round(thr, 4),
        "thr_source": "inner_oof" if threshold is not None else "test_set",
        "sensitivity":round(sens, 4),
        "specificity":round(spec, 4),
        "brier":      float(brier_score_loss(y_true, y_prob)),
        "cal_slope":  cal_sl,
        "cal_int":    cal_int,
        "n_pres":     int(y_true.sum()),
        "n_bg":       int((y_true==0).sum()),
    }


# ── Inner CV: HP + PA ratio + threshold from inner OOF ───────────────────────

def _get_inner_block_cells(cell_lons_tr, cell_lats_tr, partition, k):
    """
    Return boolean mask for cells belonging to inner spatial block k.
    Uses the same projection boundaries as the inner presence folds.
    Falls back to all cells if no spatial partition is available.
    """
    if partition is None:
        return np.ones(len(cell_lons_tr), dtype=bool)
    return assign_cells_to_blocks(cell_lons_tr, cell_lats_tr, partition) == k


def _run_inner_fold(k, inner_folds, inner_centroids,
                    X_pres_tr, pres_tr_lo, pres_tr_la,
                    cell_lons_tr, cell_lats_tr, cell_X_tr,
                    feat_idx, algo, pa_method, mult, seed, hp):
    """Run one inner fold; returns (y_te, p_te) or (None, None) on failure."""
    vm = (inner_folds == k); tm = ~vm
    if tm.sum() == 0 or vm.sum() == 0:
        return None, None

    tr_lo = pres_tr_lo[tm]; tr_la = pres_tr_la[tm]
    X_p_tr = X_pres_tr[:, feat_idx][tm]
    X_p_te = X_pres_tr[:, feat_idx][vm]
    n_pa_inner = max(1, int(mult * tm.sum()))

    # Training background: from inner TRAINING block cells only
    train_cell_mask = ~_get_inner_block_cells(cell_lons_tr, cell_lats_tr, inner_centroids, k)
    if train_cell_mask.sum() < 10:
        train_cell_mask = np.ones(len(cell_lons_tr), dtype=bool)
    cell_lo_in_tr = cell_lons_tr[train_cell_mask]
    cell_la_in_tr = cell_lats_tr[train_cell_mask]
    cell_X_in_tr  = cell_X_tr[train_cell_mask]

    bg_lo, bg_la = get_training_pa(pa_method, cell_lo_in_tr, cell_la_in_tr,
                                    tr_lo, tr_la, cell_X_in_tr, n_pa_inner, seed+k)
    bg_X = _sample_cell_X(bg_lo, bg_la, cell_lo_in_tr, cell_la_in_tr, cell_X_in_tr)[:, feat_idx]
    bg_ok = np.isfinite(bg_X).all(axis=1); bg_X = bg_X[bg_ok]

    p_ok = np.isfinite(X_p_tr).all(axis=1)
    X_tr_a = np.vstack([X_p_tr[p_ok], bg_X])
    y_tr_a = np.concatenate([np.ones(p_ok.sum()), np.zeros(len(bg_X))])

    # Eval background: from inner TEST block cells only (leakage-free inner eval)
    test_cell_mask = _get_inner_block_cells(cell_lons_tr, cell_lats_tr, inner_centroids, k)
    test_cells_lo  = cell_lons_tr[test_cell_mask]
    test_cells_la  = cell_lats_tr[test_cell_mask]
    test_cells_X   = cell_X_tr[test_cell_mask]

    n_in_bg = min(n_pa_inner, len(test_cells_lo)) if len(test_cells_lo) > 0 else 0
    if n_in_bg == 0:
        return None, None
    rng_in  = np.random.default_rng(seed + k + 999)
    ib_idx  = rng_in.choice(len(test_cells_lo), n_in_bg, replace=False)
    X_bg_te = test_cells_X[ib_idx][:, feat_idx]
    in_ok   = np.isfinite(X_bg_te).all(axis=1); X_bg_te = X_bg_te[in_ok]

    te_ok_p = np.isfinite(X_p_te).all(axis=1)
    X_te_a = np.vstack([X_p_te[te_ok_p], X_bg_te])
    y_te_a = np.concatenate([np.ones(te_ok_p.sum()), np.zeros(len(X_bg_te))])
    if y_te_a.sum() == 0 or (y_te_a == 0).sum() == 0 or len(X_tr_a) < 5:
        return None, None

    try:
        if algo == "RF":
            m = fit_rf(X_tr_a, y_tr_a, hp, seed)
        elif algo == "XGB":
            if not XGB_AVAILABLE: return None, None
            m = fit_xgb(X_tr_a, y_tr_a, hp, seed)
        else:
            m = fit_maxnet(X_p_tr[p_ok], bg_X, hp, seed)
        p_ = m.predict_proba(X_te_a)[:, 1]
        return y_te_a, p_
    except Exception:
        return None, None


def inner_cv(X_pres_tr, pres_tr_lo, pres_tr_la,
             cell_lons_tr, cell_lats_tr, cell_X_tr,
             feat_idx, algo, pa_method, seed,
             n_inner=N_INNER_FOLDS, n_hp=N_HP_CANDIDATES):
    """
    Fully nested spatial inner CV:
    - Inner folds assigned by spatial k-means on training presences.
    - Training PA background sampled from inner TRAINING block cells only.
    - Eval background sampled from inner TEST block cells only.
    - Returns (best_hp, best_threshold) where threshold is Youden J from
      pooled inner OOF predictions — never from the outer test set.
    """
    # Spatial inner fold + centroids for cell-block assignment
    inner_folds, inner_centroids, fold_type = assign_inner_spatial_folds(
        pres_tr_lo, pres_tr_la, n_inner, seed)

    hp_space   = (RF_HP_SPACE if algo == "RF" else
                  XGB_HP_SPACE if algo == "XGB" else MXN_HP_SPACE)
    candidates = list(ParameterSampler(hp_space, n_iter=n_hp, random_state=seed))
    best_auc   = -1.0
    best_hp    = candidates[0]

    # Phase 1: HP search
    for hp in candidates:
        mult = PA_RATIO_MULT.get(hp.get("_pa_ratio", "1:5"), 5)
        aucs = []
        for k in range(n_inner):
            y_te, p_ = _run_inner_fold(
                k, inner_folds, inner_centroids,
                X_pres_tr, pres_tr_lo, pres_tr_la,
                cell_lons_tr, cell_lats_tr, cell_X_tr,
                feat_idx, algo, pa_method, mult, seed, hp)
            if y_te is not None and y_te.sum() >= 1:
                try:
                    aucs.append(roc_auc_score(y_te, p_))
                except Exception:
                    pass
        if aucs and np.mean(aucs) > best_auc:
            best_auc = np.mean(aucs)
            best_hp  = hp

    # Phase 2: collect inner OOF with best HP → derive Youden threshold
    mult = PA_RATIO_MULT.get(best_hp.get("_pa_ratio", "1:5"), 5)
    # Use different seed offset to avoid duplicating Phase 1 background draws
    inner_oof_y, inner_oof_p = [], []
    for k in range(n_inner):
        seed2 = seed + 10000  # distinct from Phase 1 seeds
        y_te, p_ = _run_inner_fold(
            k, inner_folds, inner_centroids,
            X_pres_tr, pres_tr_lo, pres_tr_la,
            cell_lons_tr, cell_lats_tr, cell_X_tr,
            feat_idx, algo, pa_method, mult, seed2, best_hp)
        if y_te is not None:
            inner_oof_y.extend(y_te.tolist())
            inner_oof_p.extend(p_.tolist())

    inner_oof_y = np.array(inner_oof_y)
    inner_oof_p = np.array(inner_oof_p)
    if inner_oof_y.sum() > 0 and (inner_oof_y == 0).sum() > 0:
        best_threshold = youden_threshold(inner_oof_y, inner_oof_p)
    else:
        best_threshold = 0.5

    return best_hp, best_threshold


# ── One outer fold ─────────────────────────────────────────────────────────────

def run_outer_fold_v2(fold_id, rep_seed,
                      pres_tr_lo, pres_tr_la,
                      pres_te_lo, pres_te_la, pres_te_occ_ids,
                      train_cell_lo, train_cell_la,
                      test_cell_lo,  test_cell_la,
                      eval_bg_lo, eval_bg_la,
                      feature_names, arrays, transform,
                      algo, pa_method):
    """
    Leakage-free evaluation:
    - Training background via PA method (training cells only)
    - Evaluation background pre-generated (fixed, random, no test presence info)
    - TSS threshold from inner OOF predictions
    - PA ratio relative to n_training_presences
    - MXN uses same PA-selected background as RF/XGB
    """
    # Feature extraction at training cells
    cell_X_tr_raw = sample_at_coords(arrays, feature_names, train_cell_lo, train_cell_la, transform)
    ok_tr         = np.isfinite(cell_X_tr_raw).all(axis=1)
    cell_X_tr     = cell_X_tr_raw[ok_tr]
    cell_lo_tr    = train_cell_lo[ok_tr]; cell_la_tr = train_cell_la[ok_tr]
    if len(cell_lo_tr) < 5:
        return None, None

    # Feature extraction at presences
    X_pres_tr_raw = sample_at_coords(arrays, feature_names, pres_tr_lo, pres_tr_la, transform)
    X_pres_te_raw = sample_at_coords(arrays, feature_names, pres_te_lo, pres_te_la, transform)
    tr_ok = np.isfinite(X_pres_tr_raw).all(axis=1)
    te_ok = np.isfinite(X_pres_te_raw).all(axis=1)
    X_pres_tr = X_pres_tr_raw[tr_ok]; lo_tr=pres_tr_lo[tr_ok]; la_tr=pres_tr_la[tr_ok]
    X_pres_te = X_pres_te_raw[te_ok]; lo_te=pres_te_lo[te_ok]; la_te=pres_te_la[te_ok]
    occ_ids_te = pres_te_occ_ids[te_ok]
    n_dropped  = int((~te_ok).sum())

    if len(lo_te) == 0 or len(lo_tr) < 3:
        return None, None

    # VIF selection
    n_vif_bg   = min(200, len(cell_lo_tr))
    rng_vif    = np.random.default_rng(rep_seed + fold_id)
    X_vif      = np.vstack([X_pres_tr, cell_X_tr[rng_vif.choice(len(cell_lo_tr), n_vif_bg, replace=False)]])
    feat_idx, sel_names = vif_selection(X_vif, feature_names)

    # Inner CV: HP selection + threshold from inner OOF
    best_hp, inner_threshold = inner_cv(
        X_pres_tr, lo_tr, la_tr,
        cell_lo_tr, cell_la_tr, cell_X_tr,
        feat_idx, algo, pa_method, seed=rep_seed + fold_id * 7
    )

    # PA ratio relative to actual n_training_presences
    pa_ratio_key = best_hp.get("_pa_ratio", "1:5")
    mult = PA_RATIO_MULT.get(pa_ratio_key, 5)
    n_pa = max(1, int(mult * len(lo_tr)))

    # Training data
    bg_tr_lo, bg_tr_la = get_training_pa(pa_method, cell_lo_tr, cell_la_tr,
                                          lo_tr, la_tr, cell_X_tr, n_pa,
                                          seed=rep_seed + fold_id)
    bg_X_tr_raw = _sample_cell_X(bg_tr_lo, bg_tr_la, cell_lo_tr, cell_la_tr, cell_X_tr)
    bg_X_tr = bg_X_tr_raw[:, feat_idx]
    bg_ok_tr = np.isfinite(bg_X_tr).all(axis=1); bg_X_tr = bg_X_tr[bg_ok_tr]

    X_pres_tr_sel = X_pres_tr[:, feat_idx]
    pres_ok_tr    = np.isfinite(X_pres_tr_sel).all(axis=1)
    X_tr_all = np.vstack([X_pres_tr_sel[pres_ok_tr], bg_X_tr])
    y_tr_all = np.concatenate([np.ones(pres_ok_tr.sum()), np.zeros(len(bg_X_tr))])

    # Evaluation data — FIXED background (pre-generated, no test presence used)
    cell_X_eval = sample_at_coords(arrays, feature_names, eval_bg_lo, eval_bg_la, transform)
    eval_sel    = cell_X_eval[:, feat_idx]
    eval_ok     = np.isfinite(eval_sel).all(axis=1)
    eval_sel    = eval_sel[eval_ok]
    eval_lo_f   = eval_bg_lo[eval_ok]; eval_la_f = eval_bg_la[eval_ok]

    X_pres_te_sel = X_pres_te[:, feat_idx]
    pres_ok_te    = np.isfinite(X_pres_te_sel).all(axis=1)
    X_pres_te_f   = X_pres_te_sel[pres_ok_te]
    lo_te_f       = lo_te[pres_ok_te]; la_te_f = la_te[pres_ok_te]
    occ_ids_te_f  = occ_ids_te[pres_ok_te]

    X_te_all = np.vstack([X_pres_te_f, eval_sel])
    y_te_all = np.concatenate([np.ones(len(X_pres_te_f)), np.zeros(len(eval_sel))])

    if y_te_all.sum() == 0 or (y_te_all == 0).sum() == 0 or len(X_tr_all) < 5:
        return None, None

    # Fit model — all algorithms use the same PA-selected training background
    try:
        if algo == "RF":
            model = fit_rf(X_tr_all, y_tr_all, best_hp, seed=rep_seed)
        elif algo == "XGB":
            model = fit_xgb(X_tr_all, y_tr_all, best_hp, seed=rep_seed)
        else:  # MXN: use same PA background (X_pres_tr_sel[pres_ok_tr] + bg_X_tr)
            model = fit_maxnet(X_pres_tr_sel[pres_ok_tr], bg_X_tr, best_hp, seed=rep_seed)
        p_te = model.predict_proba(X_te_all)[:, 1]
    except Exception as e:
        print(f"    ERROR {algo}: {e}")
        return None, None

    # Metrics with threshold from inner OOF (not from test set)
    met = compute_metrics(y_te_all, p_te, threshold=inner_threshold)
    if not met:
        return None, None

    met["n_feat_vif"]     = len(sel_names)
    met["selected"]       = ",".join(sel_names)
    met["pa_ratio"]       = pa_ratio_key
    met["n_pa_actual"]    = int(n_pa)
    met["n_pres_tr"]      = int(len(lo_tr))
    met["n_pres_dropped"] = n_dropped
    met["n_eval_bg"]      = int(len(eval_sel))
    met["inner_threshold"]= round(inner_threshold, 4)
    met["hp_str"]         = json.dumps({k: v for k, v in best_hp.items()
                                         if not k.startswith("_")})

    # OOF rows with occ_id for clustered bootstrap
    p_pres = p_te[:len(X_pres_te_f)]
    p_bg   = p_te[len(X_pres_te_f):]

    oof_rows = [{"occ_id": int(oid), "longitude": float(lo), "latitude": float(la),
                 "suit_oof": float(s), "label": 1}
                for oid, lo, la, s in zip(occ_ids_te_f, lo_te_f, la_te_f, p_pres)]
    for lo, la, s in zip(eval_lo_f, eval_la_f, p_bg):
        oof_rows.append({"occ_id": -1, "longitude": float(lo), "latitude": float(la),
                         "suit_oof": float(s), "label": 0})

    return met, oof_rows


# ── Main ──────────────────────────────────────────────────────────────────────

MIN_PRES_PER_FOLD = 3   # minimum presences in any outer test fold
MIN_PRES_INNER    = 3   # minimum presences in any inner training fold


def check_fold_adequacy(occ_df, k, base_seed, label=""):
    """
    Check minimum presences per fold and return a report dict.
    Returns (fold_sizes, min_size, adequate) where adequate = min_size >= MIN_PRES_PER_FOLD.
    """
    try:
        blk, actual_k, _ = assign_spatial_folds_stable(occ_df, k=k, base_seed=base_seed)
        sizes = [int((blk["spatial_fold"] == f).sum()) for f in range(1, actual_k + 1)]
        min_s = min(sizes)
        ok    = min_s >= MIN_PRES_PER_FOLD
        print(f"  {label}k={actual_k} fold sizes={sizes} min={min_s} "
              f"{'OK' if ok else 'WARNING: <' + str(MIN_PRES_PER_FOLD)}")
        return sizes, min_s, ok
    except Exception as e:
        print(f"  {label}fold check failed: {e}")
        return [], 0, False


def main():
    print("=" * 70)
    print("Step 20 v2: Full Factorial Experiment - LEAKAGE-FREE (Paper 1)")
    print("=" * 70)

    # Load and filter occurrences — enforce year<=2024 regardless of prep script
    occ = pd.read_csv(OCC_FINAL, encoding="utf-8-sig")
    n_raw = len(occ)
    if "year" in occ.columns:
        future = occ[occ["year"] > 2024]
        if len(future) > 0:
            yr_counts = future["year"].value_counts().sort_index()
            print(f"  Year filter: removing {len(future)} records with year>2024:")
            for yr, cnt in yr_counts.items():
                print(f"    year={yr:.0f}: {cnt} records")
        occ = occ[occ["year"] <= 2024].copy()
    occ = occ.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    occ["occ_id"] = occ.index
    n_final = len(occ)
    print(f"  Occurrences: raw={n_raw} -> after year filter={n_final}")

    # ── SAMPLE SIZE AND FOLD ADEQUACY CHECK ──────────────────────────────────
    print(f"\n  {'='*58}")
    print(f"  SAMPLE SIZE AND SPATIAL FOLD ASSESSMENT (n={n_final})")
    print(f"  {'='*58}")
    if n_final < 20:
        print(f"  WARNING: n={n_final} is very small for SDM.")
        print(f"  Standard SDM literature recommends n>=20 for stable CV.")

    # Check if multiple reps produce the SAME partition (degenerate repetition)
    partition_fingerprints = set()
    for rep_check in range(min(5, N_OUTER_REPS)):
        seed_c = RANDOM_STATE + rep_check * 100
        try:
            blk_c, k_c, _ = assign_spatial_folds_stable(occ, k=N_OUTER_FOLDS, base_seed=seed_c)
            # Compare actual memberships, not only fold sizes (balanced folds
            # intentionally have identical size profiles across repetitions).
            fp = tuple(blk_c["spatial_fold"].astype(int).tolist())
            partition_fingerprints.add(fp)
        except Exception:
            pass
    if len(partition_fingerprints) == 1:
        print(f"  DEGENERATE REPETITION: all {N_OUTER_REPS} rep seeds yield the SAME k={N_OUTER_FOLDS} partition.")
        print(f"  '5-repeat' design provides no additional independent evaluation.")
        print("  The 5 reps average PA background sampling noise only, not spatial partition variance.")

    # Determine effective k: prefer k=3 if min>=3, otherwise fall back to k=2
    print(f"\n  k={N_OUTER_FOLDS} fold check:")
    _, min_k3, ok_k3 = check_fold_adequacy(occ, k=N_OUTER_FOLDS, base_seed=RANDOM_STATE, label="  ")
    print(f"\n  k=2 fold check (fallback):")
    _, min_k2, ok_k2 = check_fold_adequacy(occ, k=2, base_seed=RANDOM_STATE, label="  ")

    if not ok_k3 and ok_k2:
        print(f"\n  RECOMMENDATION: use k=2 for this dataset (min={min_k2} >= {MIN_PRES_PER_FOLD}).")
        print(f"  k=3 produces a fold with only {min_k3} test presence(s); AUC unreliable.")
        print(f"  Set N_OUTER_FOLDS_EFFECTIVE = 2 for this run.")
        N_OUTER_FOLDS_EFFECTIVE = 2
    elif not ok_k3 and not ok_k2:
        print(f"\n  WARNING: both k=3 and k=2 have min<{MIN_PRES_PER_FOLD}. Proceeding with k=2.")
        N_OUTER_FOLDS_EFFECTIVE = 2
    else:
        N_OUTER_FOLDS_EFFECTIVE = N_OUTER_FOLDS

    print(f"\n  Inner fold check (approx training n={int(n_final*(N_OUTER_FOLDS_EFFECTIVE-1)/N_OUTER_FOLDS_EFFECTIVE)}):")
    n_inner_sim = int(n_final * (N_OUTER_FOLDS_EFFECTIVE - 1) / N_OUTER_FOLDS_EFFECTIVE)
    occ_inner_sim = occ.sample(n=min(n_inner_sim, len(occ)), random_state=RANDOM_STATE).reset_index(drop=True)
    if len(occ_inner_sim) >= N_INNER_FOLDS:
        check_fold_adequacy(occ_inner_sim, k=N_INNER_FOLDS, base_seed=RANDOM_STATE, label="  inner ")
    print(f"  {'='*58}\n")

    if not XGB_AVAILABLE:
        print("  WARNING: xgboost not installed; XGB skipped")

    print("  Loading rasters...")
    arrays, transform, _ = load_all_rasters()
    common_lo, common_la = get_common_valid_cells(arrays, transform)
    print(f"  Common domain (all features valid + water): {len(common_lo)} cells")

    algorithms = ["RF", "XGB", "MXN"]
    if not XGB_AVAILABLE:
        algorithms = [a for a in algorithms if a != "XGB"]

    all_fold_rows = []
    all_oof_rows  = []
    t_start       = time.time()

    for rep in range(N_OUTER_REPS):
        rep_seed = RANDOM_STATE + rep * 100
        print(f"\n{'='*60}")
        print(f"  Repeat {rep+1}/{N_OUTER_REPS}  (base_seed={rep_seed})")

        occ_blocks, actual_k, centroids = assign_spatial_folds_stable(
            occ, k=N_OUTER_FOLDS_EFFECTIVE, base_seed=rep_seed)
        cell_block = assign_cells_to_blocks(common_lo, common_la, centroids)

        if actual_k < N_OUTER_FOLDS_EFFECTIVE:
            print(f"  NOTE: k-means used k={actual_k} (target={N_OUTER_FOLDS_EFFECTIVE})")

        for fold in range(1, actual_k + 1):
            te_mask = (occ_blocks["spatial_fold"] == fold)
            tr_mask = ~te_mask

            pres_tr_lo  = occ_blocks[tr_mask]["longitude"].values
            pres_tr_la  = occ_blocks[tr_mask]["latitude"].values
            pres_te_lo  = occ_blocks[te_mask]["longitude"].values
            pres_te_la  = occ_blocks[te_mask]["latitude"].values
            pres_te_ids = occ_blocks[te_mask]["occ_id"].values

            if len(pres_te_lo) == 0:
                print(f"  rep={rep+1} fold={fold} SKIP (no test presences)")
                continue

            tr_cell_mask = (cell_block != (fold - 1))
            te_cell_mask = (cell_block == (fold - 1))
            train_lo = common_lo[tr_cell_mask]; train_la = common_la[tr_cell_mask]
            test_lo  = common_lo[te_cell_mask]; test_la  = common_la[te_cell_mask]

            # PRE-GENERATE FIXED EVAL BACKGROUND (leakage-free)
            n_eval  = min(N_EVAL_BG, len(test_lo))
            rng_ev  = np.random.default_rng(rep_seed + fold * 97 + 1000)
            ev_idx  = rng_ev.choice(len(test_lo), n_eval, replace=False)
            eval_bg_lo = test_lo[ev_idx]; eval_bg_la = test_la[ev_idx]

            print(f"\n  rep={rep+1} fold={fold}: "
                  f"n_pres_tr={len(pres_tr_lo)} n_pres_te={len(pres_te_lo)} "
                  f"n_eval_bg={n_eval} test_cells={len(test_lo)}")

            for algo in algorithms:
                for mv_name, mv_feats in MODEL_VARIANTS.items():
                    avail = [f for f in mv_feats if f in arrays]
                    if len(avail) < 3:
                        continue
                    mv_arrays = {k: arrays[k] for k in avail}

                    for pa_method in PA_METHODS:
                        combo_tag = f"{algo}/{mv_name}/{pa_method}"
                        t0 = time.time()

                        met, oof = run_outer_fold_v2(
                            fold_id=fold, rep_seed=rep_seed,
                            pres_tr_lo=pres_tr_lo, pres_tr_la=pres_tr_la,
                            pres_te_lo=pres_te_lo, pres_te_la=pres_te_la,
                            pres_te_occ_ids=pres_te_ids,
                            train_cell_lo=train_lo, train_cell_la=train_la,
                            test_cell_lo=test_lo,   test_cell_la=test_la,
                            eval_bg_lo=eval_bg_lo,  eval_bg_la=eval_bg_la,
                            feature_names=avail, arrays=mv_arrays,
                            transform=transform, algo=algo, pa_method=pa_method,
                        )
                        elapsed = time.time() - t0

                        if met is None:
                            print(f"    {combo_tag}  SKIP")
                            continue

                        row = {"algo": algo, "feat_set": mv_name,
                               "pa_method": pa_method, "rep": rep+1, "fold": fold,
                               "k_effective": N_OUTER_FOLDS_EFFECTIVE, **met}
                        all_fold_rows.append(row)

                        if oof:
                            for r in oof:
                                r.update({"algo": algo, "feat_set": mv_name,
                                          "pa_method": pa_method,
                                          "rep": rep+1, "fold": fold})
                            all_oof_rows.extend(oof)

                        print(f"    {combo_tag}  "
                              f"AUC={met['auc_roc']:.3f} PR={met['pr_auc']:.3f} "
                              f"TSS={met['tss']:.3f} thr={met.get('inner_threshold',0):.3f} "
                              f"PA={met.get('pa_ratio','?')}({met.get('n_pa_actual','?')}) "
                              f"drop={met.get('n_pres_dropped',0)} {elapsed:.1f}s")

    # Save
    df_folds = pd.DataFrame(all_fold_rows)
    df_folds.to_csv(OUT5 / "experiment_fold_metrics.csv", index=False)
    print(f"\n  Fold metrics: {len(df_folds)} rows -> output5/experiment_fold_metrics.csv")

    if all_oof_rows:
        df_oof = pd.DataFrame(all_oof_rows)
        df_oof.to_csv(OUT5 / "experiment_oof_predictions.csv", index=False)
        print(f"  OOF predictions: {len(df_oof)} rows")

    if len(df_folds):
        for label, col in [("algo","algo"), ("feat","feat_set"), ("PA","pa_method")]:
            print(f"\n  AUC by {label}:")
            for v in df_folds[col].unique():
                s = df_folds[df_folds[col]==v]["auc_roc"].dropna()
                print(f"    {v}: {s.mean():.3f} +/- {s.std():.3f} (n={len(s)})")

        if "n_pres_dropped" in df_folds.columns:
            d = df_folds["n_pres_dropped"].dropna()
            print(f"\n  Attrition (NA presence drops): mean={d.mean():.1f} max={d.max():.0f}")
        if "thr_source" in df_folds.columns:
            n_inner = (df_folds["thr_source"]=="inner_oof").sum()
            n_test  = (df_folds["thr_source"]=="test_set").sum()
            print(f"  Threshold source: inner_oof={n_inner}  test_set_fallback={n_test}")

    total_min = (time.time() - t_start) / 60
    print(f"\n  Total elapsed: {total_min:.1f} min")


if __name__ == "__main__":
    main()
