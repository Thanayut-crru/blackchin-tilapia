"""
Step 21 v2 — Pooled OOF metrics + statistical comparison (output5)

FIXES vs v1:
  1. CLUSTERED BOOTSTRAP: CI computed by resampling occ_ids (presences)
     and (rep, fold) pairs (background), not individual rows.
     Avoids underestimating uncertainty from pseudo-replicated OOF.

  2. PAIRED PERMUTATION: Algorithm/feature-set comparisons aggregate to
     matched pairs (one score per level per block = rep × fold × all other
     factors), then use Bernoulli sign-flip permutation. Preserves factorial
     interaction structure. Reports p_paired_perm alongside raw KW.

  3. READS FROM output5/ (leakage-free experiment)

  4. uncertainty_zone (was: shap_fuzzy) in PA method labels

  5. BOYCE CSI: moving-window P/E ratio + Spearman correlation
     (Hirzel et al. 2006), not fixed-bin + Pearson.

  6. PAIRED PERMUTATION STRUCTURE: for each level-pair comparison,
     aggregate to matched pairs (one score per level per unique
     rep × fold × all-other-factors block), then apply Bernoulli
     sign-flip. Correctly preserves factorial experiment structure.
"""

import sys, os, warnings
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss, roc_curve)
from sklearn.linear_model import LogisticRegression

from blackchin_tilapia_analysis.scripts.s00_config import OUT5

OUT5.mkdir(exist_ok=True)

OOF_FILE  = OUT5 / "experiment_oof_predictions.csv"
FOLD_FILE = OUT5 / "experiment_fold_metrics.csv"

N_BOOTSTRAP = 500


# ── Metric helpers ────────────────────────────────────────────────────────────

def youden_thr(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def calibration_slope(y, p):
    eps = 1e-6
    lo  = np.log(np.clip(p, eps, 1-eps) / (1 - np.clip(p, eps, 1-eps)))
    try:
        lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=500)
        lr.fit(lo.reshape(-1, 1), y)
        return float(lr.coef_[0][0]), float(lr.intercept_[0])
    except Exception:
        return np.nan, np.nan


def boyce_csi(p_pres, p_bg, window_width=0.1, n_steps=20):
    """
    Continuous Boyce Index (Hirzel et al. 2006) using moving window P/E ratio
    and Spearman rank correlation. Fixes: moving window (not fixed bins),
    Spearman (not Pearson), P/E uses marginal frequencies.
    """
    try:
        if len(p_pres) < 3 or len(p_bg) < 3:
            return np.nan
        n_pr_total = len(p_pres)
        n_bg_total = len(p_bg)
        steps     = np.linspace(0.0, 1.0 - window_width, n_steps)
        midpoints, pe_vals = [], []
        for lo in steps:
            hi = lo + window_width
            n_pr = ((p_pres >= lo) & (p_pres < hi)).sum()
            n_bg = ((p_bg   >= lo) & (p_bg   < hi)).sum()
            # Expected: proportion of background in this window × total presences
            if n_bg_total == 0:
                continue
            expected_pr = (n_bg / n_bg_total) * n_pr_total
            if expected_pr == 0:
                continue
            pe_vals.append(n_pr / expected_pr)
            midpoints.append(lo + window_width / 2.0)
        if len(pe_vals) < 3:
            return np.nan
        r, _ = stats.spearmanr(midpoints, pe_vals)
        return round(float(r), 4)
    except Exception:
        return np.nan


def full_metrics(y, p, thr_inner=None):
    """
    thr_inner: Youden J threshold derived from inner OOF (passed in from fold_df).
               If None, TSS is not computed to avoid test-set optimism.
               AUC/PR-AUC/Brier are threshold-free and always computed.
    """
    if y.sum() == 0 or (y == 0).sum() == 0:
        return {}
    try:
        boyc = boyce_csi(p[y == 1], p[y == 0])
        cal_sl, cal_int = calibration_slope(y, p)
        out = {
            "auc_roc":   round(float(roc_auc_score(y, p)), 4),
            "pr_auc":    round(float(average_precision_score(y, p)), 4),
            "brier":     round(float(brier_score_loss(y, p)), 4),
            "cal_slope": cal_sl,
            "cal_int":   cal_int,
            "boyce_csi": boyc,
            "n_pres":    int(y.sum()),
            "n_bg":      int((y == 0).sum()),
        }
        if thr_inner is not None and np.isfinite(thr_inner):
            thr = float(thr_inner)
            yp  = (p >= thr).astype(int)
            tp  = int(((yp == 1) & (y == 1)).sum())
            tn  = int(((yp == 0) & (y == 0)).sum())
            fp  = int(((yp == 1) & (y == 0)).sum())
            fn  = int(((yp == 0) & (y == 1)).sum())
            sens = tp / (tp + fn + 1e-9)
            spec = tn / (tn + fp + 1e-9)
            out.update({
                "tss":         round(float(sens + spec - 1.0), 4),
                "tss_thr":     round(thr, 4),
                "sensitivity": round(sens, 4),
                "specificity": round(spec, 4),
                "bal_acc":     round((sens + spec) / 2, 4),
                "thr_source":  "inner_oof",
            })
        else:
            out["thr_source"] = "not_computed"
        return out
    except Exception:
        return {}


# ── Clustered bootstrap CI ────────────────────────────────────────────────────

def bootstrap_ci_clustered(y, p, occ_ids, rep_fold_ids,
                            n_boot=N_BOOTSTRAP, seed=42, metric="auc_roc"):
    """
    Cluster bootstrap: resample occurrence IDs (for presences) and
    (rep, fold) pairs (for background) independently.
    Avoids CI underestimation from pseudo-replicated OOF rows.

    occ_ids:      integer array, occ_id per row (-1 for background)
    rep_fold_ids: string array, "rep_fold" identifier per row
    """
    rng = np.random.default_rng(seed)

    # Unique presence occ_ids and background (rep, fold) cluster IDs
    pres_mask  = (y == 1)
    bg_mask    = (y == 0)
    unique_occ = np.unique(occ_ids[pres_mask]) if pres_mask.any() else np.array([])
    unique_rf  = np.unique(rep_fold_ids[bg_mask]) if bg_mask.any() else np.array([])

    if len(unique_occ) < 2:
        return np.nan, np.nan

    vals = []
    for _ in range(n_boot):
        # Resample occurrence IDs (with replacement)
        boot_occ = rng.choice(unique_occ, len(unique_occ), replace=True)
        # Resample (rep, fold) clusters for background
        boot_rf  = rng.choice(unique_rf, len(unique_rf), replace=True) if len(unique_rf) else unique_rf

        y_b, p_b = [], []
        for oid in boot_occ:
            mask = pres_mask & (occ_ids == oid)
            y_b.extend(y[mask]); p_b.extend(p[mask])
        for rfid in boot_rf:
            mask = bg_mask & (rep_fold_ids == rfid)
            y_b.extend(y[mask]); p_b.extend(p[mask])

        y_b = np.array(y_b); p_b = np.array(p_b)
        if y_b.sum() == 0 or (y_b == 0).sum() == 0:
            continue
        try:
            m = full_metrics(y_b, p_b)
            if metric in m and np.isfinite(m[metric]):
                vals.append(m[metric])
        except Exception:
            pass

    if len(vals) < 10:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ── Pooled OOF metrics ────────────────────────────────────────────────────────

def compute_pooled_oof(oof_df, fold_df=None):
    """
    Compute pooled OOF metrics with clustered bootstrap CI per configuration.

    fold_df: if provided, uses per-(rep,fold) inner_threshold from fold_df
             to compute TSS without test-set optimism.
             TSS is skipped (thr_source='not_computed') if fold_df is None.
    """
    has_occ_id = "occ_id" in oof_df.columns

    # Build threshold lookup: (algo, feat_set, pa_method, rep, fold) -> inner_threshold
    thr_lookup = {}
    if fold_df is not None and "inner_threshold" in fold_df.columns:
        for _, r in fold_df.iterrows():
            key = (r["algo"], r["feat_set"], r["pa_method"], r["rep"], r["fold"])
            thr_lookup[key] = float(r["inner_threshold"])

    rows   = []
    groups = ["algo", "feat_set", "pa_method"]
    for key, grp in oof_df.groupby(groups):
        y    = grp["label"].values.astype(int)
        p    = grp["suit_oof"].values.astype(float)
        ok   = np.isfinite(p) & (p >= 0) & (p <= 1)
        y    = y[ok]; p = p[ok]
        if len(y) < 5 or y.sum() == 0 or (y == 0).sum() == 0:
            continue

        # Weighted-average inner threshold across all (rep, fold) for this config
        algo, fs, pm = key if isinstance(key, tuple) else (key, None, None)
        thr_vals = [v for (a, f, m_, r, fo), v in thr_lookup.items()
                    if a == algo and f == fs and m_ == pm]
        thr_inner = float(np.mean(thr_vals)) if thr_vals else None

        m = full_metrics(y, p, thr_inner=thr_inner)
        if not m:
            continue

        # Clustered bootstrap CI (threshold-free metrics only for robustness)
        if has_occ_id:
            occ_ids = grp["occ_id"].values.astype(int)[ok]
            rf_ids  = (grp["rep"].astype(str) + "_" + grp["fold"].astype(str)).values[ok]
        else:
            occ_ids = np.arange(len(y))
            rf_ids  = np.zeros(len(y), dtype=str)

        for met_name in ["auc_roc", "pr_auc", "brier"]:
            lo, hi = bootstrap_ci_clustered(
                y, p, occ_ids, rf_ids, metric=met_name, seed=42)
            m[f"{met_name}_ci_lo"] = round(lo, 4) if np.isfinite(lo) else np.nan
            m[f"{met_name}_ci_hi"] = round(hi, 4) if np.isfinite(hi) else np.nan

        row = dict(zip(groups, key if isinstance(key, tuple) else (key,)))
        row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
        rows.append(row)

    return pd.DataFrame(rows)


# ── Statistical comparison ────────────────────────────────────────────────────

def _paired_permutation(fold_df, factor, response, n_perm=500, seed=42):
    """
    Proper paired permutation test for each level-pair.

    For factors with >2 levels, tests each unique pair. Within each pair:
    - Aggregate to matched observations: for each unique combination of
      (rep, fold, <all OTHER factors except `factor`>), we have one score
      for level A and one for level B → these form a matched pair.
    - Observed statistic: mean(A_score - B_score) across all matched pairs.
    - Permutation: randomly swap A and B within each matched pair (Bernoulli).
    - p-value (two-tailed): (extreme + 1) / (n_perm + 1), preventing
      impossible zero Monte Carlo p-values.

    This preserves the factorial interaction structure, unlike label-shuffling
    within (rep, fold) blocks, which ignores feat_set or pa_method pairing.
    """
    rng = np.random.default_rng(seed)
    df  = fold_df.dropna(subset=[response]).copy()
    other_factors = [f for f in ["algo", "feat_set", "pa_method"] if f != factor]
    # Block columns: rep, fold + all other experimental factors
    block_cols = ["rep", "fold"] + other_factors

    levels = sorted(df[factor].unique())
    rows   = []

    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            la, lb = levels[i], levels[j]
            df_ab  = df[df[factor].isin([la, lb])].copy()

            # Build matched pairs: one observation per level per block
            pairs_a, pairs_b = [], []
            for block_key, grp in df_ab.groupby(block_cols):
                row_a = grp[grp[factor] == la][response].values
                row_b = grp[grp[factor] == lb][response].values
                if len(row_a) == 1 and len(row_b) == 1:
                    pairs_a.append(row_a[0])
                    pairs_b.append(row_b[0])

            if len(pairs_a) < 3:
                continue

            pairs_a = np.array(pairs_a)
            pairs_b = np.array(pairs_b)
            diffs   = pairs_a - pairs_b
            obs_stat = diffs.mean()

            # Paired permutation: swap A and B within each pair independently
            count_ge = 0
            for _ in range(n_perm):
                signs     = rng.choice([-1, 1], size=len(diffs))
                perm_stat = (diffs * signs).mean()
                if abs(perm_stat) >= abs(obs_stat):
                    count_ge += 1

            p_val = (count_ge + 1) / (n_perm + 1)
            rows.append({
                "factor":         factor,
                "level_a":        la,
                "level_b":        lb,
                "mean_a":         round(float(pairs_a.mean()), 4),
                "mean_b":         round(float(pairs_b.mean()), 4),
                "diff_a_b":       round(float(obs_stat), 4),
                "n_pairs":        len(pairs_a),
                "p_paired_perm":  round(p_val, 4),
                "response":       response,
            })

    df_out = pd.DataFrame(rows)
    if len(df_out) == 0:
        return df_out

    # Holm-Bonferroni correction within this factor's pairwise tests
    pvals = df_out["p_paired_perm"].values
    m = len(pvals)
    order   = np.argsort(pvals)
    p_adj   = np.zeros(m)
    for rank, idx in enumerate(order):
        correction = m - rank
        adjusted   = pvals[idx] * correction
        p_adj[idx] = min(adjusted, 1.0)
    # Holm ensures monotonicity
    for rank_pos, idx in enumerate(order[1:], start=1):
        prev_idx = order[rank_pos - 1]
        if p_adj[idx] < p_adj[prev_idx]:
            p_adj[idx] = p_adj[prev_idx]

    df_out["p_holm"] = np.round(p_adj, 4)
    df_out["sig"]    = df_out["p_holm"].apply(
        lambda v: "**" if v < 0.01 else ("*" if v < 0.05 else "ns"))

    return df_out


def kruskal_comparison(fold_df, response="auc_roc"):
    """
    Kruskal-Wallis H-test per factor.
    Note: rows are not independent (pseudo-replication) — use blocked
    permutation results for inference; KW is reported for transparency.
    """
    rows = []
    for factor in ["algo", "feat_set", "pa_method"]:
        groups = [grp[response].dropna().values
                  for _, grp in fold_df.groupby(factor)]
        if len(groups) < 2:
            continue
        try:
            stat, p = stats.kruskal(*groups)
            rows.append({
                "factor":    factor,
                "kruskal_H": round(stat, 4),
                "p_value":   round(p, 4),
                "response":  response,
                "note":      "non-independent rows; see blocked-perm for inference",
            })
        except Exception:
            pass
    return pd.DataFrame(rows)


def calibration_data(oof_df, n_bins=10):
    rows = []
    for (algo, fs, pm), grp in oof_df.groupby(["algo","feat_set","pa_method"]):
        y = grp["label"].values.astype(int)
        p = grp["suit_oof"].values.astype(float)
        ok = np.isfinite(p)
        y = y[ok]; p = p[ok]
        bins = np.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            mask = (p >= bins[i]) & (p < bins[i+1])
            if mask.sum() == 0:
                continue
            rows.append({
                "algo": algo, "feat_set": fs, "pa_method": pm,
                "bin_midpoint": round((bins[i]+bins[i+1])/2, 2),
                "mean_pred":    round(float(p[mask].mean()), 4),
                "mean_obs":     round(float(y[mask].mean()), 4),
                "n_pts":        int(mask.sum()),
            })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Step 21 v2: Pooled OOF Metrics + Statistical Comparison (output5)")
    print("  Clustered bootstrap CI (by occ_id)")
    print("  Blocked permutation tests (within rep x fold)")
    print("=" * 70)

    if not OOF_FILE.exists():
        print(f"  OOF file not found: {OOF_FILE}")
        print("  Run step 20 v2 first: python -m scripts_v2.20_full_experiment_v2")
        return
    if not FOLD_FILE.exists():
        print(f"  Fold metrics not found: {FOLD_FILE}")
        return

    oof_df  = pd.read_csv(OOF_FILE)
    fold_df = pd.read_csv(FOLD_FILE)
    print(f"  OOF rows:    {len(oof_df)}")
    print(f"  Fold rows:   {len(fold_df)}")
    print(f"  Has occ_id:  {'occ_id' in oof_df.columns}")

    n_pres = (oof_df["label"] == 1).sum() if "label" in oof_df.columns else 0
    n_bg   = (oof_df["label"] == 0).sum() if "label" in oof_df.columns else 0
    print(f"  OOF presences: {n_pres}  backgrounds: {n_bg}")

    # Attrition report
    if "n_pres_dropped" in fold_df.columns:
        drop = fold_df["n_pres_dropped"].dropna()
        print(f"  Attrition: mean dropped/fold={drop.mean():.1f}  "
              f"max={drop.max():.0f}")

    # A. Pooled OOF metrics with clustered bootstrap CI
    print("\n[A] Pooled OOF metrics with clustered bootstrap CI (n_boot=500)...")
    pooled = compute_pooled_oof(oof_df, fold_df=fold_df)
    if len(pooled):
        pooled.to_csv(OUT5 / "pooled_oof_metrics.csv", index=False)
        print(f"  Saved: pooled_oof_metrics.csv ({len(pooled)} configurations)")

        auc_col = "auc_roc"
        if auc_col in pooled.columns:
            top = (pooled.sort_values(auc_col, ascending=False)
                         .groupby(["algo","feat_set"])
                         .first().reset_index())
            disp = [c for c in ["algo","feat_set","pa_method",
                                 "auc_roc","auc_roc_ci_lo","auc_roc_ci_hi",
                                 "pr_auc","tss","brier"] if c in top.columns]
            print("\n  Best configuration per (algo, feat_set):")
            print(top[disp].to_string(index=False))
            top.to_csv(OUT5 / "pooled_best_per_model.csv", index=False)

    # B. Statistical comparison
    print("\n[B] Statistical comparison...")
    for response in ["auc_roc", "pr_auc", "tss"]:
        kw = kruskal_comparison(fold_df, response)
        kw.to_csv(OUT5 / f"stats_kruskal_{response}.csv", index=False)
        print(f"  Kruskal-Wallis ({response}):")
        if len(kw):
            print(kw[["factor","kruskal_H","p_value"]].to_string(index=False))

    # Paired permutation pairwise comparisons (AUC only — most critical)
    print("\n  Paired permutation pairwise comparisons (AUC-ROC)...")
    print("  NOTE: this may take a few minutes (500 permutations x 3 factors)")
    for factor in ["algo", "feat_set", "pa_method"]:
        pw = _paired_permutation(fold_df, factor, "auc_roc", n_perm=500, seed=42)
        if len(pw):
            pw.to_csv(OUT5 / f"stats_paired_perm_{factor}.csv", index=False)
            print(f"\n  {factor} pairwise (paired permutation + Holm correction, AUC-ROC):")
            print(pw[["level_a","level_b","mean_a","mean_b",
                       "diff_a_b","n_pairs","p_paired_perm","p_holm","sig"]].to_string(index=False))

    # C. Calibration data
    print("\n[C] Calibration data...")
    cal = calibration_data(oof_df)
    if len(cal):
        cal.to_csv(OUT5 / "calibration_data.csv", index=False)
        print(f"  calibration_data.csv ({len(cal)} rows)")

    # D. AUC matrix
    if len(fold_df):
        pivot = (fold_df.groupby(["algo","feat_set"])["auc_roc"]
                        .mean().round(3).unstack("feat_set"))
        print("\n  AUC matrix (algo x feat_set, averaged over PA method):")
        print(pivot.to_string())
        pivot.to_csv(OUT5 / "auc_matrix_algo_featset.csv")

    print(f"\n  Done. Outputs: output5/")
    print("  NOTE: CI from clustered bootstrap is wider than row-level bootstrap.")
    print("  NOTE: Paired permutation p-values preserve factorial structure; more conservative than plain KW.")
    print("  NOTE: Boyce CSI uses moving-window P/E ratio with Spearman correlation (Hirzel et al. 2006).")


if __name__ == "__main__":
    main()
