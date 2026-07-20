"""
Step 23 — Publication figures for output5

Generates:
  Fig1  : ROC curves per algorithm × feature set (4-panel)
  Fig2  : PR curves per algorithm × feature set (4-panel)
  Fig3  : Calibration plots (predicted vs observed)
  Fig4  : AUC heatmap: algorithm × feature set × PA method
  Fig5  : Algorithm comparison boxplot (AUC, PR-AUC, TSS)
  Fig6  : SHAP importance comparison across algorithms
  Fig7  : Ensemble suitability map (mean of RF/XGB/MXN M_SAP)
  Fig8  : Model disagreement map (SD across algorithms)
  FigS1 : Sensitivity analysis tornado plot
  FigS2 : Calibration slope per model

All saved as PNG + PDF at 300 DPI in output5/figures/
"""

import sys, os, warnings
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from blackchin_tilapia_analysis.scripts.s00_config import DATA, OUT3, OUT5, THAILAND_BBOX

FIG_DIR  = OUT5 / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OOF_FILE  = OUT5 / "experiment_oof_predictions.csv"
FOLD_FILE = OUT5 / "experiment_fold_metrics.csv"
POOL_FILE = OUT5 / "pooled_oof_metrics.csv"
SENS_FILE = OUT5 / "sensitivity" / "sensitivity_summary.csv"
MAP_DIR   = OUT5 / "maps"

GADM0 = DATA / "thailand_boundary.geojson"
WATER_TIF = DATA / "water" / "water_mask_envgrid.tif"

ALGO_COLORS  = {"RF": "#1b7837", "XGB": "#762a83", "MXN": "#d6604d"}
FEAT_COLORS  = {"M_S": "#4393c3", "M_SA": "#f4a582", "M_SP": "#92c5de", "M_SAP": "#d6604d"}
ALGO_LABELS  = {"RF": "Random Forest", "XGB": "XGBoost", "MXN": "MaxNet"}
FEAT_LABELS  = {"M_S": "M_S (Env)", "M_SA": "M_SA (Env+A)",
                "M_SP": "M_SP (Env+P)", "M_SAP": "M_SAP (All)"}
PA_LABELS    = {"random": "Random", "spatial_constrained": "Spatial",
                "two_step": "Two-step", "uncertainty_zone": "Uncertainty Zone"}

matplotlib.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize":8,
    "ytick.labelsize":8,
    "legend.fontsize":8,
    "figure.dpi":     150,
})

DPI_SAVE = 300


def savefig(fig, name):
    p_png = FIG_DIR / f"{name}.png"
    p_pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(str(p_png), dpi=DPI_SAVE, bbox_inches="tight")
    fig.savefig(str(p_pdf), dpi=DPI_SAVE, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}.png/.pdf")


def load_csv(path, fallback=None):
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    if fallback and Path(fallback).exists():
        return pd.read_csv(fallback)
    return pd.DataFrame()


# ── Fig 1 & 2: ROC and PR curves ─────────────────────────────────────────────

def fig_roc_pr(oof_df, which="roc"):
    feat_sets = ["M_S", "M_SA", "M_SP", "M_SAP"]
    algos     = sorted(oof_df["algo"].unique()) if "algo" in oof_df.columns else ["RF"]
    n_algo    = len(algos)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    axes      = axes.flatten()
    title_sfx = "ROC" if which == "roc" else "Precision-Recall"

    for ai, fs in enumerate(feat_sets):
        ax = axes[ai]
        sub = oof_df[oof_df["feat_set"] == fs] if "feat_set" in oof_df.columns else oof_df
        for algo in algos:
            a_sub = sub[sub["algo"] == algo] if "algo" in sub.columns else sub
            if len(a_sub) == 0:
                continue
            y  = a_sub["label"].values.astype(int)
            p  = a_sub["suit_oof"].values.astype(float)
            ok = np.isfinite(p) & (p >= 0) & (p <= 1)
            y  = y[ok]; p = p[ok]
            if y.sum() == 0:
                continue
            color = ALGO_COLORS.get(algo, "gray")
            if which == "roc":
                fpr, tpr, _ = roc_curve(y, p)
                auc_val = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, lw=1.5,
                        label=f"{ALGO_LABELS.get(algo, algo)} (AUC={auc_val:.3f})")
            else:
                prec, rec, _ = precision_recall_curve(y, p)
                pr_auc = auc(rec, prec)
                ax.plot(rec, prec, color=color, lw=1.5,
                        label=f"{ALGO_LABELS.get(algo, algo)} (PR-AUC={pr_auc:.3f})")

        if which == "roc":
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        else:
            prev = y.mean() if len(y) > 0 else 0.1
            ax.axhline(prev, color="k", linestyle="--", lw=0.8, alpha=0.5)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")

        ax.set_title(FEAT_LABELS.get(fs, fs))
        ax.legend(loc="lower right" if which == "roc" else "upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Pooled OOF {title_sfx} Curves by Feature Set and Algorithm",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


# ── Fig 3: Calibration plots ──────────────────────────────────────────────────

def fig_calibration(cal_df, oof_df):
    feat_sets = ["M_S", "M_SA", "M_SP", "M_SAP"]
    algos = sorted(oof_df["algo"].unique()) if "algo" in oof_df.columns else ["RF"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes      = axes.flatten()
    for ai, fs in enumerate(feat_sets):
        ax = axes[ai]
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Perfect calibration")
        for algo in algos:
            if len(cal_df):
                sub_cal = cal_df[(cal_df["feat_set"] == fs) & (cal_df["algo"] == algo)]
            else:
                sub_oof = oof_df[(oof_df["feat_set"] == fs) & (oof_df["algo"] == algo)] \
                          if "feat_set" in oof_df.columns else oof_df[oof_df["algo"] == algo]
                y  = sub_oof["label"].values.astype(int) if len(sub_oof) else np.array([])
                p  = sub_oof["suit_oof"].values.astype(float) if len(sub_oof) else np.array([])
                if len(y) < 5:
                    continue
                bins = np.linspace(0, 1, 11)
                x_c = []; y_c = []
                for i in range(10):
                    m = (p >= bins[i]) & (p < bins[i+1])
                    if m.sum() > 0:
                        x_c.append(p[m].mean()); y_c.append(y[m].mean())
                sub_cal = pd.DataFrame({"mean_pred": x_c, "mean_obs": y_c})
            if len(sub_cal) == 0:
                continue
            color = ALGO_COLORS.get(algo, "gray")
            ax.plot(sub_cal["mean_pred"], sub_cal["mean_obs"],
                    "o-", color=color, lw=1.5, ms=4,
                    label=ALGO_LABELS.get(algo, algo))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted suitability"); ax.set_ylabel("Mean observed rate")
        ax.set_title(FEAT_LABELS.get(fs, fs))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Calibration Plots (Pooled OOF)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


# ── Fig 4: AUC heatmap ────────────────────────────────────────────────────────

def fig_auc_heatmap(fold_df):
    if len(fold_df) == 0:
        return None
    feat_sets = ["M_S","M_SA","M_SP","M_SAP"]
    algos     = sorted(fold_df["algo"].unique())
    pa_methods= sorted(fold_df["pa_method"].unique()) if "pa_method" in fold_df.columns else ["random"]

    n_pa = len(pa_methods)
    fig, axes = plt.subplots(1, n_pa, figsize=(4 * n_pa, 4.5), sharey=True)
    if n_pa == 1:
        axes = [axes]

    for pi, pm in enumerate(pa_methods):
        ax = axes[pi]
        sub = fold_df[fold_df["pa_method"] == pm] if "pa_method" in fold_df.columns else fold_df
        mat = np.full((len(algos), len(feat_sets)), np.nan)
        for ai, algo in enumerate(algos):
            for fi, fs in enumerate(feat_sets):
                grp = sub[(sub["algo"] == algo) & (sub["feat_set"] == fs)]["auc_roc"].dropna()
                if len(grp) > 0:
                    mat[ai, fi] = grp.mean()
        im = ax.imshow(mat, vmin=0.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(feat_sets)))
        ax.set_xticklabels([FEAT_LABELS.get(f, f) for f in feat_sets], rotation=30, ha="right")
        if pi == 0:
            ax.set_yticks(range(len(algos)))
            ax.set_yticklabels([ALGO_LABELS.get(a, a) for a in algos])
        for ai in range(len(algos)):
            for fi in range(len(feat_sets)):
                v = mat[ai, fi]
                if np.isfinite(v):
                    ax.text(fi, ai, f"{v:.3f}", ha="center", va="center", fontsize=8,
                            color="black" if v > 0.65 else "white")
        ax.set_title(PA_LABELS.get(pm, pm))
        plt.colorbar(im, ax=ax, shrink=0.8, label="Mean AUC")

    fig.suptitle("Mean AUC-ROC: Algorithm × Feature Set × PA Method",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


# ── Fig 5: Algorithm comparison boxplot ──────────────────────────────────────

def fig_boxplot_comparison(fold_df):
    if len(fold_df) == 0:
        return None
    algos   = sorted(fold_df["algo"].unique())
    metrics = ["auc_roc", "pr_auc", "tss", "brier"]
    labels  = ["AUC-ROC", "PR-AUC", "TSS (Youden)", "Brier Score"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for mi, (met, lbl) in enumerate(zip(metrics, labels)):
        ax = axes[mi]
        data   = [fold_df[fold_df["algo"] == a][met].dropna().values for a in algos]
        bp     = ax.boxplot(data, patch_artist=True, widths=0.5)
        for patch, algo in zip(bp["boxes"], algos):
            patch.set_facecolor(ALGO_COLORS.get(algo, "gray"))
            patch.set_alpha(0.8)
        ax.set_xticks(range(1, len(algos)+1))
        ax.set_xticklabels([ALGO_LABELS.get(a, a) for a in algos], rotation=15)
        ax.set_ylabel(lbl)
        ax.set_title(lbl)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Algorithm Performance Comparison (all feature sets pooled)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


# ── Fig 7 & 8: Ensemble and disagreement maps ─────────────────────────────────

def load_suit_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nd  = src.nodata
        if nd is not None:
            arr[np.abs(arr - nd) < 1e-3] = np.nan
        arr[arr < 0] = np.nan
        return arr, src.transform


def fig_ensemble_maps():
    """
    Load all available model suitability rasters, compute ensemble mean and SD.
    Returns (fig_ensemble, fig_disagreement).
    """
    suit_files = list(MAP_DIR.glob("suitability_*.tif")) if MAP_DIR.exists() else []
    if not suit_files:
        return None, None

    arrays_list = []
    labels      = []
    transform   = None
    for f in sorted(suit_files):
        try:
            arr, tr = load_suit_raster(f)
            arrays_list.append(arr)
            labels.append(f.stem.replace("suitability_",""))
            if transform is None:
                transform = tr
        except Exception:
            pass

    if not arrays_list:
        return None, None

    stack    = np.stack(arrays_list, axis=0)
    ens_mean = np.nanmean(stack, axis=0)
    ens_sd   = np.nanstd(stack, axis=0)

    def _make_map(arr, title, cmap="YlOrRd", label="Suitability"):
        try:
            import geopandas as gpd
            from matplotlib.patches import PathPatch
            from matplotlib.path import Path as MplPath
        except ImportError:
            gpd = None

        # Clip raster to Thailand boundary polygon
        arr_plot = arr.copy()
        if gpd is not None and GADM0.exists():
            try:
                from rasterio.features import geometry_mask
                import rasterio.transform as rtransform
                thai_geom = gpd.read_file(GADM0).to_crs("EPSG:4326").geometry.values
                H, W = arr.shape
                th_mask = geometry_mask(
                    thai_geom,
                    transform=transform,
                    invert=False,  # True = inside polygon, False = outside
                    out_shape=(H, W),
                )
                # th_mask is True OUTSIDE Thailand — mask those out
                arr_plot[th_mask] = np.nan
            except Exception:
                pass  # If clipping fails, proceed without it

        fig, ax = plt.subplots(figsize=(5, 8))
        H, W = arr_plot.shape
        xmin = transform.c; xmax = transform.c + W * transform.a
        ymax = transform.f; ymin = transform.f + H * transform.e
        ext  = [xmin, xmax, ymin, ymax]
        valid = arr_plot[np.isfinite(arr_plot)]
        vmax  = float(np.nanquantile(valid, 0.99)) if len(valid) > 0 else 1.0
        im   = ax.imshow(arr_plot, extent=ext, cmap=cmap, origin="upper",
                         vmin=0, vmax=vmax, aspect="auto")
        if gpd is not None and GADM0.exists():
            gpd.read_file(GADM0).to_crs("EPSG:4326").boundary.plot(
                ax=ax, linewidth=0.8, color="black")
        plt.colorbar(im, ax=ax, label=label, shrink=0.6)
        ax.set_xlim(THAILAND_BBOX["xmin"], THAILAND_BBOX["xmax"])
        ax.set_ylim(THAILAND_BBOX["ymin"], THAILAND_BBOX["ymax"])
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title(title)
        plt.tight_layout()
        return fig

    fig_ens  = _make_map(ens_mean, "Ensemble Mean Suitability\n(RF + XGB + MaxNet × all feature sets)", "YlOrRd")
    fig_dis  = _make_map(ens_sd,   "Model Disagreement\n(SD across algorithms × feature sets)", "PuBu", "SD of suitability")
    return fig_ens, fig_dis


# ── FigS1: Sensitivity tornado ────────────────────────────────────────────────

def fig_sensitivity_tornado(sens_df):
    if len(sens_df) == 0:
        return None
    factors = sens_df["sensitivity_factor"].unique()
    base_val = sens_df.groupby("sensitivity_factor")["auc_mean"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = plt.cm.Set2(np.linspace(0, 1, len(factors)))
    y_pos   = 0
    y_ticks = []; y_labels = []
    for i, fac in enumerate(factors):
        sub  = sens_df[sens_df["sensitivity_factor"] == fac].sort_values("auc_mean")
        lo   = sub["auc_mean"].min(); hi = sub["auc_mean"].max()
        ax.barh(y_pos, hi - lo, left=lo, color=colors[i], alpha=0.8, height=0.6)
        ax.text(lo - 0.005, y_pos, f"{lo:.3f}", va="center", ha="right", fontsize=8)
        ax.text(hi + 0.005, y_pos, f"{hi:.3f}", va="center", ha="left",  fontsize=8)
        y_ticks.append(y_pos); y_labels.append(fac.replace("_", " "))
        y_pos += 1

    ax.set_yticks(y_ticks); ax.set_yticklabels(y_labels)
    ax.set_xlabel("Mean AUC-ROC")
    ax.set_title("Sensitivity Analysis: Range of AUC across factor values")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Step 23: Publication Figures (Paper 1)")
    print("=" * 65)

    oof_df  = load_csv(OOF_FILE)
    fold_df = load_csv(FOLD_FILE)
    pool_df = load_csv(POOL_FILE)
    cal_df  = load_csv(OUT5 / "calibration_data.csv")
    sens_df = load_csv(SENS_FILE)

    if len(oof_df) == 0:
        print("  No OOF predictions found. Run steps 20-22 first.")
    else:
        print(f"  OOF: {len(oof_df)} rows | Fold: {len(fold_df)} | Pool: {len(pool_df)}")

    # Fig 1: ROC curves
    if len(oof_df) > 0:
        fig = fig_roc_pr(oof_df, "roc")
        savefig(fig, "Fig1_roc_curves")

    # Fig 2: PR curves
    if len(oof_df) > 0:
        fig = fig_roc_pr(oof_df, "pr")
        savefig(fig, "Fig2_pr_curves")

    # Fig 3: Calibration
    if len(oof_df) > 0:
        fig = fig_calibration(cal_df, oof_df)
        savefig(fig, "Fig3_calibration")

    # Fig 4: AUC heatmap
    if len(fold_df) > 0:
        fig = fig_auc_heatmap(fold_df)
        if fig:
            savefig(fig, "Fig4_auc_heatmap")

    # Fig 5: Boxplot comparison
    if len(fold_df) > 0:
        fig = fig_boxplot_comparison(fold_df)
        if fig:
            savefig(fig, "Fig5_algorithm_boxplot")

    # Fig 7 & 8: Ensemble and disagreement maps
    fig_ens, fig_dis = fig_ensemble_maps()
    if fig_ens:
        savefig(fig_ens, "Fig7_ensemble_suitability")
    if fig_dis:
        savefig(fig_dis, "Fig8_model_disagreement")

    # FigS1: Sensitivity
    if len(sens_df) > 0:
        fig = fig_sensitivity_tornado(sens_df)
        if fig:
            savefig(fig, "FigS1_sensitivity_tornado")

    figs = list(FIG_DIR.glob("*.png"))
    print(f"\n  Saved {len(figs)} figures to {FIG_DIR}")
    for f in sorted(figs):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
