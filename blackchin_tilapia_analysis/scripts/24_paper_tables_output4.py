"""
Step 24 — Paper tables for output5 (multi-algorithm full factorial experiment)

Generates manuscript-ready tables:
  Table A: Algorithm x Feature Set AUC matrix (mean +/- SD, 95% CI)
  Table B: Best configuration per algorithm (pooled OOF metrics)
  Table C: Statistical comparison (Kruskal-Wallis + pairwise paired permutation + Holm)
  Table D: PA method comparison
  Table E: Sensitivity analysis summary
  Table F: Province risk (output5/validation/)

All CSV outputs: output5/paper_tables/
"""

import sys, os, warnings
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from blackchin_tilapia_analysis.scripts.s00_config import OUT3, OUT5

OUT_TAB  = OUT5 / "paper_tables"
OUT_TAB.mkdir(exist_ok=True)


def load(path, **kwargs):
    p = Path(path)
    return pd.read_csv(p, **kwargs) if p.exists() else pd.DataFrame()


# ── Table A: AUC matrix (algo x feat_set x PA method) ────────────────────────

def table_a_auc_matrix():
    fold = load(OUT5 / "experiment_fold_metrics.csv")
    if len(fold) == 0:
        print("  Table A: fold metrics not found"); return None

    algos = ["RF", "XGB", "MXN"]
    feats = ["M_S", "M_SA", "M_SP", "M_SAP"]

    rows = []
    for algo in algos:
        for fs in feats:
            sub = fold[(fold["algo"] == algo) & (fold["feat_set"] == fs)]
            if len(sub) == 0:
                continue
            for met in ["auc_roc", "pr_auc", "tss", "brier"]:
                v = sub[met].dropna()
                rows.append({
                    "algo": algo, "feat_set": fs, "metric": met,
                    "mean": round(v.mean(), 4), "sd": round(v.std(), 4),
                    "n_folds": len(v),
                })

    df = pd.DataFrame(rows)

    # Pivot for AUC only: algo (rows) x feat_set (cols)
    auc_df = df[df["metric"] == "auc_roc"].copy()
    auc_df["mean_sd"] = auc_df.apply(lambda r: f"{r['mean']:.3f} ({r['sd']:.3f})", axis=1)
    pivot = auc_df.pivot_table(index="algo", columns="feat_set", values="mean_sd", aggfunc="first")
    pivot = pivot.reindex(index=algos, columns=feats)
    pivot.to_csv(OUT_TAB / "tableA_auc_matrix.csv")
    print(f"  Table A (AUC matrix):")
    print(pivot.to_string())

    # Full metrics table (all algo x feat_set x metric)
    df.to_csv(OUT_TAB / "tableA_full_metrics.csv", index=False)
    return df


# ── Table B: Best configuration per algorithm (pooled OOF) ───────────────────

def table_b_best_config():
    pool = load(OUT5 / "pooled_oof_metrics.csv")
    fold = load(OUT5 / "experiment_fold_metrics.csv")

    if len(pool) == 0 and len(fold) == 0:
        print("  Table B: no pooled metrics"); return None

    # Use pooled OOF if it has AUC, else use fold-level mean
    if "auc_roc" in pool.columns:
        auc_col = "auc_roc"
    elif "auc_roc_mean" in pool.columns:
        auc_col = "auc_roc_mean"
    else:
        print("  Table B: no AUC column in pooled metrics"); return None

    # Best configuration per (algo, feat_set)
    best = (pool.sort_values(auc_col, ascending=False)
                .groupby(["algo", "feat_set"])
                .first().reset_index())

    # Format CI columns
    ci_lo = "auc_roc_ci_lo" if "auc_roc_ci_lo" in best.columns else None
    ci_hi = "auc_roc_ci_hi" if "auc_roc_ci_hi" in best.columns else None

    out_rows = []
    for _, r in best.iterrows():
        row = {
            "Algorithm": r["algo"],
            "Feature Set": r["feat_set"],
            "Best PA Method": r["pa_method"],
        }
        auc_val = r[auc_col]
        if ci_lo and ci_hi and pd.notna(r.get(ci_lo)) and pd.notna(r.get(ci_hi)):
            row["AUC-ROC (95% CI)"] = f"{auc_val:.3f} [{r[ci_lo]:.3f}–{r[ci_hi]:.3f}]"
        else:
            row["AUC-ROC"] = f"{auc_val:.3f}"

        for met, label in [("pr_auc","PR-AUC"), ("tss","TSS"), ("brier","Brier")]:
            colname = met if met in r.index else f"{met}_mean"
            if colname in r.index and pd.notna(r[colname]):
                row[label] = f"{r[colname]:.3f}"
        out_rows.append(row)

    tab = pd.DataFrame(out_rows)
    tab.to_csv(OUT_TAB / "tableB_best_config.csv", index=False)
    print(f"  Table B (Best configuration per algo x feat_set):")
    print(tab.to_string(index=False))
    return tab


# ── Table C: Statistical comparison ──────────────────────────────────────────

def table_c_stats():
    rows = []

    # Kruskal-Wallis results (stats_kruskal_*.csv from 21_v2)
    for met in ["auc_roc", "pr_auc", "tss"]:
        p = OUT5 / f"stats_kruskal_{met}.csv"
        if p.exists():
            df = pd.read_csv(p)
            if "kruskal_H" in df.columns:
                for _, r in df.iterrows():
                    rows.append({
                        "Response": met.upper().replace("_", "-"),
                        "Factor": r["factor"],
                        "Test": "Kruskal-Wallis",
                        "Statistic": f"H={r['kruskal_H']:.2f}",
                        "p-value": r["p_value"],
                        "Note": r.get("note", "non-independent; see paired perm"),
                    })

    tab_kw = pd.DataFrame(rows)
    tab_kw.to_csv(OUT_TAB / "tableC_kruskal_wallis.csv", index=False)
    print(f"  Table C (Kruskal-Wallis):")
    if len(tab_kw):
        print(tab_kw.to_string(index=False))

    # Paired permutation pairwise comparisons + Holm correction (from 21_v2)
    pw_rows = []
    for factor in ["algo", "feat_set", "pa_method"]:
        p = OUT5 / f"stats_paired_perm_{factor}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["factor"] = factor
            pw_rows.append(df)

    if pw_rows:
        pw = pd.concat(pw_rows, ignore_index=True)
        pw.to_csv(OUT_TAB / "tableC_pairwise.csv", index=False)
        print(f"\n  Pairwise paired-perm + Holm ({len(pw)} comparisons):")
        disp = [c for c in ["factor","level_a","level_b","diff_a_b",
                             "p_paired_perm","p_holm","sig"] if c in pw.columns]
        print(pw[disp].to_string(index=False))

    return tab_kw


# ── Table D: PA method comparison ─────────────────────────────────────────────

def table_d_pa_method():
    fold = load(OUT5 / "experiment_fold_metrics.csv")
    if len(fold) == 0:
        print("  Table D: no fold metrics"); return None

    pa_methods = ["random", "spatial_constrained", "two_step", "uncertainty_zone"]
    rows = []
    for pm in pa_methods:
        sub = fold[fold["pa_method"] == pm]
        for met in ["auc_roc", "pr_auc", "tss", "brier"]:
            v = sub[met].dropna()
            if len(v) > 0:
                rows.append({
                    "PA Method": pm.replace("_", " ").title(),
                    "Metric": met.upper().replace("_", "-"),
                    "Mean": round(v.mean(), 4),
                    "SD": round(v.std(), 4),
                    "N": len(v),
                })

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT_TAB / "tableD_pa_method.csv", index=False)

    # Pivot for AUC only
    auc_tab = tab[tab["Metric"] == "AUC-ROC"].copy()
    auc_tab["Mean (SD)"] = auc_tab.apply(lambda r: f"{r['Mean']:.3f} ({r['SD']:.3f})", axis=1)
    print(f"  Table D (PA method AUC):")
    print(auc_tab[["PA Method", "Mean (SD)"]].to_string(index=False))
    return tab


# ── Table E: Sensitivity analysis ─────────────────────────────────────────────

def table_e_sensitivity():
    sens = load(OUT5 / "sensitivity" / "sensitivity_summary.csv")
    if len(sens) == 0:
        print("  Table E: sensitivity_summary.csv not found"); return None

    # Format for display
    rows = []
    for _, r in sens.iterrows():
        row = {
            "Sensitivity Factor": str(r["sensitivity_factor"]).replace("_", " "),
            "Level": str(r["value"]),
            "N Folds": int(r.get("n_folds", 0)),
            "AUC Mean": f"{r['auc_mean']:.3f}",
            "AUC SD": f"{r['auc_sd']:.3f}",
            "PR-AUC Mean": f"{r['prauc_mean']:.3f}",
        }
        if "n_occ" in r and pd.notna(r["n_occ"]):
            row["N Occurrences"] = int(r["n_occ"])
        rows.append(row)

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT_TAB / "tableE_sensitivity.csv", index=False)
    print(f"  Table E (Sensitivity analysis):")
    print(tab.to_string(index=False))
    return tab


# ── Table F: Province risk ─────────────────────────────────────────────────────

def table_f_province_risk():
    prov = load(OUT3 / "validation_v2" / "province_risk.csv")
    if len(prov) == 0:
        prov = load(OUT3 / "validation" / "province_risk.csv")
    if len(prov) == 0:
        print("  Table F: province_risk.csv not found"); return None

    prov.to_csv(OUT_TAB / "tableF_province_risk.csv", index=False)
    print(f"  Table F (Province risk, top 15):")
    cols = [c for c in ["rank","province","mean_suit","max_suit","pct_high","n_cells","sd_suit"]
            if c in prov.columns]
    print(prov.head(15)[cols].to_string(index=False))
    return prov


# ── Summary statistics text ───────────────────────────────────────────────────

def write_summary_stats():
    pool = load(OUT5 / "pooled_oof_metrics.csv")
    fold = load(OUT5 / "experiment_fold_metrics.csv")

    lines = ["# Paper 1 Summary Statistics\n"]

    if len(pool) > 0:
        auc_col = "auc_roc" if "auc_roc" in pool.columns else "auc_roc_mean"
        if auc_col in pool.columns:
            best = pool.sort_values(auc_col, ascending=False).iloc[0]
            ci_lo = best.get("auc_roc_ci_lo", "?")
            ci_hi = best.get("auc_roc_ci_hi", "?")
            lines.append(f"Best configuration: {best['algo']}/{best['feat_set']}/{best['pa_method']}")
            lines.append(f"  Pooled OOF AUC: {best[auc_col]:.3f} [95% CI: {ci_lo}–{ci_hi}]")
            if "pr_auc" in best and pd.notna(best["pr_auc"]):
                lines.append(f"  PR-AUC: {best['pr_auc']:.3f}")
            if "tss" in best and pd.notna(best["tss"]):
                lines.append(f"  TSS: {best['tss']:.3f}")
            if "brier" in best and pd.notna(best["brier"]):
                lines.append(f"  Brier: {best['brier']:.3f}")
            lines.append("")

    if len(fold) > 0:
        lines.append("Fold-level mean AUC by algorithm:")
        for algo in ["RF", "XGB", "MXN"]:
            v = fold[fold["algo"] == algo]["auc_roc"].dropna()
            if len(v) > 0:
                lines.append(f"  {algo}: {v.mean():.3f} +/- {v.std():.3f}")
        lines.append("")
        lines.append("Fold-level mean AUC by feature set:")
        for fs in ["M_S", "M_SA", "M_SP", "M_SAP"]:
            v = fold[fold["feat_set"] == fs]["auc_roc"].dropna()
            if len(v) > 0:
                lines.append(f"  {fs}: {v.mean():.3f} +/- {v.std():.3f}")

    txt = "\n".join(lines)
    (OUT_TAB / "summary_stats.txt").write_text(txt, encoding="utf-8")
    print(f"\n  Summary stats:")
    print(txt)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Step 24: Paper 1 Tables")
    print("=" * 65)

    print("\n--- Table A: AUC Matrix (algo x feat_set) ---")
    table_a_auc_matrix()

    print("\n--- Table B: Best Configuration per Algorithm ---")
    table_b_best_config()

    print("\n--- Table C: Statistical Comparison ---")
    table_c_stats()

    print("\n--- Table D: PA Method Comparison ---")
    table_d_pa_method()

    print("\n--- Table E: Sensitivity Analysis ---")
    table_e_sensitivity()

    print("\n--- Table F: Province Risk ---")
    table_f_province_risk()

    print("\n--- Summary Statistics ---")
    write_summary_stats()

    files = list(OUT_TAB.glob("*.csv")) + list(OUT_TAB.glob("*.txt"))
    print(f"\n  All tables: {OUT_TAB} ({len(files)} files)")
    for f in sorted(files):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
