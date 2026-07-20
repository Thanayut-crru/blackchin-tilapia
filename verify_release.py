from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def rows(rel: str):
    with (ROOT / rel).open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def close(actual, expected, tol=5e-4):
    if not math.isclose(float(actual), expected, rel_tol=0, abs_tol=tol):
        raise AssertionError(f"expected {expected}, got {actual}")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_metrics():
    pooled = rows("blackchin_tilapia_analysis/output/pooled_best_per_model.csv")
    best = next(r for r in pooled if r["algo"] == "RF" and r["feat_set"] == "M_SAP" and r["pa_method"] == "uncertainty_zone")
    for key, value in {"auc_roc": .8520, "pr_auc": .2185, "brier": .0902, "cal_slope": 1.4678,
                       "tss": .5410, "tss_thr": .3939, "sensitivity": .7000, "specificity": .8410}.items():
        close(best[key], value)

    context = rows("revision_outputs/selected_performance_context.csv")[0]
    for key, value in {"random_pr_auc_expectation": .0307294194, "fold_auc_min": .6996153846,
                       "fold_auc_max": .9975, "fold_auc_sd": .0832138079}.items():
        close(context[key], value, 1e-8)

    grouped = {r["group"]: r for r in rows("revision_outputs/grouped_permutation_summary.csv")}
    close(grouped["S_environment"]["mean_grouped_auc_decrease"], .2018583998, 1e-8)
    close(grouped["P_human_activity"]["mean_grouped_auc_decrease"], .0511401389, 1e-8)
    close(grouped["A_waterway_context"]["mean_grouped_auc_decrease"], .0143661378, 1e-8)
    assert float(grouped["A_waterway_context"]["ci_lo"]) < 0 < float(grouped["A_waterway_context"]["ci_hi"])

    moran = rows("revision_outputs/residual_morans_i.csv")[0]
    close(moran["morans_i"], .1226080053, 1e-8)
    close(moran["permutation_p_two_sided"], .0087, 1e-8)

    aoa = rows("revision_outputs/area_of_applicability_summary.csv")[0]
    close(aoa["fraction_cells_supported_by_all_15_models"], .0239590218, 1e-8)
    close(aoa["fraction_cells_supported_by_fewer_than_8_models"], .5156972902, 1e-8)

    pop = rows("revision_outputs/population_density_observation_diagnostic.csv")[0]
    assert int(pop["n_occurrence_cells"]) == 38
    assert int(pop["n_aquatic_population_cells"]) == 17374
    close(pop["occurrence_median_people_km2"], 277.2841763, 1e-6)
    close(pop["population_density_only_auc"], .9145380575, 1e-8)
    assert int(pop["occurrences_above_background_q75"]) == 33
    print("PASS: manuscript-facing numerical claims match archived outputs.")


def verify_checksums():
    failures = []
    with (ROOT / "SHA256SUMS.txt").open(encoding="utf-8") as fh:
        for line in fh:
            expected, rel = line.rstrip("\n").split("  ", 1)
            path = ROOT / rel
            if not path.exists() or digest(path) != expected:
                failures.append(rel)
    if failures:
        raise AssertionError("Checksum failures: " + ", ".join(failures))
    print("PASS: release checksums match.")


parser = argparse.ArgumentParser()
parser.add_argument("--skip-checksums", action="store_true")
args = parser.parse_args()
verify_metrics()
if not args.skip_checksums:
    verify_checksums()
print("RELEASE VERIFICATION PASSED")
