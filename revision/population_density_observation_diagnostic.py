"""Reproduce the population-density observation-process diagnostic for Paper 1."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.stats import pointbiserialr
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=Path, required=True, help="Path to blackchin_tilapia_analysis")
parser.add_argument("--occurrences", type=Path, required=True, help="Audited 38-record CSV")
parser.add_argument("--output", type=Path, required=True, help="Output summary CSV")
args = parser.parse_args()

project = args.project
occ_path = args.occurrences
out_path = args.output
sys.path.insert(0, str(project.parent))
model = importlib.import_module("blackchin_tilapia_analysis.scripts.20_full_experiment_v2")

arrays, transform, _ = model.load_all_rasters()
with rasterio.open(model.WATER_MASK) as src:
    aquatic_mask = src.read(1) == 1
background = arrays["P1_pop_density"][aquatic_mask]
background = background[np.isfinite(background)]

occ = pd.read_csv(occ_path)
occurrence = model.sample_at_coords(
    arrays,
    ["P1_pop_density"],
    occ["longitude"].to_numpy(),
    occ["latitude"].to_numpy(),
    transform,
)[:, 0]
occurrence = occurrence[np.isfinite(occurrence)]

y = np.concatenate([np.ones(len(occurrence)), np.zeros(len(background))])
x = np.concatenate([occurrence, background])
auc = roc_auc_score(y, x)
r, p = pointbiserialr(y, np.log1p(x))
q75 = np.quantile(background, 0.75)
n_above = int(np.count_nonzero(occurrence > q75))

summary = {
    "n_occurrence_cells": int(len(occurrence)),
    "n_aquatic_population_cells": int(len(background)),
    "occurrence_median_people_km2": float(np.median(occurrence)),
    "occurrence_q1_people_km2": float(np.quantile(occurrence, 0.25)),
    "occurrence_q3_people_km2": float(np.quantile(occurrence, 0.75)),
    "background_median_people_km2": float(np.median(background)),
    "background_q1_people_km2": float(np.quantile(background, 0.25)),
    "background_q3_people_km2": float(np.quantile(background, 0.75)),
    "population_density_only_auc": float(auc),
    "background_q75_people_km2": float(q75),
    "occurrences_above_background_q75": n_above,
    "occurrences_above_background_q75_pct": float(100 * n_above / len(occurrence)),
    "point_biserial_r_log1p": float(r),
    "point_biserial_nominal_p": float(p),
    "background_definition": (
        "All water-mask cells with a finite post-processed population-density value; "
        "not the fixed outer-evaluation backgrounds and not cells complete for every predictor."
    ),
    "interpretation": (
        "Descriptive diagnostic only; aquatic population-density cells are spatially dependent, "
        "and the nominal p-value is not used for spatial inference."
    ),
}
out_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame([summary]).to_csv(out_path, index=False)
print(json.dumps(summary, indent=2))
