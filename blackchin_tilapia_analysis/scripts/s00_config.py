"""Paper 1 configuration."""

from pathlib import Path

PAPER1_ROOT  = Path(__file__).parent.parent
PROJECT_ROOT = PAPER1_ROOT.parent
BASE    = PROJECT_ROOT
DATA    = PROJECT_ROOT / "data"
ENV_DIR = DATA / "environmental" / "clipped"
WATER   = DATA / "water" / "water_mask_envgrid.tif"
OCC_CSV = BASE / "audit" / "output_audit" / "occurrence_merged_lit.csv"
OUT3    = PROJECT_ROOT / "output3"

# v2 experiment output (leakage-free, year-filtered, spatial inner CV)
OUT5    = PAPER1_ROOT / "output"

# A-layer directory (hydrological accessibility)
A_DIR   = DATA / "hydrological"

# P-layer directory (propagule pressure)
P_DIR   = DATA / "propagule"

# Aligned feature stack (output of step 12)
FEAT_MANIFEST = OUT3 / "aligned_features" / "feature_manifest.csv"
FEAT_STACK    = OUT3 / "aligned_features" / "feature_stack.npz"

THAILAND_BBOX  = dict(xmin=97.3, xmax=105.7, ymin=5.5, ymax=20.6)
TARGET_SPECIES = "Sarotherodon melanotheron"

COORD_UNCERTAINTY_MAX_M = 5000
THIN_DIST_KM            = 10.0
THIN_REPS               = 20
CELL_DEG                = 0.0417

N_OUTER_FOLDS   = 3
N_OUTER_REPS    = 5
N_INNER_FOLDS   = 3
N_HP_CANDIDATES = 10

N_PA           = 200
PA_BUFFER_KM   = 10.0
SUITABILITY_LO = 0.20
FUZZY_THRESHOLD= 0.50
TOP_FUZZY_VARS = 3

RANDOM_STATE   = 42
N_TREES_RF     = 200
N_TREES_PILOT  = 100

SUITABILITY_THRESHOLD = 0.5
