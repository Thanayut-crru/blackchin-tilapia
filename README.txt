# Associations of Environmental, Waterway-Context, and Human-Activity Predictors with Blackchin Tilapia Records in Thailand

Version **1.1.0** is the Zenodo-ready computational companion to the
associated unpublished article. The manuscript itself is not included. This
archive contains the exact analysis code, the 38-row modelling occurrence table,
redistributable/share-alike model-ready inputs, archived out-of-fold predictions,
revision diagnostics, figures, and machine-readable checks.

The national surface is a hypothesis-generating screening product. It is not a
validated occurrence probability, occupancy estimate, invasion forecast, or
operational risk map.

## Fast verification

Python 3.11 or later is recommended.

```powershell
python verify_release.py
python pre_upload_check.py
```

The first command verifies the manuscript-facing numerical claims and all file
checksums. The second intentionally fails until the repository URL, data DOI,
and reserved Zenodo DOI are completed before publication.

## Reproduction levels

1. **Archived-result verification (minutes, no third-party download):** run
   `python verify_release.py`.
2. **Environmental input reconstruction:** run
   `python prepare_worldclim.py --download`, then
   `python verify_inputs.py`. WorldClim files are fetched directly because the
   provider does not permit redistribution.
3. **Full factorial experiment (long-running):** create the pinned environment,
   prepare WorldClim, then run
   `python run_reproduction.py --stage full`.
4. **Revision diagnostics:** see `REPRODUCIBILITY.txt`; national AOA and province
   regeneration additionally requires a user-obtained GADM boundary.

## Main reported configuration

- Algorithms: RF, XGB, and MaxEnt-like logistic model
- Predictor sets: M_S, M_SA, M_SP, and M_SAP
- Pseudo-absence methods: random, spatial-constrained, two-step, uncertainty-zone
- Outer design: 5 repetitions x 3 spatial folds
- Uncertainty-zone interval: 0.20 through 0.60, inclusive
- Selected result: RF/M_SAP/uncertainty-zone, conditional pooled AUC 0.852

Read `REPRODUCIBILITY.txt`, `THIRD_PARTY_DATA.txt`, and `PRE_UPLOAD_CHECKLIST.txt`
before depositing the ZIP. Third-party data remain governed by their source
terms; the repository-level MIT licence applies to original code only.
