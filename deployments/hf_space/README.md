---
title: Marine Antifouling ML Predictor
emoji: 🌊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Marine Antifouling Material ML Prediction Platform v6.0

Predict antifouling performance from SMILES molecular structure.

## Features
- **Single Prediction**: Input SMILES → multi-model prediction with radar plot
- **Uncertainty & Applicability Domain (P0)**: every prediction returns a 95% confidence
  interval plus an in-domain / out-of-domain verdict — never a bare point value
- **Batch Compare**: Compare multiple materials side-by-side
- **Literature DB**: search 659 antifouling papers
- **Models**: Optuna-XGBoost, LightGBM, KNN, Ridge ensemble

## Performance — honest protocol
Grouped out-of-fold (Leave-One-Group-Out by SMILES), pooled R²:

| Target | Pooled out-of-fold R² | MAE |
|--------|----------------------|-----|
| Antifouling Efficiency | 0.592 | 4.46 |

> ⚠️ **Correction of an earlier claim.** A previously advertised "blind-test R² 0.97"
> came from a **random split that leaked information**: augmented variants of the same
> molecule landed in both train and test sets. It has been withdrawn. The current
> figure uses LOGO-by-SMILES (84 molecular groups, no group spans two folds) with
> pooled out-of-fold R²; random splitting was inflated by roughly 0.11 (0.704 → 0.592).
> The main bottleneck is that only **84 truly independent molecules** exist
> (≈58% of rows are augmented samples).

## Dataset
- 1158 samples (486 original materials + 672 augmented, 84 unique SMILES)
- 42 features (28 molecular descriptors + 14 interaction features)
- 8 material categories

## Deployment note — ensemble assets
`uncertainty.py` locates the out-of-fold ensemble by probing, in order:
`$MAF_ENSEMBLE_DIR` → `<root>/models/logo_ensemble` → `../models/logo_ensemble` → `./models/logo_ensemble`.

When publishing this folder as a Hugging Face Space, copy `models/logo_ensemble/`
(84 × `fold_*.ubj` + `meta.pkl`, ≈26 MB) into the Space root so the last candidate
path resolves. **Without it the app still runs** — the confidence block silently
degrades to the original single-point prediction instead of crashing.
