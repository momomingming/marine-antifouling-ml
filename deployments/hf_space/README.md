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

# Marine Antifouling Material ML Prediction Platform v5.0

Predict antifouling performance from SMILES molecular structure.

## Features
- **Single Prediction**: Input SMILES → multi-model prediction with radar plot
- **Batch Compare**: Compare multiple materials side-by-side
- **Literature DB**: Search 305 antifouling papers
- **Models**: Optuna-XGBoost, LightGBM, KNN, Ridge

## Performance (Blind Test R²)
| Target | R² |
|--------|-----|
| Antifouling Efficiency | 0.973 |
| Fouling Release | 0.979 |
| Antibacterial Rate | 0.994 |
| Diatom Removal | 0.977 |

## Dataset
- 5000 samples (238 original materials + 4762 augmented)
- 42 features (28 molecular descriptors + 14 interaction features)
- 8 material categories
