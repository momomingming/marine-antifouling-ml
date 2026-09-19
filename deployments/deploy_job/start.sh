#!/bin/bash
set -e

echo "=== [$(date)] 安装缺失依赖 ==="
pip install --no-cache-dir -q rdkit-pypi xgboost lightgbm shap optuna 2>&1 | tail -5

echo "=== [$(date)] 启动 Gradio v6.0 平台 (port 7860) ==="
python -u app.py 2>&1
