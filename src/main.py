
import yaml
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from statsforecast import StatsForecast
from data_loader import load_data
from models import get_statistical_baselines, get_ml_baseline, TTMWrapper

def main():
    # 1. Load Config
    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print(f"Running Experiment: {config['data']['dataset_type'].upper()}")

    # 2. Load Data
    series = load_data(config)
    horizon = config['experiment']['horizon']

    # Split Train/Test
    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]
    print(f"Train size: {len(train)}, Test size: {len(test)}")

    # 3. Statistical Baselines
    print("Training Statistical Baselines...")
    sf = StatsForecast(
        models=get_statistical_baselines(config['experiment']['seasonal_period']),
        freq=config['data']['freq'],
        n_jobs=-1
    )
    sf.fit(train)
    preds = sf.predict(h=horizon)

    # 4. ML Baseline
    print("Training ML Baseline (LightGBM)...")
    ml_model = get_ml_baseline(config)
    ml_model.fit(train)
    ml_preds = ml_model.predict(h=horizon)

    # Merge
    preds = preds.merge(ml_preds, on=['unique_id', 'ds'], how='left')

    # 5. TTM Inference
    print("Running TTM Inference...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ttm = TTMWrapper(config, device)
    # TTM usually predicts 96 steps, we truncate to horizon
    ttm_raw = ttm.predict(train['y'], horizon)
    preds['TTM'] = ttm_raw[:horizon]

    # 6. Evaluation
    y_true = test['y'].values
    metrics = {}
    for model in ['SeasonalNaive', 'AutoETS', 'LGBMRegressor', 'TTM']:
        mae = mean_absolute_error(y_true, preds[model])
        metrics[model] = mae

    print("\n=== FINAL RESULTS (MAE) ===")
    print(metrics)

    # Save Results
    preds.to_csv("forecasts.csv")
    print("\nForecasts saved to forecasts.csv")

if __name__ == "__main__":
    main()
