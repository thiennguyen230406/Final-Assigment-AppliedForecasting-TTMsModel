# Time Series Forecasting Assignment
**Foundation Model Analysis: IBM TTM (Tiny Time Mixers)**

## Overview
This repository compares a Foundation Model (TTM) against statistical (AutoETS) and ML (LightGBM) baselines on Energy and Finance datasets.

## Structure
- `src/`: Source code for data loading, modeling, and execution.
- `configs/`: YAML configuration files.
- `run_end_to_end.sh`: Bash script to reproduce results.

## Reproduction
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # Note: IBM tsfm must be installed separately or via git clone if using TTM specific classes
   ```
