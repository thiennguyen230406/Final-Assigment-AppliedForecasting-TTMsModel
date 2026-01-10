# Time Series Forecasting with Foundation Models: IBM TTM

## 1. Project Overview
This project investigates the performance of **IBM Tiny Time Mixers (TTM)**, a modern foundation model for time series forecasting. The model is evaluated in a **zero-shot** setting and compared against strong statistical (AutoETS, Seasonal Naive) and machine learning (LightGBM) baselines.

The evaluation covers two distinct domains to test robustness:
1.  **Finance:** Apple Stock Price (AAPL) - Daily frequency (Non-stationary, trend-heavy).
2.  **Energy:** PJM East Electricity Load - Hourly frequency (Strong seasonality).

## 2. Directory Structure
The repository is organized as follows (per assignment requirements):

```
.
├── configs/             # Configuration files (hyperparameters, horizon, lags)
├── notebooks/           # Jupyter notebooks for EDA, experiments, and visualization
├── src/                 # Modular source code
│   ├── data_loader.py   # Scripts to download and preprocess Yahoo/PJM data
│   ├── models.py        # Implementation of TTM, AutoETS, and LightGBM
│   └── main.py          # Entry point for the end-to-end experiment
├── requirements.txt     # Python dependencies
├── run_end_to_end.sh    # Bash script to reproduce results
└── README.md            # Project documentation
```

## 3. Installation & Requirements
**Note:** This project requires a specific version of `transformers` to avoid conflicts with `torch` and `flex_attention`.

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd <your-repo-name>
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment (conda or venv).
    ```bash
    pip install -r requirements.txt
    ```
    *Critical:* If you encounter import errors regarding `flex_attention` or `tsfm`, ensure you have strictly installed `transformers==4.46.3` as specified in `requirements.txt`.

3.  **IBM TTM Setup:**
    The code attempts to import `tsfm_public`. If running locally without the library installed via pip, you may need to clone the IBM repo:
    ```bash
    git clone https://github.com/IBM/tsfm.git
    cd tsfm && pip install .
    ```

## 4. Usage & Reproduction
To run the full experiment pipeline (Data Load -> Training -> Inference -> Evaluation):

**Option 1: Shell Script (Linux/Mac/Colab)**
```bash
bash run_end_to_end.sh
```

**Option 2: Python Direct**
```bash
cd src
python main.py
```

**Option 3: Notebook**
Open `notebooks/analysis.ipynb` in Jupyter Lab or Google Colab to see the interactive analysis, plots, and error breakdowns.

## 5. Model Details & Configuration

### Foundation Model
*   **Model:** `ibm/TTM` (Tiny Time Mixers).
*   **Mode:** Zero-shot (Direct inference without fine-tuning).
*   **Context Length:** 512 time steps.

### Baselines
*   **Seasonal Naive:** Repeats values from the last seasonal period ($m=7$ for Daily, $m=24$ for Hourly).
*   **AutoETS:** State-space model with automatic selection of Error, Trend, and Seasonality.
*   **LightGBM:** Gradient Boosting Regressor.
    *   *Features:* Lags, Rolling Means, Calendar features (Month, Day of Week, Hour).

### Hyperparameters
All settings are defined in `configs/config.yaml`.
*   **Seed:** 42 (Fixed for reproducibility).
*   **Validation:** Rolling-origin cross-validation (3 folds).
*   **Forecast Horizon ($H$):**
    *   Finance: 30 days.
    *   Energy: 48 hours.

## 6. Data Sources
*   **Finance:** Yahoo Finance API (`yfinance`). Data is resampled to Daily frequency.
*   **Energy:** PJM Hourly Energy Consumption (`PJME_hourly.csv`). Data is sourced from Rob Mulla's public mirror of PJM Interconnection data.

## 7. Expected Runtime & Hardware
*   **Hardware:** Optimized for T4 GPU (Google Colab) but runs efficiently on CPU.
*   **Runtime:** Approximately 2-5 minutes for the end-to-end pipeline (Data loading + Inference for both datasets).

## 8. License
*   Code is provided for educational purposes as part of the Final Assignment.
*   IBM TTM weights are subject to the Apache 2.0 License.
*   Yahoo Finance and PJM data are public datasets used for research/educational use.
