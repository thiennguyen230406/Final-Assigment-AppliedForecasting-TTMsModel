
import torch
import numpy as np
from statsforecast.models import SeasonalNaive, AutoETS
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean # <--- CRITICAL IMPORT
from lightgbm import LGBMRegressor
from transformers import AutoModelForPrediction

# Try importing TTM specific class, fallback to AutoModel
try:
    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
except ImportError:
    pass

def get_statistical_baselines(season_length):
    return [
        SeasonalNaive(season_length=season_length),
        AutoETS(model='ZZZ', season_length=season_length)
    ]

def get_ml_baseline(config):
    # Determine date features based on frequency
    freq = config['data']['freq'].lower()
    if 'h' in freq:
        date_feats = ['hour', 'dayofweek', 'month']
    else:
        date_feats = ['month', 'dayofweek']

    return MLForecast(
        models=[LGBMRegressor(verbosity=-1, random_state=config['experiment']['seed'])],
        freq=config['data']['freq'],
        lags=config['models']['lgbm_lags'],
        # <--- FIX: Using Class instead of string for RollingMean
        lag_transforms={
            1: [RollingMean(window_size=config['models']['lgbm_rolling_window'])]
        },
        date_features=date_feats
    )

class TTMWrapper:
    def __init__(self, config, device):
        self.device = device
        self.context_len = config['models']['ttm_context']
        model_id = config['models']['ttm_id']
        try:
            self.model = TinyTimeMixerForPrediction.from_pretrained(model_id, revision="main")
        except:
            # Fallback
            self.model = AutoModelForPrediction.from_pretrained(model_id, revision="main", trust_remote_code=True)
        self.model.to(device)
        self.model.eval()

    def predict(self, history_series, horizon):
        input_data = history_series.values
        # Padding logic
        if len(input_data) < self.context_len:
            pad_len = self.context_len - len(input_data)
            input_data = np.pad(input_data, (pad_len, 0), mode='edge')
        else:
            input_data = input_data[-self.context_len:]

        # Normalize
        loc = input_data.mean()
        scale = input_data.std() + 1e-6
        input_data = (input_data - loc) / scale

        batch = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(self.device)
        with torch.no_grad():
            output = self.model(past_values=batch)
            pred = output.prediction_outputs.cpu().numpy().squeeze()

        # De-normalize
        return (pred * scale) + loc
