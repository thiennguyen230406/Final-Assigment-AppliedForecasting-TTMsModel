
import yfinance as yf
import pandas as pd
import numpy as np

def load_data(config):
    """Loads and preprocesses data based on config."""
    dtype = config['data']['dataset_type']
    freq = config['data']['freq']

    if dtype == 'finance':
        print(f"Loading Finance data for {config['data']['ticker']}...")
        df = yf.download(
            config['data']['ticker'],
            start=config['data']['start_date'],
            end=config['data']['end_date']
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.resample(freq).mean().ffill()
        series = df['Close'].reset_index()
        series.columns = ['ds', 'y']
        series['unique_id'] = config['data']['ticker']

    elif dtype == 'energy':
        print(f"Loading Energy data from URL...")
        df = pd.read_csv(config['data']['url'])
        df = df.rename(columns={'Datetime': 'ds', 'PJME_MW': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').set_index('ds')

        # Resample and Interpolate
        df = df.resample(freq).mean()
        df['y'] = df['y'].interpolate(method='linear')

        series = df.reset_index()
        # Filter to reasonable size if needed (e.g. last 2 years)
        series = series.tail(24 * 365 * 2).copy()
        series['unique_id'] = "PJME_Electricity"

    return series
