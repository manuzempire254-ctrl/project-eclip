# data_pipeline.py
# ------------------------
# Downloads and cleans forex data for Project Eclipse

import pandas as pd
import numpy as np
import yfinance as yf
from config import SYMBOL, START_DATE, END_DATE, DATA_DIR
import os

def download_data():
    """Download market data from Yahoo Finance."""
    print(f"Downloading {SYMBOL} data from {START_DATE} to {END_DATE}...")
    data = yf.download(SYMBOL, start=START_DATE, end=END_DATE)
    file_path = os.path.join(DATA_DIR, f"{SYMBOL.replace('=','')}.csv")
    data.to_csv(file_path)
    print(f"Data saved to {file_path}")
    return data

def clean_data(df):
    """Basic data cleaning and feature generation."""
    df = df.dropna()
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=10).std()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df = df.dropna()
    return df

def load_or_download():
    """Check if data exists, else download new one."""
    file_path = os.path.join(DATA_DIR, f"{SYMBOL.replace('=','')}.csv")
    if os.path.exists(file_path):
        print("Loading existing data...")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        df = download_data()
    return clean_data(df)

if __name__ == "__main__":
    data = load_or_download()
    print(data.tail())
