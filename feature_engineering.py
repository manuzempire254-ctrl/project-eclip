# feature_engineering.py
# -----------------------------------------
# Creates engineered features for Project Eclipse
# Focus: ICT market structure setups (AMD, order blocks, market breaks)

import pandas as pd
import numpy as np
from config import SEQ_LENGTH
from data_pipeline import load_or_download

def detect_market_structure(df):
    """Detect simple swing highs and lows."""
    df['Swing_High'] = (df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(-1))
    df['Swing_Low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(-1))
    return df

def detect_break_of_structure(df):
    """Detect break of structure (BoS)."""
    df['BoS'] = np.where((df['High'] > df['High'].shift(3)) & (df['Swing_High']), 1,
                 np.where((df['Low'] < df['Low'].shift(3)) & (df['Swing_Low']), -1, 0))
    return df

def detect_order_blocks(df):
    """Simple bullish/bearish order block detection."""
    df['Bullish_OB'] = (df['BoS'] == 1) & (df['Low'] == df['Low'].rolling(window=5).min())
    df['Bearish_OB'] = (df['BoS'] == -1) & (df['High'] == df['High'].rolling(window=5).max())
    return df

def label_trading_phases(df):
    """Label Accumulation, Manipulation, and Distribution phases."""
    df['Phase'] = 'Accumulation'
    df.loc[df['BoS'] == 1, 'Phase'] = 'Manipulation'
    df.loc[df['BoS'] == -1, 'Phase'] = 'Distribution'
    return df

def sequence_data(df):
    """Convert time series into supervised learning sequences."""
    X, y = [], []
    for i in range(SEQ_LENGTH, len(df)):
        X.append(df.iloc[i-SEQ_LENGTH:i][['Open','High','Low','Close','Volume']].values)
        y.append(df['Close'].iloc[i])
    return np.array(X), np.array(y)

def generate_features():
    """Run all feature engineering steps."""
    df = load_or_download()
    df = detect_market_structure(df)
    df = detect_break_of_structure(df)
    df = detect_order_blocks(df)
    df = label_trading_phases(df)
    X, y = sequence_data(df)
    print(f"Feature set created: {X.shape}, Labels: {y.shape}")
    return X, y

if __name__ == "__main__":
    X, y = generate_features()
