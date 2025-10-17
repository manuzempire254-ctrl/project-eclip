# config.py
# ------------------------
# Project Eclipse Configuration File

import os

# Project Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create folders if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Data Source (Yahoo Finance default)
SYMBOL = "XAUUSD=X"  # Gold Pair
TIMEFRAME = "1d"
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"

# Model Parameters
IMG_SIZE = (64, 64)
SEQ_LENGTH = 60
BATCH_SIZE = 32
EPOCHS = 20

# Logging
LOG_FILE = os.path.join(BASE_DIR, "eclipse_log.txt")
