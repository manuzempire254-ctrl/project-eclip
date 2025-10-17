# model.py
# ---------------------------------------
# Defines the CNN-LSTM hybrid model for Project Eclipse

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from config import IMG_SIZE, SEQ_LENGTH

def build_cnn_lstm_model(input_shape):
    """
    CNN-LSTM hybrid model
    CNN extracts short-term patterns (like candlestick formations)
    LSTM captures long-term sequences (like trend memory)
    """
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        LSTM(64, return_sequences=True),
        Dropout(0.3),

        LSTM(32),
        Dense(64, activation='relu'),
        Dropout(0.3),

        Dense(1, activation='linear')  # Predict next price
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    print(model.summary())
    return model

if __name__ == "__main__":
    # Example shape for testing: (60 timesteps, 5 features)
    example_input_shape = (SEQ_LENGTH, 5)
    model = build_cnn_lstm_model(example_input_shape)
