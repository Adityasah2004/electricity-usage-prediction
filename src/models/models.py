import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def train_arima(train_data, order=(1, 1, 1)):
    """Train an ARIMA model."""
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    return model_fit

def train_sarima(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24)):
    """Train a SARIMA model."""
    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    return model_fit

def train_prophet(train_data):
    """Train a Prophet model."""
    # Prepare data for Prophet
    df = train_data.reset_index()
    df.columns = ['ds', 'y']
    
    # Initialize and train model
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df)
    
    return model

def train_lstm(X_train, y_train, input_shape):
    """Train an LSTM model."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(y_train.shape[1])
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    return model

def evaluate_model(y_true, y_pred):
    """Evaluate model performance."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }
