import sys
import os
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from datetime import datetime, timedelta

# Add the project root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.preprocess import load_data, add_features, scale_data, create_sequences

def load_model(model_type):
    """Load a trained model."""
    if model_type.lower() == 'lstm':
        model = tf.keras.models.load_model('models/lstm_model.h5')
    else:
        with open(f'models/{model_type.lower()}_model.pkl', 'rb') as f:
            model = pickle.load(f)
    return model

def predict_arima(model, steps=24):
    """Generate predictions using ARIMA model."""
    forecast = model.forecast(steps=steps)
    return forecast

def predict_sarima(model, steps=24):
    """Generate predictions using SARIMA model."""
    forecast = model.forecast(steps=steps)
    return forecast

def predict_prophet(model, steps=24):
    """Generate predictions using Prophet model."""
    future = model.make_future_dataframe(periods=steps, freq='H')
    forecast = model.predict(future)
    return forecast.tail(steps)['yhat']

def predict_lstm(model, last_data, scaler, feature_columns, sequence_length=24, steps=24):
    """Generate predictions using LSTM model."""
    # Scale the data
    last_data_scaled = scaler.transform(last_data[feature_columns].values)
    
    # Reshape for LSTM input
    X = last_data_scaled.reshape(1, sequence_length, len(feature_columns))
    
    # Predict
    predictions = model.predict(X)
    
    # Reshape predictions
    predictions = predictions.reshape(-1)
    
    # Inverse transform to get original scale
    predictions_df = pd.DataFrame(np.zeros((len(predictions), len(feature_columns))), 
                                 columns=feature_columns)
    predictions_df['consumption'] = predictions
    
    predictions_original = scaler.inverse_transform(predictions_df)[:, 0]
    
    return predictions_original

def main():
    # Load the latest data
    print("Loading latest data...")
    df = load_data('data/processed/train_data.csv')
    
    # Get the last timestamp
    last_timestamp = df.index[-1]
    
    # Generate future timestamps
    future_timestamps = [last_timestamp + timedelta(hours=i+1) for i in range(24)]
    
    # Load models
    print("Loading models...")
    arima_model = load_model('arima')
    sarima_model = load_model('sarima')
    prophet_model = load_model('prophet')
    lstm_model = load_model('lstm')
    
    # Generate predictions
    print("Generating predictions...")
    
    # ARIMA predictions
    arima_forecast = predict_arima(arima_model, steps=24)
    arima_forecast = pd.Series(arima_forecast, index=future_timestamps)
    
    # SARIMA predictions
    sarima_forecast = predict_sarima(sarima_model, steps=24)
    sarima_forecast = pd.Series(sarima_forecast, index=future_timestamps)
    
    # Prophet predictions
    prophet_forecast = predict_prophet(prophet_model, steps=24)
    prophet_forecast = pd.Series(prophet_forecast.values, index=future_timestamps)
    
    # LSTM predictions
    # For LSTM, we need the last sequence_length data points
    sequence_length = 24
    last_data = df.tail(sequence_length)
    
    # Add features
    last_data = add_features(last_data)
    
    # Define feature columns
    feature_columns = ['consumption', 'hour', 'dayofweek', 'month', 'consumption_lag_1d', 
                       'consumption_rolling_mean_24h', 'consumption_rolling_std_24h']
    
    # Scale data
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    
    # Generate LSTM predictions
    lstm_forecast = predict_lstm(lstm_model, last_data, scaler, feature_columns, 
                                sequence_length, steps=24)
    lstm_forecast = pd.Series(lstm_forecast, index=future_timestamps)
    
    # Combine all forecasts
    forecasts = pd.DataFrame({
        'ARIMA': arima_forecast,
        'SARIMA': sarima_forecast,
        'Prophet': prophet_forecast,
        'LSTM': lstm_forecast
    })
    
    # Save forecasts
    forecasts.to_csv('results/forecasts.csv')
    
    print("Predictions generated successfully!")
    print(forecasts)

if __name__ == "__main__":
    main()
