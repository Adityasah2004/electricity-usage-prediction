import sys
import os
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from datetime import datetime, timedelta

# Add the project root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.preprocess import load_data, add_features

def load_model(model_type):
    """Load a trained model."""
    if model_type.lower() == 'lstm':
        model = tf.keras.models.load_model(
            'models/lstm_model.h5',
            custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
        )
    else:
        model_path = f'models/{model_type.lower()}_model.pkl'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model file: {model_path}. Run train.py to generate it.")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    return model

def predict_arima(model, steps=24):
    """Generate predictions using ARIMA model."""
    return model.forecast(steps=steps)

def predict_sarima(model, steps=24):
    """Generate predictions using SARIMA model."""
    return model.forecast(steps=steps)

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
    """Main function to generate forecasts using different models."""
    
    # Check if processed data file exists
    data_path = 'data/processed/train_data.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing processed data: {data_path}. Run train.py to generate it.")

    # Load the latest data
    print("Loading latest data...")
    df = load_data(data_path)

    # Ensure data is not empty
    if df.empty:
        raise ValueError("Error: Loaded dataset is empty. Ensure train.py runs correctly.")

    # Get the last timestamp
    last_timestamp = df.index[-1]

    # Generate future timestamps
    future_timestamps = [last_timestamp + timedelta(hours=i + 1) for i in range(24)]

    # Ensure scaler.pkl exists
    scaler_path = 'models/scaler.pkl'
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Missing scaler.pkl. Run train.py to generate it.")

    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

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
    # Adjust sequence_length based on available data
    sequence_length = min(24, len(df))
        if sequence_length == 0:
             raise ValueError("Error: No data available in train_data.csv. Run train.py again.")

        last_data = df.tail(sequence_length)

# Ensure last_data has enough rows after feature engineering
        if last_data.shape[0] < sequence_length:
            raise ValueError(f"Error: Need at least {sequence_length} rows for LSTM prediction, but found {last_data.shape[0]}.")

    # Add features
    last_data = add_features(last_data)

    # Define feature columns
    feature_columns = [
        'consumption', 'hour', 'dayofweek', 'month',
        'consumption_lag_1d', 'consumption_rolling_mean_24h', 'consumption_rolling_std_24h'
    ]

    # Ensure last_data has enough rows after feature engineering
    if last_data.shape[0] < sequence_length:
        raise ValueError(f"Error: Need at least {sequence_length} rows for LSTM prediction, but found {last_data.shape[0]}.")

    # Scale last_data
    last_data_scaled = scaler.transform(last_data[feature_columns])

    # Generate LSTM predictions
    lstm_forecast = predict_lstm(lstm_model, last_data, scaler, feature_columns, sequence_length, steps=24)
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
