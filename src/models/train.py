import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import custom modules
from src.data.preprocess import load_data, add_features, scale_data, create_sequences, train_test_split
from src.models.models import train_arima, train_sarima, train_prophet, train_lstm, evaluate_model
from src.visualization.visualize import plot_time_series, plot_components, plot_forecast, plot_model_comparison

def scale_data(data, feature_columns, scaler=None):
    """
    Scales the given data using StandardScaler.
    
    Parameters:
    - data: DataFrame containing the data to scale.
    - feature_columns: List of column names to be scaled.
    - scaler: (Optional) Pre-fitted StandardScaler instance.

    Returns:
    - scaled_df: Scaled DataFrame with the same column names and index.
    - scaler: The StandardScaler used (newly created or existing).
    """
    if scaler is None:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[feature_columns])
    else:
        scaled_data = scaler.transform(data[feature_columns])
    
    scaled_df = pd.DataFrame(scaled_data, columns=feature_columns, index=data.index)
    return scaled_df, scaler

def main():
    # Load data
    print("Loading data...")
    df = load_data('data/raw/electricity_consumption.csv')

    # Exploratory data analysis
    print("Performing exploratory data analysis...")
    plot_time_series(df, 'consumption', title='Electricity Consumption Over Time')
    plt.savefig('results/time_series_plot.png')

    plot_components(df, 'consumption')
    plt.savefig('results/time_series_decomposition.png')

    # Feature engineering
    print("Adding features...")
    df_features = add_features(df)

    # Split data
    print("Splitting data into train and test sets...")
    train_data, test_data = train_test_split(df_features, test_size=0.2)

    # Save processed data
    train_data.to_csv('data/processed/train_data.csv')
    test_data.to_csv('data/processed/test_data.csv')

    # Train models
    print("Training models...")
    results = {}

    # 1. ARIMA Model
    print("Training ARIMA model...")
    arima_model = train_arima(train_data['consumption'], order=(2, 1, 2))
    arima_forecast = arima_model.forecast(steps=len(test_data))
    arima_forecast = pd.Series(arima_forecast, index=test_data.index)

    # Save ARIMA model
    with open('models/arima_model.pkl', 'wb') as f:
        pickle.dump(arima_model, f)

    # Evaluate ARIMA
    results['ARIMA'] = evaluate_model(test_data['consumption'], arima_forecast)
    print(f"ARIMA Results: {results['ARIMA']}")

    # Plot ARIMA forecast
    plot_forecast(test_data['consumption'], arima_forecast, title='ARIMA Forecast vs Actual')
    plt.savefig('results/arima_forecast.png')

    # 2. SARIMA Model
    print("Training SARIMA model...")
    sarima_model = train_sarima(train_data['consumption'], order=(2, 1, 2), seasonal_order=(1, 1, 1, 24))
    sarima_forecast = sarima_model.forecast(steps=len(test_data))
    sarima_forecast = pd.Series(sarima_forecast, index=test_data.index)

    # Save SARIMA model
    with open('models/sarima_model.pkl', 'wb') as f:
        pickle.dump(sarima_model, f)

    # Evaluate SARIMA
    results['SARIMA'] = evaluate_model(test_data['consumption'], sarima_forecast)
    print(f"SARIMA Results: {results['SARIMA']}")

    # Plot SARIMA forecast
    plot_forecast(test_data['consumption'], sarima_forecast, title='SARIMA Forecast vs Actual')
    plt.savefig('results/sarima_forecast.png')

    # 3. Prophet Model
    print("Training Prophet model...")
    prophet_model = train_prophet(train_data[['consumption']])

    # Future dataframe for Prophet prediction
    future = prophet_model.make_future_dataframe(periods=len(test_data), freq='H')
    prophet_forecast = prophet_model.predict(future).tail(len(test_data))
    prophet_forecast = pd.Series(prophet_forecast['yhat'].values, index=test_data.index)

    # Save Prophet model
    with open('models/prophet_model.pkl', 'wb') as f:
        pickle.dump(prophet_model, f)

    # Evaluate Prophet
    results['Prophet'] = evaluate_model(test_data['consumption'], prophet_forecast)
    print(f"Prophet Results: {results['Prophet']}")

    # Plot Prophet forecast
    plot_forecast(test_data['consumption'], prophet_forecast, title='Prophet Forecast vs Actual')
    plt.savefig('results/prophet_forecast.png')

    # 4. LSTM Model
    print("Training LSTM model...")
    
    # Feature columns for LSTM
    feature_columns = ['consumption', 'hour', 'dayofweek', 'month', 
                       'consumption_lag_1d', 'consumption_rolling_mean_24h', 'consumption_rolling_std_24h']

    # Scale data
    train_scaled, scaler = scale_data(train_data, feature_columns)
    test_scaled, _ = scale_data(test_data, feature_columns, scaler)

    # Create sequences for LSTM
    sequence_length = 24
    horizon = 24

    X_train, y_train = create_sequences(train_scaled, 'consumption', sequence_length, horizon)
    X_test, y_test = create_sequences(test_scaled, 'consumption', sequence_length, horizon)

    # Train LSTM model
    lstm_model = train_lstm(X_train, y_train, (X_train.shape[1], X_train.shape[2]))

    # Save LSTM model
    lstm_model.save('models/lstm_model.h5')

    # Predict with LSTM
    lstm_predictions = lstm_model.predict(X_test)

    # Reshape predictions for evaluation
    y_test_reshaped = y_test.reshape(-1)
    lstm_predictions_reshaped = lstm_predictions.reshape(-1)

    # Evaluate LSTM
    results['LSTM'] = evaluate_model(y_test_reshaped, lstm_predictions_reshaped)
    print(f"LSTM Results: {results['LSTM']}")

    # Compare models
    plot_model_comparison(results)
    plt.savefig('results/model_comparison.png')

    # Save model results
    with open('results/model_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("Training completed successfully!")

if __name__ == "__main__":
    main()
