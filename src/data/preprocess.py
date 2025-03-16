import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    """Load the dataset from CSV file."""
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    return df

def add_features(df):
    """Add time-based features for the model."""
    df = df.copy()
    
    # Time-based features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['quarter'] = df.index.quarter
    
    # Lag features
    df['consumption_lag_1d'] = df['consumption'].shift(24)
    df['consumption_lag_1w'] = df['consumption'].shift(168)
    
    # Rolling statistics
    df['consumption_rolling_mean_24h'] = df['consumption'].rolling(window=24).mean()
    df['consumption_rolling_std_24h'] = df['consumption'].rolling(window=24).std()
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def scale_data(df, feature_columns):
    """Scale the features using MinMaxScaler."""
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df_scaled, scaler

def create_sequences(data, target_column, sequence_length=24, horizon=24):
    """Create sequences for time series forecasting."""
    X, y = [], []
    for i in range(len(data) - sequence_length - horizon + 1):
        X.append(data[i:(i + sequence_length)].values)
        y.append(data[i + sequence_length:i + sequence_length + horizon][target_column].values)
    return np.array(X), np.array(y)

def train_test_split(df, test_size=0.2):
    """Split the data into training and testing sets."""
    train_size = int(len(df) * (1 - test_size))
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    return train_data, test_data
