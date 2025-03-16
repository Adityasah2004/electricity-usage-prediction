import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_time_series(df, column, title='Time Series Plot', figsize=(15, 6)):
    """Plot a time series."""
    plt.figure(figsize=figsize)
    plt.plot(df.index, df[column])
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.grid(True)
    plt.tight_layout()
    return plt

def plot_components(df, column):
    """Plot the decomposition of a time series."""
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Resample to daily data for better visualization
    daily_data = df[column].resample('D').mean()
    
    # Decompose the time series
    decomposition = seasonal_decompose(daily_data, model='additive', period=30)
    
    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    
    decomposition.observed.plot(ax=axes[0], title='Observed')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual')
    
    plt.tight_layout()
    return plt

def plot_forecast(actual, predicted, title='Forecast vs Actual', figsize=(15, 6)):
    """Plot the forecast against the actual values."""
    plt.figure(figsize=figsize)
    plt.plot(actual.index, actual, label='Actual', color='blue')
    plt.plot(predicted.index, predicted, label='Forecast', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Consumption')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt

def plot_feature_importance(model, feature_names):
    """Plot feature importance for tree-based models."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    importances.plot(kind='bar')
    plt.title('Feature Importance')
    plt.tight_layout()
    return plt

def plot_model_comparison(results_dict):
    """Plot comparison of different models."""
    metrics = list(next(iter(results_dict.values())).keys())
    models = list(results_dict.keys())
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
    
    for i, metric in enumerate(metrics):
        values = [results_dict[model][metric] for model in models]
        axes[i].bar(models, values)
        axes[i].set_title(f'Model Comparison - {metric}')
        axes[i].set_ylabel(metric)
        axes[i].grid(True, axis='y')
    
    plt.tight_layout()
    return plt
