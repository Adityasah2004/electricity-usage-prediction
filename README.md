A complete machine learning project for predicting electricity usage patterns using time series forecasting models.

## Project Overview

This project aims to predict electricity consumption patterns using various time series forecasting models. The models are trained on historical electricity consumption data and can be used to forecast future consumption.

## Project Structure

```
.
├── data/
│   ├── raw/                 # Raw electricity consumption data
│   └── processed/           # Processed data ready for modeling
├── models/                  # Trained model files
├── notebooks/               # Jupyter notebooks for exploration and analysis
├── src/                     # Source code
│   ├── data/                # Data processing scripts
│   ├── features/            # Feature engineering scripts
│   ├── models/              # Model training and prediction scripts
│   └── visualization/       # Visualization utilities
├── utils/                   # Utility functions
├── results/                 # Model results and visualizations
├── tests/                   # Unit tests
├── README.md                # Project documentation
└── requirements.txt         # Project dependencies
```

- **data/**: Contains raw and processed data
  - **raw/**: Raw electricity consumption data
  - **processed/**: Processed data ready for modeling
- **models/**: Trained model files
- **notebooks/**: Jupyter notebooks for exploration and analysis
- **src/**: Source code
  - **data/**: Data processing scripts
  - **features/**: Feature engineering scripts
  - **models/**: Model training and prediction scripts
  - **visualization/**: Visualization utilities
- **utils/**: Utility functions
- **results/**: Model results and visualizations
- **tests/**: Unit tests

## Models Implemented

1. **ARIMA**: Autoregressive Integrated Moving Average
2. **SARIMA**: Seasonal ARIMA
3. **Prophet**: Facebook's Prophet forecasting model
4. **LSTM**: Long Short-Term Memory neural network

## Dataset

The project uses a synthetic electricity consumption dataset with the following characteristics:
- Hourly electricity consumption data
- Daily patterns (higher during day, lower at night)
- Weekly patterns (lower on weekends)
- Seasonal patterns (higher in winter and summer)
- Long-term trend
- Random noise

In a real-world scenario, you could use datasets from:
- Kaggle: [Household Electric Power Consumption](https://www.kaggle.com/uciml/electric-power-consumption-data-set)
- UCI Machine Learning Repository: [Individual Household Electric Power Consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)
- Your local utility company's data

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages: pandas, numpy, matplotlib, scikit-learn, statsmodels, prophet, tensorflow, seaborn
