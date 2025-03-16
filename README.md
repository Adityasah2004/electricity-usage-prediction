A complete machine learning project for predicting electricity usage patterns using time series forecasting models.

## Project Overview

This project aims to predict electricity consumption patterns using various time series forecasting models. The models are trained on historical electricity consumption data and can be used to forecast future consumption.

### Key Components

1. **Data Generation**: The project creates a synthetic electricity consumption dataset with realistic patterns:
   - Daily patterns (higher during day, lower at night)
   - Weekly patterns (lower on weekends)
   - Seasonal patterns (higher in winter and summer)
   - Long-term trend
   - Random noise

2. **Data Processing**: Includes functions for loading, cleaning, feature engineering, and preparing data for modeling.

3. **Models Implemented**:
   - ARIMA (Autoregressive Integrated Moving Average)
   - SARIMA (Seasonal ARIMA)
   - Prophet (Facebook's forecasting model)
   - LSTM (Long Short-Term Memory neural network)

4. **Evaluation**: Models are evaluated using multiple metrics:
   - Mean Absolute Error (MAE)
   - Root Mean Square Error (RMSE)
   - Mean Absolute Percentage Error (MAPE)

5. **Visualization**: Comprehensive visualization tools for data exploration and model comparison.

6. **Notebooks**: Interactive Jupyter notebooks for:
   - Data exploration and analysis
   - Model comparison and evaluation

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

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Adityasah2004/electricity-usage-prediction.git

cd electricity-usage-prediction
```


2. Create a virtual environment and install dependencies:
```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```


3. Generate the dataset:
```
python main.py
```


### Running the Project

1. Explore the data:
```bash
jupyter notebook notebooks/data_exploration.ipynb
```

2. Train the models:
```
python src/models/train.py
```

3. Generate predictions:
```
python src/models/predict.py
```


4. Compare model performance:
```
jupyter notebook notebooks/model_comparison.ipynb
```
## Results

The project evaluates models using the following metrics:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

Visualizations include:
- Time series plots of actual vs. predicted values
- Error distributions
- Error by time of day
- Model comparison charts

## Future Improvements

- Implement more advanced deep learning models (e.g., Transformer-based models)
- Add feature importance analysis
- Incorporate external features (e.g., weather data, holidays)
- Develop an ensemble model with weighted averaging
- Create a web application for interactive forecasting

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Statsmodels](https://www.statsmodels.org/) for statistical models
- [Prophet](https://facebook.github.io/prophet/) for the Prophet forecasting model
- [TensorFlow](https://www.tensorflow.org/) for deep learning models
