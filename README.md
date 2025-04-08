# Gold Price Prediction System

A Streamlit web application that predicts gold prices using ARIMA (AutoRegressive Integrated Moving Average) time series analysis.

## Overview

This application analyzes historical gold price data and provides price forecasts for future periods. It utilizes statistical modeling to identify patterns in historical data and extrapolate those patterns to make predictions.

## Features

- **Automatic Data Retrieval**: Downloads the latest gold price data from Yahoo Finance
- **Data Preprocessing**: Cleans and normalizes data for analysis
- **Model Optimization**: Automatically finds the best ARIMA parameters for optimal predictions
- **Interactive Forecasting**: Allows users to specify the number of days to forecast
- **Visual Results**: Displays prediction results with interactive charts
- **Downloadable Forecasts**: Export prediction data as CSV files

## Installation

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Dependencies
Install the required packages:

```bash
pip install streamlit pandas numpy yfinance plotly statsmodels
```

### Running the Application

1. Clone this repository or download the source code
2. Navigate to the project directory
3. Run the following command:

```bash
streamlit run app.py
```

## Usage

1. Launch the application using the command above
2. Configure the model parameters in the sidebar:
   - Toggle "Model parametrelerini optimize et" to automatically find optimal parameters
   - Or manually set p, d, q parameters if optimization is turned off
   - Select the number of forecast days (7-60)
3. Click "ANALİZE BAŞLA" to start the analysis
4. View results in the three tabs:
   - "Tahmin Grafiği" (Prediction Graph): Visual representation of historical data and forecasts
   - "Tahmin Verileri" (Prediction Data): Tabular forecast data with download option
   - "Model Detayları" (Model Details): Technical information about the ARIMA model

## How It Works

1. **Data Collection**: The system retrieves historical gold price data using the yfinance library.
2. **Data Preparation**: The data is cleaned to handle missing values and prepared for analysis.
3. **Model Training**: An ARIMA model is fitted to the historical data.
4. **Parameter Optimization**: If enabled, the system tests multiple parameter combinations to find the best performing model.
5. **Forecasting**: The trained model generates price predictions for the specified number of days.
6. **Visualization**: Results are displayed through interactive charts and tables.

## Important Notes

- This application is for educational purposes only and should not be used for actual investment decisions.
- Forecasts are based on statistical patterns in historical data and do not account for unforeseen market events.
- The accuracy of predictions decreases as the forecast horizon increases.

## Technical Details

- **ARIMA Parameters**:
  - p: Order of the autoregressive part
  - d: Degree of differencing
  - q: Order of the moving average part
- **Performance Metric**: Mean Absolute Error (MAE)

## Acknowledgments

- Data provided by Yahoo Finance
- Built with Streamlit, Pandas, and Plotly
