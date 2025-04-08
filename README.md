# Gold Price Analysis and Prediction System

A comprehensive web application built with Streamlit that provides analysis and forecasting for gold prices using ARIMA time series modeling.

## Overview

This application fetches historical gold price data using Yahoo Finance API, performs various analytical operations, and provides forecasts for future price movements. The system is split into two main components:

1. **Gold Price Analysis**: Presents historical price trends, volatility analysis, moving averages, and support/resistance levels
2. **Gold Price Prediction**: Uses ARIMA (AutoRegressive Integrated Moving Average) models to forecast future prices

![Ekran görüntüsü 2025-04-08 230418](https://github.com/user-attachments/assets/e52263f7-13ab-4414-8499-d8a56f651ebe)

## Features

- Real-time fetching of gold price data via Yahoo Finance API
- Comprehensive price analysis with interactive visualizations
- Multiple timeframe analysis (1 month, 3 months, 6 months, 1 year)
- Moving average calculations (7-day, 30-day, 90-day)
- Support and resistance level identification
- ARIMA model-based price prediction with configurable parameters
- Export capabilities for forecast data

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gold-price-analysis.git
cd gold-price-analysis

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## Requirements

```
streamlit>=1.15.0
pandas>=1.3.0
numpy>=1.20.0
yfinance>=0.1.70
plotly>=5.3.0
statsmodels>=0.13.0
```

## Usage

```bash
streamlit run app.py
```

After running the command, the application will start and be accessible in your web browser at `http://localhost:8501`.

## Code Structure

### Data Management Functions

```python
def veri_cikar():
    """Fetches gold price data from Yahoo Finance API"""
    with st.spinner('Downloading gold price data...'):
        try:
            gc = yf.Ticker("GC=F")
            veri = gc.history(period="max")
            if veri.empty:
                st.error("Data could not be downloaded. Check your internet connection.")
                st.stop()
            return veri
        except Exception as e:
            st.error(f"Data download error: {str(e)}")
            st.stop()

def veriyi_kaydet(veri):
    """Saves data to a CSV file"""
    try:
        veri.to_csv('altin_fiyatlari.csv')
        st.success("Data successfully saved.")
    except Exception as e:
        st.error(f"Data save error: {str(e)}")

def veriyi_yukle():
    """Loads data from saved CSV file"""
    try:
        veri = pd.read_csv('altin_fiyatlari.csv', index_col='Date', parse_dates=['Date'])
        # Remove timezone info from index
        if isinstance(veri.index, pd.DatetimeIndex) and veri.index.tz is not None:
            veri.index = veri.index.tz_localize(None)
        return veri
    except FileNotFoundError:
        return None

def veri_temizle(veri):
    """Cleans and prepares data for analysis"""
    # Fill missing values
    veri.fillna(method='ffill', inplace=True)
    
    # Remove timezone info
    if isinstance(veri.index, pd.DatetimeIndex) and veri.index.tz is not None:
        veri.index = veri.index.tz_localize(None)
    
    # Check and fix column names
    if 'Close' not in veri.columns and 'Adj Close' in veri.columns:
        veri['Close'] = veri['Adj Close']
    
    # Convert to float type
    veri['Close'] = veri['Close'].astype(float)
    
    return veri
```

### Analysis Functions

```python
def altin_genel_analiz(veri):
    """Performs comprehensive analysis on gold price data"""
    st.subheader("Gold Price Analysis")
    
    # Last 1 year of data
    son_yil = veri.iloc[-365:].copy() if len(veri) >= 365 else veri.copy()
    
    # Price trend chart
    fig_trend = px.line(son_yil, y='Close', title='Last 1 Year Gold Price Trend')
    fig_trend.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD/oz)",
        height=400
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Percentage changes for different timeframes
    col1, col2, col3, col4 = st.columns(4)
    
    # ... (calculating and displaying price changes)
    
    # Volatility analysis
    st.subheader("Volatility Analysis")
    son_yil['Gunluk_Degisim'] = son_yil['Close'].pct_change() * 100
    
    # ... (creating volatility chart)
    
    # Moving average analysis
    st.subheader("Moving Average Analysis")
    son_yil['MA7'] = son_yil['Close'].rolling(window=7).mean()
    son_yil['MA30'] = son_yil['Close'].rolling(window=30).mean()
    son_yil['MA90'] = son_yil['Close'].rolling(window=90).mean()
    
    # ... (creating moving average chart)
    
    # Support and resistance analysis
    st.subheader("Support and Resistance Levels")
    son_30_gun = son_yil.iloc[-30:]
    direnç_seviyesi = son_30_gun['High'].max()
    destek_seviyesi = son_30_gun['Low'].min()
    pivot = (son_30_gun['High'].max() + son_30_gun['Low'].min() + son_30_gun['Close'].iloc[-1]) / 3
    
    # ... (displaying levels)
    
    return son_yil
```

### ARIMA Model Functions

```python
def tahmin_modeli_egit(veri, p, d, q, durum_metni):
    """Trains an ARIMA model with specified parameters"""
    try:
        durum_metni.text("Training model...")
        
        # Data preparation
        train_data = veri['Close'].dropna()
        
        # Model training
        model = ARIMA(train_data, order=(p, d, q))
        model_fit = model.fit()
        
        durum_metni.text("Model successfully trained ✓")
        return model_fit
    except Exception as e:
        durum_metni.error(f"Model error: {str(e)}")
        return None

def tahmin_yap(model_fit, gun_sayisi, durum_metni):
    """Makes price predictions using the trained ARIMA model"""
    durum_metni.text(f"Making predictions for {gun_sayisi} days...")
    try:
        # Check model information
        if model_fit is None:
            durum_metni.error("Model could not be trained!")
            return None
        
        # Make predictions
        forecast = model_fit.forecast(steps=gun_sayisi)
        
        # Prepare dates
        start_date = pd.Timestamp.now().normalize()
        forecast_dates = pd.date_range(start=start_date, periods=gun_sayisi, freq='D')
        
        # Create forecast series
        forecast_series = pd.Series(forecast, index=forecast_dates)
        
        # Check forecasts
        if forecast_series.isna().any():
            durum_metni.warning("Some forecast values could not be calculated.")
            forecast_series = forecast_series.fillna(method='ffill')
        
        durum_metni.text("Forecasting completed ✓")
        return forecast_series
    except Exception as e:
        durum_metni.error(f"Forecasting error: {str(e)}")
        import traceback
        durum_metni.code(traceback.format_exc())
        return None

def basit_tahmin_yap(veri, gun_sayisi):
    """Makes simple trend predictions when ARIMA fails"""
    # Average of last 30 days
    son_ortalama = veri['Close'].iloc[-30:].mean()
    
    # Calculate slope from last 7 days
    son_egim = (veri['Close'].iloc[-1] - veri['Close'].iloc[-8]) / 7
    
    # Simple linear forecast
    tahminler = [veri['Close'].iloc[-1] + son_egim * (i+1) for i in range(gun_sayisi)]
    
    forecast_dates = pd.date_range(start=pd.Timestamp.now().normalize(), periods=gun_sayisi, freq='D')
    return pd.Series(tahminler, index=forecast_dates)
```

## Main Application Function

```python
def main():
    """Main application function"""
    st.title("Gold Price Analysis and Prediction System")
    
    # Description
    st.markdown("""
    ### About the Project
    This application is designed to analyze and predict gold prices.
    It examines historical price data and attempts to forecast future price movements.
    
    **Features:**
    - Automatic downloading of historical gold price data
    - Detailed price analysis and charts
    - Moving averages and volatility calculations
    - Price prediction using ARIMA model
    - Visual graphs and statistical results
    
    **NOTE:** This application is for educational purposes only and should not be used for real investment decisions.
    """)
    
    # Create main tabs
    tab1, tab2 = st.tabs(["Gold Price Analysis", "Gold Price Prediction"])
    
    with tab1:
        # Load or download data
        data = veriyi_yukle()
        if data is None:
            st.info("Local data not found, downloading new data...")
            data = veri_cikar()
            veriyi_kaydet(data)
        
        # Clean data
        data = veri_temizle(data)
        
        # General analysis
        altin_genel_analiz(data)
    
    with tab2:
        # ... (prediction tab implementation)
```

## ARIMA Model Explanation

ARIMA (AutoRegressive Integrated Moving Average) is a statistical model used for analyzing and forecasting time series data:

- **p** (AR - AutoRegressive): The number of lag observations included in the model
- **d** (I - Integrated): The number of differencing operations required to make the series stationary
- **q** (MA - Moving Average): The size of the moving average window

The model combines three components:
1. **AR(p)**: Regression of the time series on its own past values
2. **I(d)**: Differencing to make series stationary
3. **MA(q)**: Dependency between observation and residual error from moving average model

![Ekran görüntüsü 2025-04-08 230451](https://github.com/user-attachments/assets/42580574-c4ae-421a-bb02-7157abefdca2)
![Ekran görüntüsü 2025-04-08 230434](https://github.com/user-attachments/assets/d91131e7-37eb-4d86-a566-d8a351cb92fe)


## Disclaimer

This project is for educational purposes only. Financial forecasts should not be the sole basis for investment decisions.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
