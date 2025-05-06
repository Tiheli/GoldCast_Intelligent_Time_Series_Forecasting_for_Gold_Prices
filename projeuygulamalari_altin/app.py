import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import datetime
import time
import traceback
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Set page title and configuration
st.set_page_config(page_title="Altın Fiyat Tahmin Sistemi", layout="wide")

# ----------------------- Data Management Functions -----------------------
def veri_cikar(force_download=False):
    """Güncel altın fiyat verilerini çeker"""
    with st.spinner('Altın fiyat verileri indiriliyor...'):
        try:
            gc = yf.Ticker("GC=F")
            veri = gc.history(period="max")
            if veri.empty:
                raise ValueError("Veri indirilemedi")
            
            # Veriyi günlük olarak yeniden örnekle ve eksik değerleri doldur
            veri = veri.resample('D').mean()
            veri = veri.fillna(method='ffill').fillna(method='bfill')
            
            if 'Close' not in veri.columns:
                if 'Adj Close' in veri.columns:
                    veri['Close'] = veri['Adj Close']
                else:
                    raise ValueError("Fiyat verisi bulunamadı")
            
            return veri
            
        except Exception as e:
            st.error(f"Veri indirme hatası: {str(e)}")
            return pd.DataFrame()

def doviz_kuru_cikar(force_download=False):
    """Güncel USD/TRY döviz kuru verilerini çeker"""
    with st.spinner('Döviz kuru verileri indiriliyor...'):
        try:
            usd_try = yf.Ticker("USDTRY=X")
            kur_veri = usd_try.history(period="max")
            if kur_veri.empty:
                raise ValueError("Döviz kuru verileri indirilemedi")
            
            kur_veri = kur_veri.resample('D').mean()
            kur_veri = kur_veri.fillna(method='ffill').fillna(method='bfill')
            
            if 'Close' not in kur_veri.columns:
                if 'Adj Close' in kur_veri.columns:
                    kur_veri['Close'] = kur_veri['Adj Close']
                else:
                    raise ValueError("Döviz kuru verisi bulunamadı")
                    
            return kur_veri
            
        except Exception as e:
            st.error(f"Döviz kuru verisi indirme hatası: {str(e)}")
            return pd.DataFrame()

def veriyi_kaydet(veri, dosya_adi='altin_fiyatlari.csv'):
    try:
        veri.to_csv(dosya_adi)
        st.success(f"{dosya_adi} başarıyla kaydedildi.")
    except Exception as e:
        st.error(f"Veri kaydetme hatası: {str(e)}")

def veri_yukle(dosya_adi='altin_fiyatlari.csv'):
    try:
        veri = pd.read_csv(dosya_adi, index_col='Date', parse_dates=['Date'])
        if isinstance(veri.index, pd.DatetimeIndex) and veri.index.tz is not None:
            veri.index = veri.index.tz_localize(None)
        return veri
    except FileNotFoundError:
        return None

def veri_temizle(veri):
    veri.fillna(method='ffill', inplace=True)
    if isinstance(veri.index, pd.DatetimeIndex) and veri.index.tz is not None:
        veri.index = veri.index.tz_localize(None)
    if 'Close' not in veri.columns and 'Adj Close' in veri.columns:
        veri['Close'] = veri['Adj Close']
    veri['Close'] = veri['Close'].astype(float)
    return veri

def ons_to_gram(ons_price):
    return ons_price / 31.1035

def usd_to_try(usd_price, kur_veri):
    if kur_veri is None or kur_veri.empty:
        return usd_price * 30.0
    son_kur = kur_veri['Close'].iloc[-1]
    return usd_price * son_kur

def check_stationarity(data, durum_metni):
    result = adfuller(data)
    p_value = result[1]
    if p_value > 0.05:
        durum_metni.info("Veri durağan değil, fark alma işlemi uygulanıyor...")
        diff_data = data.diff().dropna()
        result = adfuller(diff_data)
        p_value = result[1]
        if p_value > 0.05:
            diff_data = diff_data.diff().dropna()
            return diff_data, 2
        return diff_data, 1
    return data, 0

# ----------------------- Model Functions -----------------------
def tahmin_modeli_egit(veri, p, d, q, durum_metni):
    try:
        durum_metni.text("ARIMA modeli eğitiliyor...")
        train_data = veri['Close'].dropna()
        if train_data.min() <= 0:
            train_data = train_data + abs(train_data.min()) + 1  # Negatif veya sıfır değerleri pozitif yap
        train_data = np.log(train_data)
        train_data, actual_d = check_stationarity(train_data, durum_metni)
        if len(train_data) < 10:
            raise ValueError("Yeterli veri yok! En az 10 veri noktası gereklidir.")
        model = ARIMA(train_data, order=(p, actual_d, q))
        model_fit = model.fit(maxiter=1000, tol=1e-3)
        durum_metni.text("ARIMA modeli başarıyla eğitildi ✓")
        return model_fit, actual_d
    except Exception as e:
        durum_metni.error(f"ARIMA modeli hatası: {traceback.format_exc()}")
        return None, d

def tahmin_yap(model_fit, gun_sayisi, durum_metni, d_value, son_deger):
    try:
        if model_fit is None:
            durum_metni.error("ARIMA modeli eğitilemedi!")
            return None
        forecast = model_fit.forecast(steps=gun_sayisi)
        if d_value >= 1:
            last_value = np.log(son_deger)
            forecast_levels = [last_value]
            for f in forecast:
                forecast_levels.append(forecast_levels[-1] + f)
            forecast_levels = forecast_levels[1:]
            forecast = np.array(forecast_levels)
        forecast = np.exp(forecast)
        start_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
        forecast_dates = pd.date_range(start=start_date + pd.Timedelta(days=1), periods=gun_sayisi, freq='D')
        forecast_series = pd.Series(forecast, index=forecast_dates)
        if forecast_series.isna().any():
            durum_metni.warning("Bazı tahmin değerleri hesaplanamadı.")
            forecast_series = forecast_series.fillna(method='ffill')
        forecast_series[forecast_series < 0] = son_deger
        durum_metni.text("ARIMA tahmini tamamlandı ✓")
        return forecast_series
    except Exception as e:
        durum_metni.error(f"ARIMA tahmini hatası: {traceback.format_exc()}")
        return None

def prophet_tahmin_yap(veri, gun_sayisi, durum_metni):
    try:
        durum_metni.text("Prophet modeli ile tahmin yapılıyor...")
        prophet_data = pd.DataFrame({
            'ds': veri.index,
            'y': veri['Close']
        })
        if prophet_data['y'].min() <= 0:
            prophet_data['y'] = prophet_data['y'] + abs(prophet_data['y'].min()) + 1
        if len(prophet_data) < 10:
            raise ValueError("Yeterli veri yok! En az 10 veri noktası gereklidir.")
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(prophet_data)
        start_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
        future_dates = pd.date_range(start=start_date + pd.Timedelta(days=1), periods=gun_sayisi, freq='D')
        future = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future)
        forecast_series = pd.Series(forecast['yhat'].values, index=future_dates)
        son_deger = veri['Close'].iloc[-1]
        forecast_series[forecast_series < 0] = son_deger
        durum_metni.text("Prophet tahmini tamamlandı ✓")
        return forecast_series
    except Exception as e:
        durum_metni.error(f"Prophet tahmini hatası: {traceback.format_exc()}")
        return None

def linear_interpolation_tahmin(veri, gun_sayisi):
    son_deger = veri['Close'].iloc[-1]
    start_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    forecast_dates = pd.date_range(start=start_date + pd.Timedelta(days=1), periods=gun_sayisi, freq='D')
    son_7_gun = veri['Close'].iloc[-7:]
    egim = (son_7_gun.iloc[-1] - son_7_gun.iloc[0]) / 6
    tahminler = [son_deger + i * egim for i in range(gun_sayisi)]
    return pd.Series(tahminler, index=forecast_dates)

def teknik_analiz_gostergeleri(veri):
    delta = veri['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    exp1 = veri['Close'].ewm(span=12, adjust=False).mean()
    exp2 = veri['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    sma = veri['Close'].rolling(window=20).mean()
    std = veri['Close'].rolling(window=20).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return rsi, macd, signal, upper_band, lower_band

def destek_direnc_seviyeleri(veri):
    pivot = (veri['High'] + veri['Low'] + veri['Close']) / 3
    r1 = 2 * pivot - veri['Low']
    r2 = pivot + (veri['High'] - veri['Low'])
    s1 = 2 * pivot - veri['High']
    s2 = pivot - (veri['High'] - veri['Low'])
    return {'Pivot': pivot, 'R1': r1, 'R2': r2, 'S1': s1, 'S2': s2}

def trend_analizi(veri):
    uzun_ma = veri['Close'].rolling(window=200).mean()
    orta_ma = veri['Close'].rolling(window=50).mean()
    kisa_ma = veri['Close'].rolling(window=20).mean()
    trend = 'Belirsiz'
    if kisa_ma.iloc[-1] > orta_ma.iloc[-1] > uzun_ma.iloc[-1]:
        trend = 'Güçlü Yükseliş'
    elif kisa_ma.iloc[-1] < orta_ma.iloc[-1] < uzun_ma.iloc[-1]:
        trend = 'Güçlü Düşüş'
    return trend, kisa_ma, orta_ma, uzun_ma

def altin_genel_analiz(veri, birim_secimi, kur_veri):
    st.subheader("Genel Altın Fiyat Analizi")
    last_90_days = veri.iloc[-90:].copy()
    if isinstance(last_90_days.index, pd.DatetimeIndex) and last_90_days.index.tz is not None:
        last_90_days.index = last_90_days.index.tz_localize(None)
    
    # Fiyat birimine göre dönüşüm
    if birim_secimi == "USD/Gram":
        last_90_days['Close'] = last_90_days['Close'].apply(ons_to_gram)
        birim_metin = "USD/Gram"
        birim_sembol = "$"
    elif birim_secimi == "TL/Gram":
        last_90_days['Close'] = last_90_days['Close'].apply(ons_to_gram)
        last_90_days['Close'] = last_90_days['Close'].apply(lambda x: usd_to_try(x, kur_veri))
        birim_metin = "TL/Gram"
        birim_sembol = "₺"
    else:
        birim_metin = "USD/Ons"
        birim_sembol = "$"
    
    # Temel istatistikler
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Son Fiyat", f"{birim_sembol}{last_90_days['Close'].iloc[-1]:.2f}")
    with col2:
        degisim = ((last_90_days['Close'].iloc[-1] - last_90_days['Close'].iloc[-2]) / last_90_days['Close'].iloc[-2]) * 100
        st.metric("Günlük Değişim", f"%{degisim:.2f}")
    with col3:
        volatilite = last_90_days['Close'].pct_change().std() * 100
        st.metric("Volatilite", f"%{volatilite:.2f}")
    
    # Fiyat Trendi Grafiği
    st.subheader(f"Fiyat Trendi (Son 90 Gün) - {birim_metin}")
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=last_90_days.index, y=last_90_days['Close'], 
                                   name='Kapanış Fiyatı', line=dict(color='blue')))
    fig_trend.update_layout(
        title=f'Altın Fiyat Trendi ({birim_metin})',
        xaxis_title='Tarih',
        yaxis_title=f'Fiyat ({birim_metin})',
        hovermode='x unified',
        legend=dict(x=0, y=1, orientation='h'),
        height=400
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Hareketli Ortalamalar Grafiği
    st.subheader("Hareketli Ortalamalar")
    last_90_days['MA7'] = last_90_days['Close'].rolling(window=7).mean()
    last_90_days['MA30'] = last_90_days['Close'].rolling(window=30).mean()
    last_90_days['MA90'] = last_90_days['Close'].rolling(window=90).mean()
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=last_90_days.index, y=last_90_days['Close'], 
                                mode='lines', name='Günlük Kapanış', line=dict(color='blue')))
    fig_ma.add_trace(go.Scatter(x=last_90_days.index, y=last_90_days['MA7'], 
                                mode='lines', name='7 Günlük MA', line=dict(color='orange')))
    fig_ma.add_trace(go.Scatter(x=last_90_days.index, y=last_90_days['MA30'], 
                                mode='lines', name='30 Günlük MA', line=dict(color='green')))
    fig_ma.add_trace(go.Scatter(x=last_90_days.index, y=last_90_days['MA90'], 
                                mode='lines', name='90 Günlük MA', line=dict(color='red')))
    fig_ma.update_layout(
        title='Hareketli Ortalamalar',
        xaxis_title='Tarih',
        yaxis_title=f'Fiyat ({birim_metin})',
        hovermode='x unified',
        legend=dict(x=0, y=1, orientation='h'),
        height=400
    )
    st.plotly_chart(fig_ma, use_container_width=True)
    
    # Volatilite Grafiği
    st.subheader("Volatilite Analizi")
    last_90_days['Gunluk_Degisim'] = last_90_days['Close'].pct_change() * 100
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=last_90_days.index, y=last_90_days['Gunluk_Degisim'], 
                                 mode='lines', name='Günlük Değişim %', line=dict(color='purple')))
    fig_vol.update_layout(
        title='Günlük Fiyat Değişim Yüzdesi',
        xaxis_title='Tarih',
        yaxis_title='Değişim Yüzdesi (%)',
        hovermode='x unified',
        legend=dict(x=0, y=1, orientation='h'),
        height=300
    )
    st.plotly_chart(fig_vol, use_container_width=True)
    
    # Destek ve Direnç Seviyeleri
    st.subheader("Destek ve Direnç Seviyeleri")
    son_30_gun = last_90_days.iloc[-30:]
    direnc_seviyesi = son_30_gun['Close'].max()
    destek_seviyesi = son_30_gun['Close'].min()
    pivot = (son_30_gun['Close'].max() + son_30_gun['Close'].min() + son_30_gun['Close'].iloc[-1]) / 3
    col1, col2, col3 = st.columns(3)
    col1.metric("Destek Seviyesi", f"{birim_sembol}{destek_seviyesi:.2f}")
    col2.metric("Pivot", f"{birim_sembol}{pivot:.2f}")
    col3.metric("Direnç Seviyesi", f"{birim_sembol}{direnc_seviyesi:.2f}")
    
    # Trend analizi sonuçları
    trend, kisa_ma, orta_ma, uzun_ma = trend_analizi(veri)
    st.subheader("Trend Analizi")
    st.write(f"Mevcut Trend: {trend}")

# ----------------------- Main Application -----------------------
def main():
    st.title("Altın Fiyat Analiz ve Tahmin Sistemi")
    st.markdown("""
    ### Proje Hakkında
    Bu uygulama, altın fiyatlarını analiz etmek ve tahmin etmek için geliştirilmiştir. 
    Tarihsel altın fiyat verilerini kullanarak fiyat trendlerini inceler ve gelecekteki fiyat hareketlerini öngörmeye çalışır.

    **Özellikler:**
    - Tarihsel altın fiyat verilerini otomatik olarak indirme
    - Detaylı fiyat analizi ve grafikleri
    - Hareketli ortalamalar ve volatilite hesaplamaları
    - ARIMA ve Prophet modelleri ile fiyat tahmini
    - Döviz kuru çevirisi ve gram altın fiyat tahmini
    - Görsel grafik ve istatistiksel sonuçlar

    **NOT:** Bu uygulama yalnızca eğitim amaçlıdır ve gerçek yatırım kararları için kullanılmamalıdır.
    """)

    st.sidebar.header("Veri Seçenekleri")
    veri_guncelle = st.sidebar.button("Verileri güncelle")

    notification_container = st.container()
    with notification_container:
        durum_metni = st.empty()

    tab1, tab2 = st.tabs(["Altın Fiyat Analizi", "Altın Fiyat Tahmini"])

    with tab1:
        try:
            data = veri_yukle()
            if veri_guncelle or data is None:
                durum_metni.info("Veriler güncelleniyor...")
                data = veri_cikar()
                if data.empty:
                    raise ValueError("Veri güncellenemedi")
                veriyi_kaydet(data)
                st.success("Veriler başarıyla güncellendi!")
            
            data = veri_temizle(data)
            if data.empty:
                raise ValueError("Veri işlenemedi")
                
            st.info(f"Veri son güncelleme tarihi: {data.index[-1].strftime('%d.%m.%Y')}")
            birim_secimi = st.sidebar.radio("Fiyat birimi seçimi:", ["USD/Ons", "USD/Gram", "TL/Gram"], key="analiz_birim")
            kur_veri = veri_yukle('doviz_kuru.csv')
            if veri_guncelle or kur_veri is None:
                kur_veri = doviz_kuru_cikar()
            altin_genel_analiz(data, birim_secimi, kur_veri)
            
        except Exception as e:
            st.error(f"Veri yükleme hatası: {str(e)}")
            st.stop()

    with tab2:
        try:
            st.subheader("Altın Fiyat Tahmini")
            kur_veri = veri_yukle('doviz_kuru.csv')
            if veri_guncelle or kur_veri is None:
                kur_veri = doviz_kuru_cikar()
                if kur_veri.empty:
                    st.warning("Döviz kuru verisi alınamadı, varsayılan kur kullanılacak")
                    kur_veri = None
                else:
                    veriyi_kaydet(kur_veri, 'doviz_kuru.csv')
            
            if kur_veri is not None and not kur_veri.empty:
                kur_veri = veri_temizle(kur_veri)
                st.info(f"Döviz kuru son güncelleme tarihi: {kur_veri.index[-1].strftime('%d.%m.%Y')}")

            st.sidebar.header("Tahmin Parametreleri")
            p = st.sidebar.slider("p parametresi (AR)", 0, 5, 1)
            d = st.sidebar.slider("d parametresi (I)", 0, 2, 1)
            q = st.sidebar.slider("q parametresi (MA)", 0, 5, 1)
            gun_sayisi = st.sidebar.slider("Tahmin gün sayısı", 7, 60, 30)
            birim_secimi = st.sidebar.radio("Fiyat birimi seçimi:", ["USD/Ons", "USD/Gram", "TL/Gram"], key="tahmin_birim")

            start_button = st.button("TAHMİN YAP", use_container_width=True)

            if start_button:
                data = veri_yukle()
                if data is None or veri_guncelle:
                    data = veri_cikar()
                    if data.empty:
                        raise ValueError("Veri indirilemedi")
                    veriyi_kaydet(data)
                data = veri_temizle(data)
                if data.empty:
                    raise ValueError("Veri işlenemedi")
                st.info(f"Veri son güncelleme tarihi: {data.index[-1].strftime('%d.%m.%Y')}")
                
                ilerleme_cubugu = st.progress(0)
                ilerleme_cubugu.progress(0.3)
                model_fit, actual_d = tahmin_modeli_egit(data, p, d, q, durum_metni)
                ilerleme_cubugu.progress(0.6)
                arima_forecast = None
                if model_fit is not None:
                    son_deger = data['Close'].iloc[-1]
                    arima_forecast = tahmin_yap(model_fit, gun_sayisi, durum_metni, actual_d, son_deger)
                    ilerleme_cubugu.progress(0.9)
                prophet_forecast = None
                if arima_forecast is None or arima_forecast.isna().any():
                    durum_metni.warning("ARIMA modeli başarısız oldu, Prophet modeli deneniyor...")
                    prophet_forecast = prophet_tahmin_yap(data, gun_sayisi, durum_metni)
                    if prophet_forecast is None or prophet_forecast.isna().any():
                        durum_metni.warning("Prophet modeli de başarısız oldu, lineer interpolasyon kullanılıyor...")
                        forecast = linear_interpolation_tahmin(data, gun_sayisi)
                    else:
                        durum_metni.success("Prophet modeli başarıyla çalıştı!")
                        forecast = prophet_forecast
                else:
                    durum_metni.success("ARIMA modeli başarıyla çalıştı!")
                    forecast = arima_forecast
                ilerleme_cubugu.progress(1.0)
                durum_metni.success("Tahmin tamamlandı!")
                
                # Fiyat birimine göre dönüşüm
                if birim_secimi == "USD/Gram":
                    forecast = forecast.apply(ons_to_gram)
                    last_90_days = data.iloc[-90:].copy()
                    last_90_days['Close'] = last_90_days['Close'].apply(ons_to_gram)
                    birim_metin = "USD/Gram"
                    son_fiyat = ons_to_gram(data['Close'].iloc[-1])
                    birim_sembol = "$"
                elif birim_secimi == "TL/Gram":
                    forecast_gram = forecast.apply(ons_to_gram)
                    forecast = forecast_gram.apply(lambda x: usd_to_try(x, kur_veri))
                    last_90_days = data.iloc[-90:].copy()
                    last_90_days['Close'] = last_90_days['Close'].apply(ons_to_gram)
                    last_90_days['Close'] = last_90_days['Close'].apply(lambda x: usd_to_try(x, kur_veri))
                    birim_metin = "TL/Gram"
                    son_fiyat = usd_to_try(ons_to_gram(data['Close'].iloc[-1]), kur_veri)
                    birim_sembol = "₺"
                else:
                    last_90_days = data.iloc[-90:].copy()
                    birim_metin = "USD/Ons"
                    son_fiyat = data['Close'].iloc[-1]
                    birim_sembol = "$"
                
                st.subheader(f"Altın Fiyat Tahmini ({birim_metin})")
                fig = go.Figure()
                if isinstance(last_90_days.index, pd.DatetimeIndex) and last_90_days.index.tz is not None:
                    last_90_days.index = last_90_days.index.tz_localize(None)
                if isinstance(forecast.index, pd.DatetimeIndex) and forecast.index.tz is not None:
                    forecast.index = forecast.index.tz_localize(None)
                fig.add_trace(go.Scatter(x=last_90_days.index, y=last_90_days['Close'], 
                                        name='Tarihsel Veri', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=forecast.index, y=forecast, 
                                        name='Tahmin', line=dict(color='red', dash='dash')))
                model_text = "ARIMA" if arima_forecast is not None else "Prophet" if prophet_forecast is not None else "Linear Interpolation"
                fig.update_layout(
                    title=f'Altın Fiyat Tahmini ({model_text} Model)',
                    xaxis_title='Tarih',
                    yaxis_title=f'Fiyat ({birim_metin})',
                    hovermode='x unified',
                    legend=dict(x=0, y=1, orientation='h')
                )
                st.plotly_chart(fig, use_container_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Özet Bilgiler")
                    st.write(f"Model: {'ARIMA(' + str(p) + ',' + str(actual_d) + ',' + str(q) + ')' if arima_forecast is not None else 'Prophet' if prophet_forecast is not None else 'Linear Interpolation'}")
                    st.write(f"Veri miktarı: {len(data)} gün")
                    st.write(f"Tahmin yapılan gün sayısı: {gun_sayisi}")
                    st.write(f"Birim: {birim_metin}")
                    if birim_secimi.startswith("TL"):
                        st.write(f"Güncel Döviz Kuru: {kur_veri['Close'].iloc[-1]:.2f} TL/USD" if kur_veri is not None and not kur_veri.empty else "Döviz kuru verisi alınamadı")
                with col2:
                    st.subheader("Tahmin Sonucu")
                    st.write(f"Son fiyat: {birim_sembol}{son_fiyat:.2f}")
                    st.write(f"Tahmin edilen son fiyat: {birim_sembol}{forecast.iloc[-1]:.2f}")
                    change = ((forecast.iloc[-1] - son_fiyat) / son_fiyat) * 100
                    st.write(f"Beklenen değişim: %{change:.2f}")
                    if change > 0:
                        st.markdown(f"<span style='color:green; font-size:20px;'>Yükseliş bekleniyor ↑</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span style='color:red; font-size:20px;'>Düşüş bekleniyor ↓</span>", unsafe_allow_html=True)
                st.subheader("Tahmin Verileri")
                forecast_df = pd.DataFrame({
                    'Tarih': forecast.index.date, 
                    f'Tahmin Edilen Fiyat ({birim_metin})': forecast.values
                })
                st.dataframe(forecast_df)

        except Exception as e:
            st.error(f"Tahmin hatası: {traceback.format_exc()}")
            st.stop()

if __name__ == "__main__":
    main()