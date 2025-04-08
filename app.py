import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import datetime
import time

# Set page title and configuration
st.set_page_config(page_title="Altın Fiyat Analiz ve Tahmin Sistemi", layout="wide")

# ----------------------- Data Management Functions -----------------------
def veri_cikar():
    with st.spinner('Altın fiyat verileri indiriliyor...'):
        try:
            gc = yf.Ticker("GC=F")
            veri = gc.history(period="max")
            if veri.empty:
                st.error("Veri indirilemedi. İnternet bağlantınızı kontrol edin.")
                st.stop()
            return veri
        except Exception as e:
            st.error(f"Veri indirme hatası: {str(e)}")
            st.stop()

def veriyi_kaydet(veri):
    try:
        veri.to_csv('altin_fiyatlari.csv')
        st.success("Veri başarıyla kaydedildi.")
    except Exception as e:
        st.error(f"Veri kaydetme hatası: {str(e)}")

def veriyi_yukle():
    try:
        veri = pd.read_csv('altin_fiyatlari.csv', index_col='Date', parse_dates=['Date'])
        # İndeksin zaman dilimi bilgisini kaldır
        if isinstance(veri.index, pd.DatetimeIndex) and veri.index.tz is not None:
            veri.index = veri.index.tz_localize(None)
        return veri
    except FileNotFoundError:
        return None

def veri_temizle(veri):
    # Eksik değerleri doldur
    veri.fillna(method='ffill', inplace=True)
    
    # İndeksin zaman dilimi bilgisini kaldır
    if isinstance(veri.index, pd.DatetimeIndex) and veri.index.tz is not None:
        veri.index = veri.index.tz_localize(None)
    
    # Sütun isimlerini kontrol et ve düzelt
    if 'Close' not in veri.columns and 'Adj Close' in veri.columns:
        veri['Close'] = veri['Adj Close']
    
    # Veriyi float tipine çevir
    veri['Close'] = veri['Close'].astype(float)
    
    return veri

# ----------------------- Analysis Functions -----------------------
def altin_genel_analiz(veri):
    st.subheader("Altın Fiyat Analizi")
    
    # Son 1 yıllık veri
    son_yil = veri.iloc[-365:].copy() if len(veri) >= 365 else veri.copy()
    
    # Fiyat trendini gösteren grafik
    fig_trend = px.line(son_yil, y='Close', title='Son 1 Yıllık Altın Fiyat Trendi')
    fig_trend.update_layout(
        xaxis_title="Tarih",
        yaxis_title="Fiyat (USD/ons)",
        height=400
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Fiyat değişim oranları
    col1, col2, col3, col4 = st.columns(4)
    
    # Son bir aylık değişim
    son_ay_degisim = ((son_yil['Close'].iloc[-1] - son_yil['Close'].iloc[-30]) / son_yil['Close'].iloc[-30]) * 100 if len(son_yil) >= 30 else 0
    col1.metric("Son 1 Ay", f"%{son_ay_degisim:.2f}", f"{son_ay_degisim:.2f}%")
    
    # Son 3 aylık değişim
    son_3ay_degisim = ((son_yil['Close'].iloc[-1] - son_yil['Close'].iloc[-90]) / son_yil['Close'].iloc[-90]) * 100 if len(son_yil) >= 90 else 0
    col2.metric("Son 3 Ay", f"%{son_3ay_degisim:.2f}", f"{son_3ay_degisim:.2f}%")
    
    # Son 6 aylık değişim
    son_6ay_degisim = ((son_yil['Close'].iloc[-1] - son_yil['Close'].iloc[-180]) / son_yil['Close'].iloc[-180]) * 100 if len(son_yil) >= 180 else 0
    col3.metric("Son 6 Ay", f"%{son_6ay_degisim:.2f}", f"{son_6ay_degisim:.2f}%")
    
    # Son 1 yıllık değişim
    son_yil_degisim = ((son_yil['Close'].iloc[-1] - son_yil['Close'].iloc[0]) / son_yil['Close'].iloc[0]) * 100
    col4.metric("Son 1 Yıl", f"%{son_yil_degisim:.2f}", f"{son_yil_degisim:.2f}%")
    
    # Volatilite analizi
    st.subheader("Volatilite Analizi")
    
    # Günlük değişim yüzdesi hesapla
    son_yil['Gunluk_Degisim'] = son_yil['Close'].pct_change() * 100
    
    # Volatilite grafiği
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=son_yil.index, y=son_yil['Gunluk_Degisim'], mode='lines', name='Günlük Değişim %'))
    fig_vol.update_layout(
        title='Günlük Fiyat Değişim Yüzdesi',
        xaxis_title='Tarih',
        yaxis_title='Değişim Yüzdesi (%)',
        height=300
    )
    st.plotly_chart(fig_vol, use_container_width=True)
    
    # Hareketli ortalama analizi
    st.subheader("Hareketli Ortalama Analizi")
    
    # 7, 30, 90 günlük hareketli ortalamalar
    son_yil['MA7'] = son_yil['Close'].rolling(window=7).mean()
    son_yil['MA30'] = son_yil['Close'].rolling(window=30).mean()
    son_yil['MA90'] = son_yil['Close'].rolling(window=90).mean()
    
    # Hareketli ortalama grafiği
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=son_yil.index, y=son_yil['Close'], mode='lines', name='Günlük Kapanış'))
    fig_ma.add_trace(go.Scatter(x=son_yil.index, y=son_yil['MA7'], mode='lines', name='7 Günlük Ortalama'))
    fig_ma.add_trace(go.Scatter(x=son_yil.index, y=son_yil['MA30'], mode='lines', name='30 Günlük Ortalama'))
    fig_ma.add_trace(go.Scatter(x=son_yil.index, y=son_yil['MA90'], mode='lines', name='90 Günlük Ortalama'))
    
    fig_ma.update_layout(
        title='Hareketli Ortalamalar',
        xaxis_title='Tarih',
        yaxis_title='Fiyat (USD/ons)',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_ma, use_container_width=True)
    
    # Destek ve direnç analizi
    st.subheader("Destek ve Direnç Seviyeleri")
    
    # Basit destek ve direnç hesaplama
    son_30_gun = son_yil.iloc[-30:]
    
    # Yerel maksimum ve minimumları bul
    direnç_seviyesi = son_30_gun['High'].max()
    destek_seviyesi = son_30_gun['Low'].min()
    son_fiyat = son_yil['Close'].iloc[-1]
    
    # Pivot seviyesi (basit ortalama)
    pivot = (son_30_gun['High'].max() + son_30_gun['Low'].min() + son_30_gun['Close'].iloc[-1]) / 3
    
    # Seviyeleri göster
    col1, col2, col3 = st.columns(3)
    col1.metric("Destek Seviyesi", f"${destek_seviyesi:.2f}")
    col2.metric("Pivot", f"${pivot:.2f}")
    col3.metric("Direnç Seviyesi", f"${direnç_seviyesi:.2f}")
    
    return son_yil

# ----------------------- ARIMA Model Functions -----------------------
def tahmin_modeli_egit(veri, p, d, q, durum_metni):
    try:
        durum_metni.text("Model eğitiliyor...")
        
        # Veri hazırlama
        train_data = veri['Close'].dropna()
        
        # Model eğitimi
        model = ARIMA(train_data, order=(p, d, q))
        model_fit = model.fit()
        
        durum_metni.text("Model başarıyla eğitildi ✓")
        return model_fit
    except Exception as e:
        durum_metni.error(f"Model hatası: {str(e)}")
        return None

def tahmin_yap(model_fit, gun_sayisi, durum_metni):
    durum_metni.text(f"{gun_sayisi} gün için tahmin yapılıyor...")
    try:
        # Model bilgilerini kontrol et
        if model_fit is None:
            durum_metni.error("Model eğitilemedi!")
            return None
        
        # Tahmin yap
        forecast = model_fit.forecast(steps=gun_sayisi)
        
        # Tarihleri hazırla
        start_date = pd.Timestamp.now().normalize()  # Normalize to remove time component
        forecast_dates = pd.date_range(start=start_date, periods=gun_sayisi, freq='D')
        
        # Tahmin serisi oluştur
        forecast_series = pd.Series(forecast, index=forecast_dates)
        
        # Tahminleri kontrol et
        if forecast_series.isna().any():
            durum_metni.warning("Bazı tahmin değerleri hesaplanamadı.")
            forecast_series = forecast_series.fillna(method='ffill')  # NaN değerleri doldur
        
        durum_metni.text("Tahmin işlemi tamamlandı ✓")
        return forecast_series
    except Exception as e:
        durum_metni.error(f"Tahmin hatası: {str(e)}")
        import traceback
        durum_metni.code(traceback.format_exc())  # Tam hata izini göster
        return None

def basit_tahmin_yap(veri, gun_sayisi):
    """ARIMA çalışmadığında basit trend tahmini yapar"""
    # Son 30 günün ortalamasını al
    son_ortalama = veri['Close'].iloc[-30:].mean()
    
    # Son 7 günün eğilimini hesapla
    son_egim = (veri['Close'].iloc[-1] - veri['Close'].iloc[-8]) / 7
    
    # Basit doğrusal tahmin
    tahminler = [veri['Close'].iloc[-1] + son_egim * (i+1) for i in range(gun_sayisi)]
    
    forecast_dates = pd.date_range(start=pd.Timestamp.now().normalize(), periods=gun_sayisi, freq='D')
    return pd.Series(tahminler, index=forecast_dates)

# ----------------------- Main Application -----------------------
def main():
    st.title("Altın Fiyat Analiz ve Tahmin Sistemi")
    
    # Project description
    st.markdown("""
    ### Proje Hakkında
    Bu uygulama, altın fiyatlarını analiz etmek ve tahmin etmek için geliştirilmiştir. 
    Tarihsel altın fiyat verilerini kullanarak fiyat trendlerini inceler ve gelecekteki fiyat hareketlerini öngörmeye çalışır.
    
    **Özellikler:**
    - Tarihsel altın fiyat verilerini otomatik olarak indirme
    - Detaylı fiyat analizi ve grafikleri
    - Hareketli ortalamalar ve volatilite hesaplamaları
    - ARIMA modeli ile fiyat tahmini
    - Görsel grafik ve istatistiksel sonuçlar
    
    **NOT:** Bu uygulama yalnızca eğitim amaçlıdır ve gerçek yatırım kararları için kullanılmamalıdır.
    """)
    
    # Ana sekmeleri oluştur
    tab1, tab2 = st.tabs(["Altın Fiyat Analizi", "Altın Fiyat Tahmini"])
    
    with tab1:
        # Veri yükleme veya indirme
        data = veriyi_yukle()
        if data is None:
            st.info("Yerel veri bulunamadı, yeni veri indiriliyor...")
            data = veri_cikar()
            veriyi_kaydet(data)
        
        # Veriyi temizle
        data = veri_temizle(data)
        
        # Genel analiz
        altin_genel_analiz(data)
    
    with tab2:
        st.subheader("Altın Fiyat Tahmini")
        
        # ARIMA parametreleri
        st.sidebar.header("Tahmin Parametreleri")
        
        p = st.sidebar.slider("p parametresi", 0, 5, 2)
        d = st.sidebar.slider("d parametresi", 0, 2, 1)
        q = st.sidebar.slider("q parametresi", 0, 5, 2)
        
        gun_sayisi = st.sidebar.slider("Tahmin gün sayısı", 7, 60, 30)
        
        # Başlat butonu
        start_button = st.button("TAHMİN YAP", use_container_width=True)
        
        if start_button:
            # Veri yükleme veya indirme
            data = veriyi_yukle()
            if data is None:
                st.info("Yerel veri bulunamadı, yeni veri indiriliyor...")
                data = veri_cikar()
                veriyi_kaydet(data)
            
            # Veriyi temizle
            data = veri_temizle(data)
            
            # Durum mesajları için placeholder
            durum_metni = st.empty()
            ilerleme_cubugu = st.progress(0)
            
            # Model eğitimi
            ilerleme_cubugu.progress(0.3)
            model_fit = tahmin_modeli_egit(data, p, d, q, durum_metni)
            ilerleme_cubugu.progress(0.6)
            
            # Tahmin yap
            if model_fit is not None:
                forecast = tahmin_yap(model_fit, gun_sayisi, durum_metni)
                ilerleme_cubugu.progress(0.9)
                
                if forecast is None or forecast.isna().any():
                    st.warning("ARIMA modeli başarısız oldu, basit tahmin kullanılıyor.")
                    forecast = basit_tahmin_yap(data, gun_sayisi)
            else:
                st.warning("Model eğitilemedi, basit tahmin kullanılıyor.")
                forecast = basit_tahmin_yap(data, gun_sayisi)
            
            ilerleme_cubugu.progress(1.0)
            durum_metni.success("Tahmin tamamlandı!")
            
            # Tahmin sonuçlarını göster
            st.subheader("Altın Fiyat Tahmini")
            
            # Grafik oluştur
            fig = go.Figure()
            
            # Son 90 günlük veriler
            last_90_days = data.iloc[-90:].copy()
            
            # İndeksi kontrol et
            if isinstance(last_90_days.index, pd.DatetimeIndex) and last_90_days.index.tz is not None:
                last_90_days.index = last_90_days.index.tz_localize(None)
            
            # Forecast indexini kontrol et
            if isinstance(forecast.index, pd.DatetimeIndex) and forecast.index.tz is not None:
                forecast.index = forecast.index.tz_localize(None)
            
            # Grafiğe verileri ekle
            fig.add_trace(go.Scatter(x=last_90_days.index, y=last_90_days['Close'], 
                                    name='Tarihsel Veri', line=dict(color='blue')))
            
            fig.add_trace(go.Scatter(x=forecast.index, y=forecast, 
                                    name='Tahmin', line=dict(color='red', dash='dash')))
            
            # Grafik düzeni
            fig.update_layout(
                title='Altın Fiyat Tahmini (ARIMA)',
                xaxis_title='Tarih',
                yaxis_title='Fiyat (USD/ons)',
                hovermode='x unified',
                legend=dict(x=0, y=1, orientation='h')
            )
            
            # Grafiği göster
            st.plotly_chart(fig, use_container_width=True)
            
            # Detaylı bilgiler
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Özet Bilgiler")
                st.write(f"Model: ARIMA({p},{d},{q})")
                st.write(f"Veri miktarı: {len(data)} gün")
                st.write(f"Tahmin yapılan gün sayısı: {gun_sayisi}")
            
            with col2:
                st.subheader("Tahmin Sonucu")
                st.write(f"Son fiyat: ${data['Close'].iloc[-1]:.2f}")
                st.write(f"Tahmin edilen son fiyat: ${forecast.iloc[-1]:.2f}")
                
                # Değişim oranı
                change = ((forecast.iloc[-1] - data['Close'].iloc[-1]) / data['Close'].iloc[-1]) * 100
                st.write(f"Beklenen değişim: %{change:.2f}")
                
                # Yükseliş/düşüş göstergesi
                if change > 0:
                    st.markdown(f"<span style='color:green; font-size:20px;'>Yükseliş bekleniyor ↑</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='color:red; font-size:20px;'>Düşüş bekleniyor ↓</span>", unsafe_allow_html=True)
            
            # Tahmin verilerini tablo olarak göster
            st.subheader("Tahmin Verileri")
            forecast_df = pd.DataFrame({'Tarih': forecast.index.date, 'Tahmin Edilen Fiyat': forecast.values})
            st.dataframe(forecast_df, use_container_width=True)
            
            # CSV indirme butonu
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Tahmin verilerini CSV olarak indir",
                data=csv,
                file_name='altin_fiyat_tahmini.csv',
                mime='text/csv',
            )

if __name__ == "__main__":
    main()