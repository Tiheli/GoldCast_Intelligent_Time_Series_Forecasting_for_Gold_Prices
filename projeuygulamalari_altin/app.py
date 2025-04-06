import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import datetime
import time

# Set page title and configuration
st.set_page_config(page_title="Altın Fiyat Tahmin Sistemi", layout="wide")

# ----------------------- Data Management Functions -----------------------
def veri_cikar():
    with st.spinner('Altın fiyat verileri indiriliyor...'):
        gc = yf.Ticker("GC=F")
        veri = gc.history(period="max")
        return veri

def veriyi_kaydet(veri):
    veri.to_csv('altin_fiyatlari.csv')

def veriyi_yukle():
    try:
        return pd.read_csv('altin_fiyatlari.csv', index_col='Date', parse_dates=['Date'])
    except FileNotFoundError:
        return None

def veri_temizle(veri):
    # Eksik değerleri doldur
    veri.fillna(method='ffill', inplace=True)
    return veri

def veri_normallestir(veri):
    # Veriyi normalize et
    return (veri - veri.mean()) / veri.std()

# ----------------------- Model Functions -----------------------
def tahmin_modeli_egit(veri, p, d, q, durum_metni):
    try:
        durum_metni.text("Model eğitiliyor...")
        model = ARIMA(veri['Close'], order=(p, d, q))
        model_fit = model.fit()
        durum_metni.text("Model başarıyla eğitildi ✓")
        return model_fit
    except Exception as e:
        durum_metni.error(f"Model hatası: {str(e)}")
        return None

def model_performans_hesapla(tahminler, gercek_degerler):
    # MAE (Mean Absolute Error) hesapla
    mae = np.mean(np.abs(tahminler - gercek_degerler))
    return mae

def model_optimize_et(veri, durum_metni, ilerleme_cubugu):
    # En iyi p, d, q parametrelerini bul
    durum_metni.text("Model parametreleri optimize ediliyor...")
    en_iyi_mae = float('inf')
    en_iyi_parametreler = (1, 1, 1)  # Default parameters
    
    total_iterations = 9  # 3x3x1 (we're only using d=1 for speed)
    current_iteration = 0
    
    for p in range(1, 4):
        for d in [1]:  # Simplify to just d=1 for speed
            for q in range(1, 4):
                durum_metni.text(f"ARIMA({p},{d},{q}) modeli test ediliyor...")
                try:
                    model = ARIMA(veri['Close'].values, order=(p, d, q))
                    model_fit = model.fit()
                    tahminler = model_fit.forecast(steps=10)
                    test_size = min(10, len(veri) - 10)
                    mae = model_performans_hesapla(tahminler[:test_size], veri['Close'].values[-test_size:])
                    
                    durum_metni.text(f"ARIMA({p},{d},{q}) - MAE: {mae:.2f}")
                    time.sleep(0.2)  # İlerlemeyi görmek için kısa bir bekleme

                    if mae < en_iyi_mae:
                        en_iyi_mae = mae
                        en_iyi_parametreler = (p, d, q)
                except:
                    durum_metni.text(f"ARIMA({p},{d},{q}) - Başarısız")
                    pass
                
                current_iteration += 1
                ilerleme_cubugu.progress(current_iteration / total_iterations)

    durum_metni.text(f"Optimizasyon tamamlandı! En iyi model: ARIMA{en_iyi_parametreler} (MAE: {en_iyi_mae:.2f}) ✓")
    return en_iyi_parametreler, en_iyi_mae

def tahmin_yap(model_fit, gun_sayisi, durum_metni):
    durum_metni.text(f"{gun_sayisi} gün için tahmin yapılıyor...")
    forecast = model_fit.forecast(steps=gun_sayisi)
    forecast_dates = pd.date_range(start=datetime.datetime.now(), periods=gun_sayisi)
    forecast_series = pd.Series(forecast, index=forecast_dates)
    durum_metni.text("Tahmin işlemi tamamlandı ✓")
    return forecast_series

# ----------------------- Main Application -----------------------
def main():
    st.title("Altın Fiyat Tahmin Sistemi")
    
    # Project description
    st.markdown("""
    ### Proje Hakkında
    Bu uygulama, altın fiyatlarını tahmin etmek için ARIMA (AutoRegressive Integrated Moving Average) modelini kullanır. 
    Geçmiş fiyat verilerini analiz ederek gelecekteki fiyat hareketlerini öngörmeye çalışır.
    
    **Özellikler:**
    - Tarihsel altın fiyat verilerini otomatik olarak indirme
    - ARIMA model parametrelerini otomatik optimizasyon
    - İstenen gün sayısı için fiyat tahmini
    - Görsel grafik ve istatistiksel sonuçlar
    
    **NOT:** Bu uygulama yalnızca eğitim amaçlıdır ve gerçek yatırım kararları için kullanılmamalıdır.
    """)
    
    # Sidebar for parameters
    st.sidebar.header("Model Parametreleri")
    optimize_model = st.sidebar.checkbox("Model parametrelerini optimize et", value=True)
    
    if not optimize_model:
        p = st.sidebar.slider("p parametresi", 1, 5, 1)
        d = st.sidebar.slider("d parametresi", 0, 2, 1)
        q = st.sidebar.slider("q parametresi", 1, 5, 1)
    else:
        p, d, q = 1, 1, 1  # Default değerler, optimize edilecek
    
    gun_sayisi = st.sidebar.slider("Tahmin gün sayısı", 7, 60, 30)
    
    # Start button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        start_button = st.button("ANALİZE BAŞLA", use_container_width=True)
    
    if start_button:
        # Create placeholders for status updates
        durum_metni = st.empty()
        ilerleme_cubugu = st.progress(0)
        
        # Execute the main workflow
        run_analysis(optimize_model, p, d, q, gun_sayisi, durum_metni, ilerleme_cubugu)
    else:
        # Show initial instructions
        st.info("Başlamak için 'ANALİZE BAŞLA' butonuna tıklayın.")
        
        # Optional: Display sample chart for visual appeal before starting
        placeholder_fig = go.Figure()
        placeholder_fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3], mode='lines'))
        placeholder_fig.update_layout(
            title='Analize başladıktan sonra burada tahmin grafiğiniz görüntülenecek',
            xaxis_title='Tarih',
            yaxis_title='Fiyat (USD/ons)',
            showlegend=False,
            height=400
        )
        st.plotly_chart(placeholder_fig, use_container_width=True)

def run_analysis(optimize_model, p, d, q, gun_sayisi, durum_metni, ilerleme_cubugu):
    # Step 1: Data loading
    durum_metni.text("Veri yükleniyor...")
    ilerleme_cubugu.progress(0.1)
    
    data = veriyi_yukle()
    if data is None:
        durum_metni.text("Veri bulunamadı, yeni veri indiriliyor...")
        data = veri_cikar()
        veriyi_kaydet(data)
    
    # Step 2: Data cleaning
    durum_metni.text("Veriler temizleniyor ve hazırlanıyor...")
    ilerleme_cubugu.progress(0.2)
    data = veri_temizle(data)
    time.sleep(0.5)  # Simulate processing time
    
    # Step 3: Model optimization if requested
    if optimize_model:
        ilerleme_cubugu.progress(0.3)
        (p, d, q), mae = model_optimize_et(data, durum_metni, ilerleme_cubugu)
        ilerleme_cubugu.progress(0.6)
    else:
        ilerleme_cubugu.progress(0.5)
    
    # Step 4: Train model
    model_fit = tahmin_modeli_egit(data, p, d, q, durum_metni)
    ilerleme_cubugu.progress(0.8)
    
    if model_fit is None:
        durum_metni.error("Model eğitilemedi. Lütfen farklı parametreler deneyin.")
        return
    
    # Step 5: Make forecast
    forecast = tahmin_yap(model_fit, gun_sayisi, durum_metni)
    ilerleme_cubugu.progress(0.9)
    
    # Step 6: Display results
    durum_metni.text("Sonuçlar hazırlanıyor...")
    time.sleep(0.5)  # Simulate processing time
    ilerleme_cubugu.progress(1.0)
    durum_metni.success("Analiz tamamlandı! Sonuçlar aşağıda gösteriliyor.")
    
    # Create tabs for results
    tab1, tab2, tab3 = st.tabs(["Tahmin Grafiği", "Tahmin Verileri", "Model Detayları"])
    
    with tab1:
        st.subheader("Altın Fiyat Tahmini")
        
        # Create plot
        fig = go.Figure()
        
        # Historical data - last 90 days
        last_90_days = data.iloc[-90:]
        fig.add_trace(go.Scatter(x=last_90_days.index, y=last_90_days['Close'], 
                                name='Tarihsel Veri', line=dict(color='blue')))
        
        # Forecast
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast, 
                                name='Tahmin', line=dict(color='red', dash='dash')))
        
        fig.update_layout(
            title='Altın Fiyat Tahmini (ARIMA)',
            xaxis_title='Tarih',
            yaxis_title='Fiyat (USD/ons)',
            hovermode='x unified',
            legend=dict(x=0, y=1, orientation='h')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
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
            change = ((forecast.iloc[-1] - data['Close'].iloc[-1]) / data['Close'].iloc[-1]) * 100
            st.write(f"Beklenen değişim: %{change:.2f}")
            
            if change > 0:
                st.markdown(f"<span style='color:green; font-size:20px;'>Yükseliş bekleniyor ↑</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:red; font-size:20px;'>Düşüş bekleniyor ↓</span>", unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Tahmin Verileri")
        forecast_df = pd.DataFrame({'Tarih': forecast.index, 'Tahmin Edilen Fiyat': forecast.values})
        forecast_df['Tarih'] = forecast_df['Tarih'].dt.date
        st.dataframe(forecast_df, use_container_width=True)
        
        # Add download button
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Tahmin verilerini CSV olarak indir",
            data=csv,
            file_name='altin_fiyat_tahmini.csv',
            mime='text/csv',
        )
    
    with tab3:
        st.subheader("Model Detayları")
        
        # Display model summary
        st.write("### ARIMA Model Özeti")
        st.text(str(model_fit.summary()))
        
        # Historical data graph
        st.write("### Tüm Tarihsel Veri")
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Tarihsel Veri'))
        hist_fig.update_layout(
            title='Altın Fiyat Geçmişi',
            xaxis_title='Tarih',
            yaxis_title='Fiyat (USD/ons)'
        )
        st.plotly_chart(hist_fig, use_container_width=True)

if __name__ == "__main__":
    main()