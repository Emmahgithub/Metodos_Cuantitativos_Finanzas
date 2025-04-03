import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis, skew, shapiro ,norm, t
import altair as alt



st.cache_data.clear()


st.title("Calculo de Value-At-Risk y de Expected Shortfall.")

#######################################---BACKEND---##################################################


st.title("Visualización de Rendimientos de Acciones")
# st.write('hola')
@st.cache_data
def obtener_datos(stocks):
    df = yf.download(stocks, start="2010-01-01")['Close']
    return df

@st.cache_data
def calcular_rendimientos(df):
    return df.pct_change().dropna()

def var_es_historico(df_rendimientos, stock_seleccionado, alpha):
    hVaR = df_rendimientos[stock_seleccionado].quantile(1 - alpha)
    ES_hist = df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= hVaR].mean()
    return hVaR, ES_hist

def var_es_parametrico_normal(rendimiento_medio, std_dev, alpha, df_rendimientos, stock_seleccionado):
    VaR_norm = norm.ppf(1 - alpha, rendimiento_medio, std_dev)
    ES_norm = df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= VaR_norm].mean()
    return VaR_norm, ES_norm

def var_es_parametrico_t(rendimiento_medio, std_dev, df_t, alpha, df_rendimientos, stock_seleccionado):
    t_ppf = t.ppf(1 - alpha, df_t)
    VaR_t = rendimiento_medio + std_dev * t_ppf * np.sqrt((df_t - 2) / df_t)
    ES_t = df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= VaR_t].mean()
    return VaR_t, ES_t

def var_es_montecarlo(rendimiento_medio, std_dev, alpha, df_rendimientos, stock_seleccionado, num_sim=10000):
    simulaciones = np.random.normal(rendimiento_medio, std_dev, num_sim)
    VaR_mc = np.percentile(simulaciones, (1 - alpha) * 100)
    ES_mc = df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= VaR_mc].mean()
    return VaR_mc, ES_mc

# Lista de acciones de ejemplo
stocks_lista = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']


with st.spinner("Descargando datos..."):
    df_precios = obtener_datos(stocks_lista)
    df_rendimientos = calcular_rendimientos(df_precios)


#######################################---FRONTEND---##################################################

st.header("Selección de Acción")

st.text("Selecciona una acción de la lista ya que apartir de ella se calculara todo lo que se indica en cada ejercicio")

stock_seleccionado = st.selectbox("Selecciona una acción", stocks_lista)

if stock_seleccionado:
    st.subheader(f"Métricas de Rendimiento: {stock_seleccionado}")
    
    rendimiento_medio = df_rendimientos[stock_seleccionado].mean()
    Kurtosis = kurtosis(df_rendimientos[stock_seleccionado])
    skew = skew(df_rendimientos[stock_seleccionado])
    

    
    col1, col2, col3= st.columns(3)
    col1.metric("Rendimiento Medio Diario", f"{rendimiento_medio:.4%}")
    col2.metric("Kurtosis", f"{Kurtosis:.4}")
    col3.metric("Skew", f"{skew:.2}")

    #Calculo de Value-At-Risk y de Expected Shortfall (historico)

    std_dev = np.std(df_rendimientos[stock_seleccionado])

    # Definir niveles de confianza
    alphas = [0.95, 0.975, 0.99]
    resultados = []
    df_size = df_rendimientos[stock_seleccionado].size
    df_t = df_size - 1  # Grados de libertad para t-Student
    # Calcular VaR y ES para cada nivel de confianza
    for alpha in alphas:
        hVaR, ES_hist = var_es_historico(df_rendimientos, stock_seleccionado, alpha)
        VaR_norm, ES_norm = var_es_parametrico_normal(rendimiento_medio, std_dev, alpha, df_rendimientos, stock_seleccionado)
        VaR_t, ES_t = var_es_parametrico_t(rendimiento_medio, std_dev, df_t, alpha, df_rendimientos, stock_seleccionado)
        VaR_mc, ES_mc = var_es_montecarlo(rendimiento_medio, std_dev, alpha, df_rendimientos, stock_seleccionado)
        
        resultados.append([alpha, hVaR, ES_hist, VaR_norm, ES_norm, VaR_t, ES_t, VaR_mc, ES_mc])

    df_resultados = pd.DataFrame(resultados, columns=["Alpha", "hVaR", "ES_hist", "VaR_Norm", "ES_Norm", "VaR_t", "ES_t", "VaR_MC", "ES_MC"])

    st.subheader("Tabla comparativa de VaR y ES")
    st.text("Esta tabla muestra los resultados de los diferentes métodos de cálculo de VaR y ES")
    st.dataframe(
        df_resultados.set_index("Alpha").style.format("{:.4%}")
        .applymap(lambda _: "background-color: #FFDDC1; color: black;", subset=["hVaR"])  # Durazno 
        .applymap(lambda _: "background-color: #C1E1FF; color: black;", subset=["ES_hist"])  # Azul 
        .applymap(lambda _: "background-color: #B5EAD7; color: black;", subset=["VaR_Norm"])  # Verde 
        .applymap(lambda _: "background-color: #FFB3BA; color: black;", subset=["ES_Norm"])  # Rosa 
        .applymap(lambda _: "background-color: #FFDAC1; color: black;", subset=["VaR_t"])  # Naranja 
        .applymap(lambda _: "background-color: #E2F0CB; color: black;", subset=["ES_t"])  # Verde 
        .applymap(lambda _: "background-color: #D4A5A5; color: black;", subset=["VaR_MC"])  # Rojo 
        .applymap(lambda _: "background-color: #CBAACB; color: black;", subset=["ES_MC"])  # Lila 
    )

    st.subheader("Gráfico de comparación de VaR y ES")
    st.text("Este gráfico muestra la comparación de los diferentes métodos de cálculo de VaR y ES")
    st.bar_chart(df_resultados.set_index("Alpha").T)

    
    ##################################################################################################
    
    #Calculo de VaR y ES con Rolling Window

    st.subheader("Cálculo de VaR y ES con Rolling Window")

    
    ##################################################################################################
    
    #Calculo de VaR y ES con Rolling Window

    st.subheader("Cálculo de VaR y ES con Rolling Window")

    window = 252  # Tamaño de la ventana móvil

    rolling_mean = df_rendimientos[stock_seleccionado].rolling(window).mean()
    rolling_std = df_rendimientos[stock_seleccionado].rolling(window).std()

    
    #Calculamos el valor de VaR_R (Parametrico normal) 95%
    VaRN_R_95 = (norm.ppf(1-0.95, rolling_mean, rolling_std) * 100).round(4)
    VaRN_rolling_df_95 = pd.DataFrame({'Date': df_rendimientos.index, '0.95% VaR Rolling': VaRN_R_95}).set_index('Date')

    #Calculamos el valor para ESN_R (Parametrico) 95%

    ESN_R_95 = df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= VaRN_R_95].mean()
    ESN_rolling_df_95 = pd.DataFrame({'Date': df_rendimientos.index, '0.95% ESN Rolling': ESN_R_95}).set_index('Date')

    #Calculamos el valor para VaRH_R 95%

    VaRH_R_95 = df_rendimientos[stock_seleccionado].rolling(window).quantile(1 - 0.95)
    VaRH_rolling_df_95 = pd.DataFrame({'Date': df_rendimientos.index, '0.95% VaR Rolling': VaRH_R_95}).set_index('Date')

    #Calculamos el valor para ESH_R 95%

    ESH_R_95 = df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= VaRH_R_95].mean()
    ESH_rolling_df_95 = pd.DataFrame({'Date': df_rendimientos.index, '0.95% ESN Rolling': ESH_R_95}).set_index('Date')

###################################################

   #Calculamos el valor de VaR_R (Parametrico normal) 99%
    VaRN_R_99 = norm.ppf(1-0.99, rolling_mean, rolling_std)
    VaRN_rolling_df_99 = pd.DataFrame({'Date': df_rendimientos.index, '0.99% VaR Rolling': VaRN_R_99}).set_index('Date')

    #Calculamos el valor para ESN_R (Parametrico) 99%

    ESN_R_99 = df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= VaRN_R_99].mean()
    ESN_rolling_df_99 = pd.DataFrame({'Date': df_rendimientos.index, '0.99% ESN Rolling': ESN_R_99}).set_index('Date')

    #Calculamos el valor para VaRH_R 99%

    VaRH_R_99 = df_rendimientos[stock_seleccionado].rolling(window).quantile(1 - 0.99)
    VaRH_rolling_df_99 = pd.DataFrame({'Date': df_rendimientos.index, '0.99% VaR Rolling': VaRH_R_99}).set_index('Date')

    #Calculamos el valor para ESH_R 99%

    ESH_R_99 = df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= VaRH_R_99].mean()
    ESH_rolling_df_99 = pd.DataFrame({'Date': df_rendimientos.index, '0.99% ESN Rolling': ESH_R_99}).set_index('Date')

    print(VaRN_rolling_df_95)


    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_rendimientos.index, df_rendimientos[stock_seleccionado] * 100, label='Daily Returns (%)', color='blue', alpha=0.5)
    ax.plot(VaRN_rolling_df_95.index, VaRN_rolling_df_95['0.95% VaR Rolling'], label='0.95% VaR Rolling', color='red')
    ax.set_title('Daily Returns and 0.95% VaR Rolling')
    ax.set_xlabel('Date')
    ax.set_ylabel('Values (%)')
    ax.legend()
    st.pyplot(fig)
