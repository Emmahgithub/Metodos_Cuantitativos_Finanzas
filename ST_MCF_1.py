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
#inciso a), b), c)

st.title("Visualización de Rendimientos de Acciones")
# st.write('hola')
@st.cache_data
def obtener_datos(stocks):
    df = yf.download(stocks, start="2010-01-01")['Close']
    return df

@st.cache_data
def calcular_rendimientos(df):
    return df.pct_change().dropna()



#####################################################################################################################



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

#########################################################################################################################

# Expected Shortfall (ES) Rolling - Paramétrico Normal al 0.95% (Esto es para el inciso d)) 
def calcular_es_normal_r_95(rendimientos):
    if len(rendimientos) < window:
        return np.nan
    var = norm.ppf(1 - 0.95, rendimientos.mean(), rendimientos.std())
    return rendimientos[rendimientos <= var].mean()
# Expected Shortfall (ES) Rolling - Paramétrico Normal al 0.99% (Esto es para el inciso d))

def calcular_es_normal_r_99(rendimientos):
    if len(rendimientos) < window:
        return np.nan
    var = norm.ppf(1 - 0.99, rendimientos.mean(), rendimientos.std())
    return rendimientos[rendimientos <= var].mean()

# Expected Shortfall (ES) Rolling - Histórico al 95% 
def calcular_es_historico_r_95(rendimientos):
    rendimientos = pd.Series(rendimientos)  # Convertir a Pandas Series
    if len(rendimientos) < window:
        return np.nan
    var = rendimientos.quantile(1 - 0.95)
    return rendimientos[rendimientos <= var].mean()

# Expected Shortfall (ES) Rolling - Histórico al 99%
def calcular_es_historico_r_99(rendimientos):
    rendimientos = pd.Series(rendimientos)
    if len(rendimientos) < window:
        return np.nan
    var = rendimientos.quantile(1 - 0.99)
    return rendimientos[rendimientos <= var].mean()
#################################################################################################################33
#inciso e)

def calcular_violaciones_var(df_rendimientos, stock_seleccionado, var_dict):
    resultados = {}
    total_observaciones = len(df_rendimientos)

    for metodo, var_series in var_dict.items():
        violaciones = (df_rendimientos[stock_seleccionado] < var_series).sum()
        porcentaje_violaciones = (violaciones / total_observaciones) * 100
        resultados[metodo] = (violaciones, porcentaje_violaciones)
    
    return resultados




###################################################################################################################
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

    #GRAFICA AGREGADA AL FINAL CUANDO MIS HUEVOS PELIGRABAN DE LOS RENDIMIENTOS DIARIOS PA Q SE VEA BONITO

    st.subheader("Gráfico de Rendimientos Diarios") #oliwis :)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_rendimientos.index, df_rendimientos[stock_seleccionado] * 100, label='rendimientos (%)', color='blue', alpha=0.5)
    ax.set_xlabel('fecha')
    ax.set_ylabel('porcentajes (%)')
    ax.legend()
    st.pyplot(fig)

    # Calcular rendimientos logarítmicos 
    #Aaaaaaaaaaaaaaaa pero queria que se viera bonito


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



    window = 252  # Tamaño de la ventana móvil


    rolling_mean = df_rendimientos[stock_seleccionado].rolling(window).mean()
    rolling_std = df_rendimientos[stock_seleccionado].rolling(window).std()

    
    #Calculamos el valor de VaR_R (Parametrico normal) 95%
    VaRN_R_95 = norm.ppf(1-0.95, rolling_mean, rolling_std) 
    VaRN_rolling_df_95 = pd.DataFrame({'Date': df_rendimientos.index, '0.95% VaRN Rolling': VaRN_R_95}).set_index('Date')

    #Calculamos el valor para ESN_R (Parametrico) 95%

    ESN_R_95 =  df_rendimientos[stock_seleccionado].rolling(window).apply(calcular_es_normal_r_95, raw=True)
    ESN_rolling_df_95 = pd.DataFrame({'Date': df_rendimientos.index, '0.95% ESN Rolling': ESN_R_95}).set_index('Date')

    #Calculamos el valor para VaRH_R 95%

    VaRH_R_95 = df_rendimientos[stock_seleccionado].rolling(window).quantile(1 - 0.95)
    VaRH_rolling_df_95 = pd.DataFrame({'Date': df_rendimientos.index, '0.95% VaRH Rolling': VaRH_R_95}).set_index('Date')

    #Calculamos el valor para ESH_R 95%


    ESH_R_95 = df_rendimientos[stock_seleccionado].rolling(window).apply(calcular_es_historico_r_95, raw=True)
    ESH_rolling_df_95 = pd.DataFrame({'Date': df_rendimientos.index, '0.95% ESH Rolling': ESH_R_95}).set_index('Date')

################################################### Esta mamada de parte como la oodie ojala se retuersan en sus tumbas las personas que hicieron esto

   #Calculamos el valor de VaR_R (Parametrico normal) 99%
    VaRN_R_99 = norm.ppf(1-0.99, rolling_mean, rolling_std)
    VaRN_rolling_df_99 = pd.DataFrame({'Date': df_rendimientos.index, '0.99% VaRN Rolling': VaRN_R_99}).set_index('Date')

    #Calculamos el valor para ESN_R (Parametrico) 99%

    ESN_R_99 = df_rendimientos[stock_seleccionado].rolling(window).apply(calcular_es_normal_r_99, raw=True)
    ESN_rolling_df_99 = pd.DataFrame({'Date': df_rendimientos.index, '0.99% ESN Rolling': ESN_R_99}).set_index('Date')

    #Calculamos el valor para VaRH_R 99%

    VaRH_R_99 = df_rendimientos[stock_seleccionado].rolling(window).quantile(1 - 0.99)
    VaRH_rolling_df_99 = pd.DataFrame({'Date': df_rendimientos.index, '0.99% VaRH Rolling': VaRH_R_99}).set_index('Date')

    #Calculamos el valor para ESH_R 99%

    ESH_R_99 = df_rendimientos[stock_seleccionado].rolling(window).apply(calcular_es_historico_r_99, raw=True) #mira nomas esta mmda estas mmdaassss de ESSSSSSSs me tocaron mis huevoossss me queria colgar ahora no pyedo leer ES pq me quiero colgar de los huevos
    ESH_rolling_df_99 = pd.DataFrame({'Date': df_rendimientos.index, '0.99% ESH Rolling': ESH_R_99}).set_index('Date')


    st.subheader("Gráficos del VaR y ES con Rolling Window al 95% y 99% (Parametrico (Normal) y Historico)")

    st.text("Acontinuacion observaremos los resultados del VaR parametrico (Normal) como tambien el historico al 99% y al 95%")

    # Graficamos los resultados de VaR y ES con Rolling Window al 95%


    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_rendimientos.index, df_rendimientos[stock_seleccionado] * 100, label='Retornos Diarios (%)', color='blue', alpha=0.5)
    ax.plot(VaRN_rolling_df_95.index, VaRN_rolling_df_95['0.95% VaRN Rolling'] *100, label='0.95% VaRN Rolling', color='green')
    ax.plot(VaRH_rolling_df_95.index, VaRH_rolling_df_95['0.95% VaRH Rolling'] *100, label='0.95% VaRH Rolling', color='red')
    ax.plot(VaRN_rolling_df_99.index, VaRN_rolling_df_99['0.99% VaRN Rolling'] *100, label='0.99% VaRN Rolling', color='blue')
    ax.plot(VaRH_rolling_df_99.index, VaRH_rolling_df_99['0.99% VaRH Rolling'] *100, label='0.99% VaRH Rolling', color='orange')
    ax.set_title('Retornos diaros, 0.95% VaR Rolling y 0.95% ESN Rolling')
    ax.set_xlabel('fehca')
    ax.set_ylabel('procentaje (%)')
    ax.legend()
    st.pyplot(fig)


#inga tu roña mira q grafica tan bonita pensar q me tomo 3 dias hacerla y entender pq daba error me tomo solo mi salud mental

    st.text("Acontinuacion onbservaremos los resultados del ES parametrico (Normal) como tambien el historico al 99% y al 95%")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_rendimientos.index, df_rendimientos[stock_seleccionado] * 100, label='Retornos Diarios (%)', color='blue', alpha=0.5)
    ax.plot(ESN_rolling_df_95.index, ESN_rolling_df_95['0.95% ESN Rolling'] *100, label='0.95% ESN Rolling', color='green')
    ax.plot(ESH_rolling_df_95.index, ESH_rolling_df_95['0.95% ESH Rolling'] *100, label='0.95% ESH Rolling', color='red')
    ax.plot(ESN_rolling_df_99.index, ESN_rolling_df_99['0.99% ESN Rolling'] *100, label='0.99% ESN Rolling', color='blue')
    ax.plot(ESH_rolling_df_99.index, ESH_rolling_df_99['0.99% ESH Rolling'] *100, label='0.99% ESH Rolling', color='orange')
    ax.set_title('Retornos diaros, 0.95% VaR Rolling y 0.95% ESN Rolling')
    ax.set_xlabel('fecha')
    ax.set_ylabel('porcentajes (%)')
    ax.legend()
    st.pyplot(fig)


    #################################################################### vtl aqui casi me cago

    # Calculo de violaciones de VaR y ES con Rolling Window

    st.header("Cálculo de Violaciones de VaR y ES con Rolling Window")
    st.text("Acontinuacion se calcularan las violaciones de los resultados obtenidos anteriormente es decir calcularemos el porcentaje de violaciones que hubo en cada una de las medidas de riesgo que se calcularon con rolling window")
    
    var_dict = {#como odio los diccionarios
        "VaR Normal 95%": VaRN_rolling_df_95['0.95% VaRN Rolling'],
        "ES Normal 95%": ESN_rolling_df_95['0.95% ESN Rolling'],
        "VaR Histórico 95%": VaRH_rolling_df_95['0.95% VaRH Rolling'],
        "ES Histórico 95%": ESH_rolling_df_95['0.95% ESH Rolling'],
        "VaR Normal 99%": VaRN_rolling_df_99['0.99% VaRN Rolling'],
        "ES Normal 99%": ESN_rolling_df_99['0.99% ESN Rolling'],
        "VaR Histórico 99%": VaRH_rolling_df_99['0.99% VaRH Rolling'],
        "ES Histórico 99%": ESH_rolling_df_99['0.99% ESH Rolling'],
    }

    resultados_var = calcular_violaciones_var(df_rendimientos, stock_seleccionado, var_dict)

    for metodo, (violaciones, porcentaje) in resultados_var.items():
        st.text(f"{metodo}: {violaciones} violaciones ({porcentaje:.2f}%)")



    ############################################################################################### jajajaj q cagdo pongo esto pa separar y creo q se ve mas ogt

    st.subheader("Cálculo de VaR con Volatilidad Móvil")

    # Percentiles para la distribución normal estándar
    q_5 = norm.ppf(0.05)  # Para α = 0.05
    q_1 = norm.ppf(0.01)  # Para α = 0.01

    # Calcular el VaR con volatilidad móvil (mas me muevo yo mientras me retuerso)
    VaR_vol_95 = q_5 * rolling_std
    VaR_vol_99 = q_1 * rolling_std

    # Convertir a DataFrame para graficar si dios existe pq no estoy con el
    VaR_vol_df = pd.DataFrame({
        'Date': df_rendimientos.index,
        'VaR_vol_95': VaR_vol_95,
        'VaR_vol_99': VaR_vol_99
    }).set_index('Date')

    # Graficar odio odio odio
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_rendimientos.index, df_rendimientos[stock_seleccionado] * 100, label='Retornos Diarios (%)', color='blue', alpha=0.5)
    ax.plot(VaR_vol_df.index, VaR_vol_df['VaR_vol_95'] * 100, label='VaR 95% (Vol Movil)', color='green')
    ax.plot(VaR_vol_df.index, VaR_vol_df['VaR_vol_99'] * 100, label='VaR 99% (Vol Movil)', color='red')
    ax.set_title(f'VaR con Volatilidad Móvil para {stock_seleccionado}')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('VaR (%)')
    ax.legend()
    st.pyplot(fig)

    #Ya solo faltan las violaciones pero la neta q flojera, si las voy a hacer pero la neta q coraje me quurisad colgar d lis guevoosssssssssss

    # Calcular violaciones mas violado me senti yo haciendo esto :))))
    var_dict2 = {
    "VaR Volatilidad Móvil 95%": VaR_vol_df["VaR_vol_95"],
    "VaR Volatilidad Móvil 99%": VaR_vol_df["VaR_vol_99"]
    }

    resultados_var2 = calcular_violaciones_var(df_rendimientos, stock_seleccionado, var_dict2)


    for metodo, (violaciones, porcentaje) in resultados_var2.items():
        st.text(f"{metodo}: {violaciones} violaciones ({porcentaje:.2f}%)")