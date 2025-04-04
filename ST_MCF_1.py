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
    return hVaR *(-1), ES_hist *(-1)

def var_es_parametrico_normal(rendimiento_medio, std_dev, alpha, df_rendimientos, stock_seleccionado):
    VaR_norm = norm.ppf(1 - alpha, rendimiento_medio, std_dev)
    ES_norm = df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= VaR_norm].mean()
    return VaR_norm*(-1), ES_norm *(-1)

def var_es_parametrico_t(rendimiento_medio, std_dev, df_t, alpha, df_rendimientos, stock_seleccionado):
    t_ppf = t.ppf(1 - alpha, df_t)
    VaR_t = rendimiento_medio + std_dev * t_ppf * np.sqrt((df_t - 2) / df_t)
    ES_t = df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= VaR_t].mean()
    return VaR_t*(-1), ES_t *(-1)

def var_es_montecarlo(rendimiento_medio, std_dev, alpha, df_rendimientos, stock_seleccionado, num_sim=10000):
    simulaciones = np.random.normal(rendimiento_medio, std_dev, num_sim)
    VaR_mc = np.percentile(simulaciones, (1 - alpha) * 100)
    ES_mc = df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= VaR_mc].mean()
    return VaR_mc*(-1), ES_mc*(-1)

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

    for metodo, var_series in var_dict.items():
        serie_aligned = var_series.reindex(df_rendimientos.index)
        n_valid = serie_aligned.notna().sum()
        violaciones = (df_rendimientos[stock_seleccionado] < serie_aligned).sum()
        porcentaje_violaciones = ((violaciones / n_valid) * 100 ) if n_valid > 0 else 0.0
        resultados[metodo] = (violaciones, porcentaje_violaciones)
    
    return resultados

def color_porcentaje(val):
        color = 'red' if val > 2.5 else 'green'
        return f'color: {color}; font-weight: bold'
###################################################################################################################

# Lista de acciones de ejemplo
stocks_lista = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN','SPY']


with st.spinner("Descargando datos..."):
    df_precios = obtener_datos(stocks_lista)
    df_rendimientos = calcular_rendimientos(df_precios)


#######################################---FRONTEND---##################################################

st.header("Selección de Acción")

st.text("Selecciona una acción de la lista ya que a esta se le aplicarán las métricas de riesgo, además de su robustes de los modelos a traves del número de violaciones")

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


    st.subheader("Gráfico de Rendimientos Diarios") #oliwis :)

    chart = alt.Chart(df_rendimientos[stock_seleccionado].reset_index()).mark_line(color='white', opacity=0.5).encode(
        x=alt.X('Date', title='Fecha'),
        y=alt.Y(f'{stock_seleccionado}', axis=alt.Axis(format='%', title='Rendimiento (%)')),
        tooltip=[alt.Tooltip('Date', title='Fecha'), 
             alt.Tooltip(f'{stock_seleccionado}', format='.2%', title='Rendimiento')]
    ).properties(
        width=800,
        height=400,
        title=f'Rendimientos Diarios de {stock_seleccionado}'
    )

    st.altair_chart(chart, use_container_width=True)


    # Calcular rendimientos logarítmicos 

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
    st.text("Sin profundizar en cada método, el ES bajo la t de Student (ES_t) se posiciona como el estimador de riesgo más robusto, al capturar " \
    "adecuadamente las pérdidas extremas en activos volátiles. Aunque puede sobreestimar en algunos casos, esto se debe a la alta volatilidad de " \
    "los últimos 10 años, por lo que una mejora sería acotar el historial a un rango más reciente, como 5 años.") 

    st.text("En contraste, los demás métodos tienden a subestimar o sobreestimar el riesgo según la forma de los retornos y los eventos incluidos. El método histórico es el que más tiende " \
    "a sobreestimar el riesgo, debido a la inclusión de eventos extremos poco representativos del contexto actual, como crisis financieras o la pandemia, " \
    "lo que lo convierte en el menos confiable de los analizados.")

    st.text("Solo el modelo normal se aproxima en algunos escenarios al de la t dado su parentesco matemático." \
    "En resumen, ES_t destaca como el modelo más coherente para una gestión de riesgo prudente, siempre que se ajuste adecuadamente el periodo histórico a analizar.")
    
    ###############################################################################################################################
#inciso d)
    
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

    st.subheader("Gráficos del VaR y ES con Rolling Window al 95% y 99% (Paramétrico (Normal) e Histórico)")

    st.text("A continuación observaremos los resultados del VaR Paramétrico (Normal) como también el histórico al 99% y al 95%.")

    # Graficamos los resultados de VaR y ES con Rolling Window al 95%

    # Preparar datos combinados
    df_var = pd.concat([
        VaRN_rolling_df_95.rename(columns={'0.95% VaRN Rolling': 'value'}).assign(Metrica='0.95% VaRN Rolling'),
        VaRH_rolling_df_95.rename(columns={'0.95% VaRH Rolling': 'value'}).assign(Metrica='0.95% VaRH Rolling'),
        VaRN_rolling_df_99.rename(columns={'0.99% VaRN Rolling': 'value'}).assign(Metrica='0.99% VaRN Rolling'),
        VaRH_rolling_df_99.rename(columns={'0.99% VaRH Rolling': 'value'}).assign(Metrica='0.99% VaRH Rolling')
    ]).reset_index()

    # Convertir a porcentaje y limpiar datos
    df_var = df_var.dropna()

    # Dataframe de rendimientos (convertir índice a columna 'Date')
    df_rend_plot = df_rendimientos.reset_index().rename(columns={'index': 'Date'})

    # Convertir a porcentaje
    df_rendimientos_plot = df_rendimientos.reset_index()

    # Crear la gráfica base
    base = alt.Chart(df_rendimientos_plot).mark_line(
        color='white',
        opacity=0.5,
    ).encode(
        x=alt.X('Date', title='Fecha'),
        y=alt.Y(f'{stock_seleccionado}', axis=alt.Axis(format='%', title='Rendimiento (%)')
    ))

    # Capa de VaR

    var_layer = alt.Chart(df_var).mark_line(
        strokeWidth=2
    ).encode(
        x='Date',
        y=alt.Y('value', title='VaR (%)'),
        color=alt.Color('Metrica', scale=alt.Scale(
            domain=['0.95% VaRN Rolling', '0.95% VaRH Rolling', '0.99% VaRN Rolling', '0.99% VaRH Rolling'],
            range=['#4daf4a', '#e41a1c', '#377eb8', '#ff7f00']  # Verde, Rojo, Azul, Naranja
        )),
        tooltip=[
            alt.Tooltip('Date', title='Fecha'),
            alt.Tooltip('value', title='VaR', format='%',),
            alt.Tooltip('Metrica', title='Métrica')
        ]
    )

    # Combinar y personalizar
    chart = (base + var_layer).properties(
        title=f'VaR con Volatilidad Móvil - {stock_seleccionado}',
        width=800,
        height=400
    ).configure_legend(
        title=None,
        orient='bottom',
        labelFontSize=12,
        symbolStrokeWidth=6,
        padding=10
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_view(
        strokeWidth=0
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

    st.text("Tras analizar las gráficas de VaR para AAPL, MSFT, GOOGL, TSLA, AMZN y SPY, se observa que el VaR Histórico (VaRH) tiende a sobreestimar " \
    "el riesgo en comparación con el VaR Paramétrico (VaRN). Esto se debe a la inclusión de eventos extremos en el historial de datos, como la crisis " \
    "financiera de 2008 y la pandemia de COVID-19, que distorsionan la estimación del riesgo actual.")

    st.text("En particular, TSLA y AMZN muestran una mayor sensibilidad a estos eventos extremos en el VaRH, lo que sugiere que sus perfiles de riesgo son " \
    "más propensos a ser impactados por shocks de mercado. Esto se refleja en los picos más pronunciados en sus gráficas de VaRH durante la crisis de 2020. ")

    st.text("Por otro lado, AAPL, MSFT y GOOGL exhiben una mayor estabilidad en sus estimaciones de VaR, con una menor discrepancia entre el VaRH y el VaRN. Esto indica " \
    "que sus perfiles de riesgo son menos sensibles a los eventos extremos y que el modelo paramétrico normal se aproxima más a la realidad.")

    st.text("A continuación observaremos los resultados del ES paramétrico (Normal) como también el historico al 99% y al 95%.")
    
    # Preparar los datos combinados
    df_es = pd.concat([
        ESN_rolling_df_95.rename(columns={'0.95% ESN Rolling': 'value'}).assign(Metrica='0.95% ESN Rolling'),
        ESH_rolling_df_95.rename(columns={'0.95% ESH Rolling': 'value'}).assign(Metrica='0.95% ESH Rolling'),
        ESN_rolling_df_99.rename(columns={'0.99% ESN Rolling': 'value'}).assign(Metrica='0.99% ESN Rolling'),
        ESH_rolling_df_99.rename(columns={'0.99% ESH Rolling': 'value'}).assign(Metrica='0.99% ESH Rolling')
    ]).reset_index()

    # Capa de ES
    es_layer = alt.Chart(df_es.dropna()).mark_line(
        strokeWidth=2
    ).encode(
        x='Date',
        y=alt.Y('value', title='ES (%)'),
        color=alt.Color('Metrica', scale=alt.Scale(
            domain=['0.95% ESN Rolling', '0.95% ESH Rolling', '0.99% ESN Rolling', '0.99% ESH Rolling'],
            range=['#4daf4a', '#e41a1c', '#377eb8', '#ff7f00']  # Verde, Rojo, Azul, Naranja
        )),
        tooltip=[
            alt.Tooltip('Date', title='Fecha'),
            alt.Tooltip('value', title='ES', format='%'),
            alt.Tooltip('Metrica', title='Métrica')
        ]
    ).properties(
        width=800,
        height=400
    )

    # Combinar las capas

    chart = (base + es_layer).properties(
        title=f'VaR con Volatilidad Móvil - {stock_seleccionado}',
        width=800,
        height=400
    ).configure_legend(
        title=None,
        orient='bottom',
        labelFontSize=12,
        symbolStrokeWidth=6,
        padding=10
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_view(
        strokeWidth=0
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

    ###################################################################################################################################
#inciso e)

    # Cálculo de violaciones de VaR y ES con Rolling Window

    st.header("Cálculo de Violaciones de VaR y ES con Rolling Window")
    st.text("A continuación se calcularán las violaciones de los resultados obtenidos anteriormente, es decir, calcularemos el porcentaje de violaciones que hubo en cada una de las medidas de riesgo que se calcularón con Rolling Window.")
    
    var_dict ={
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

    df_resultados = pd.DataFrame.from_dict(resultados_var, orient='index', columns=['Violaciones', 'Porcentaje (%)']).reset_index()
    df_resultados[['Método', 'Nivel de Confianza']] = df_resultados['index'].str.rsplit(' ', n=1, expand=True)
    df_resultados = df_resultados[['Método', 'Nivel de Confianza', 'Violaciones', 'Porcentaje (%)']]

    # Mostrar tabla con estilo

    st.dataframe(
    df_resultados.style
    .format({'Porcentaje (%)': '{:.2f}%', 'Observaciones Válidas': '{:.0f}'})
    .applymap(color_porcentaje, subset=['Porcentaje (%)']),
    hide_index=True  # Ocultar la primera columna de índice
    )

    ##############################################################################################################################
#inciso f)

    st.subheader("Cálculo de VaR con Volatilidad Móvil")

    # Percentiles para la distribución normal estándar

    q_5 = norm.ppf(0.05)  # Para α = 0.05
    q_1 = norm.ppf(0.01)  # Para α = 0.01

    # Calcular el VaR con volatilidad móvil 

    VaR_vol_95 = q_5 * rolling_std
    VaR_vol_99 = q_1 * rolling_std

    # Convertir a DataFrame para graficar 

    VaR_vol_df = pd.DataFrame({
        'Date': df_rendimientos.index,
        'VaR_vol_95': VaR_vol_95,
        'VaR_vol_99': VaR_vol_99
    }).set_index('Date')

    # Preparar datos

    df_rend_plot = df_rendimientos.reset_index().rename(columns={'index': 'Date'})

    # Preparar datos de VaR

    df_var_vol = VaR_vol_df.reset_index().melt(id_vars='Date', 
                                            value_vars=['VaR_vol_95', 'VaR_vol_99'],
                                            var_name='Métrica', 
                                            value_name='Valor')
    df_var_vol['Métrica'] = df_var_vol['Métrica'].replace({
        'VaR_vol_95': 'VaR 95% (Vol Móvil)',
        'VaR_vol_99': 'VaR 99% (Vol Móvil)'
    })

    # Capa de VaR

    var_layer = alt.Chart(df_var_vol.dropna()).mark_line(
        strokeWidth=2
    ).encode(
        x='Date',
        y=alt.Y('Valor', title='VaR (%)'),
        color=alt.Color('Métrica', scale=alt.Scale(
            domain=['VaR 95% (Vol Móvil)', 'VaR 99% (Vol Móvil)'],
            range=['#4682B4', '#FFD700']  # Azul, DoradoD
        )),
        tooltip=[
            alt.Tooltip('Date', title='Fecha', format='%Y-%m-%d'),
            alt.Tooltip('Valor', title='VaR', format='%'),
            alt.Tooltip('Métrica', title='Nivel de Confianza')
        ]
    )

    # Combinar y personalizar

    chart = (base + var_layer).properties(
        title=f'VaR con Volatilidad Móvil - {stock_seleccionado}',
        width=800,
        height=400
    ).configure_legend(
        title=None,
        orient='bottom',
        labelFontSize=12,
        symbolStrokeWidth=6,
        padding=10
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_view(
        strokeWidth=0
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

    # Calcular violaciones 

    var_dict2 = {
    "VaR Volatilidad Móvil 95%": VaR_vol_df["VaR_vol_95"],
    "VaR Volatilidad Móvil 99%": VaR_vol_df["VaR_vol_99"]
    }

    resultados_var2 = calcular_violaciones_var(df_rendimientos, stock_seleccionado, var_dict2)

    for metodo, (violaciones, porcentaje) in resultados_var2.items():
        st.text(f"{metodo}: {violaciones} violaciones ({porcentaje:.2f}%)")


    df_resultados2 = pd.DataFrame.from_dict(resultados_var2, orient='index', columns=['Violaciones', 'Porcentaje (%)']).reset_index()
    df_resultados2[['Método', 'Nivel de Confianza']] = df_resultados2['index'].str.rsplit(' ', n=1, expand=True)
    df_resultados2 = df_resultados2[['Método', 'Nivel de Confianza', 'Violaciones', 'Porcentaje (%)']]

    st.dataframe(
    df_resultados2.style
    .format({'Porcentaje (%)': '{:.2f}%', 'Observaciones Válidas': '{:.0f}'})
    .applymap(color_porcentaje, subset=['Porcentaje (%)']),
    hide_index=True  # Ocultar la primera columna de índice
    )
