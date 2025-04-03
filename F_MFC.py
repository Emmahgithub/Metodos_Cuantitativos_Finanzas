'''
Este es un Scrit generado para almacenar todas las funciones,
generadas a alolargo de la primera parte del curso de MCF

'''

# Paqueterias necesarias 

# Manejo de datos
import pandas as pd
import numpy as np
 
# Visualizacion de datos
import matplotlib.pyplot as plt

# Api de Yahoo Finanzas
import yfinance as yf

def obtener_datos(stocks):
    '''
    El objetivo de esta funcion es descargar el precio
    de cierre de un o varios activos en una ventana de un año

    Input = Ticker del activo en string 
    Output = DataFrame del precio del activo

    '''
    df = yf.download(stocks, period = "1y")['Close']
    return df

def calcular_rendimientos(df):
    '''
    Funcion de calcula los rendimientos de un activo

    Input = Data Frame de precios por activo

    Output = Data Frame de  rendimientos

    '''
    return df.pct_change().dropna()



#for i in range(len(returns) - window):
        #window_data = returns[i:i + window]
        #mean, std = np.mean(window_data), np.std(window_data)
        

# Lista de acciones de ejemplo
#url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
#df = pd.read_html(url, header=0)[0]  # Extrae la tabla de Wikipedia
#stocks_lista = df['Symbol'].tolist()
#stocks_lista = [ticker.replace('.', '-') for ticker in df['Symbol'].tolist()]
