# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:15:49 2023

@author: Luan Cenci

## Prever a serie temporal com o Prophet

Baseado no artigo :"https://towardsdatascience.com/the-complete-guide-to-time-series-analysis-and-forecasting-70d476bfe775"

"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

%matplotlib inline

from prophet import Prophet
import logging

## Carga dos Dados

data = pd.read_csv("https://raw.githubusercontent.com/LuanCenci/dados_portfolio/main/EMBR3.SA.csv")
data.head()

## Verificar colunas e aportuguesa-las

data.columns

data.columns = ['dia','abertura','alta','baixa','fechamento','fechamento_ajustado','volume']

data.columns

## Modelagem

# eliminar colunas desncessárias

eliminar = ['abertura', 'alta', 'baixa','fechamento_ajustado','volume']

data = data.drop(eliminar,axis=1)

data.head()

# ajustar os valores para o algoritmo Prophet

df = data

df

df.columns = ['ds','y']

df.head()

#Treinamento do Modelo

tamanho = 30

train_df = df[:-tamanho]

train_df.shape

# Inicializando instancia e execuatando o treinamento

m = Prophet()

m.fit(train_df)

futuro = m.make_future_dataframe(periods=tamanho)
forecast = m. predict(futuro)
forecast.head()

m.plot(forecast)

m.plot_components(forecast)

# Comparando o modelo com a Serie 

def make_comparison_dataframe(historical,forecast):
    return forecast.set_index('ds')[['yhat','yhat_lower','yhat_upper']].join(historical.set_index('ds'))

cmp_df = make_comparison_dataframe(df, forecast)

cmp_df.head()

## Calculando os Erros

def calculate_forecast_errors(df, prediction_size):
    df = df.copy()
    
    df['e'] = df['y'] - df['yhat']
    df['p'] = 100 * df['e'] / df['y']
    
    predicted_part = df[-prediction_size:]
    
    error_mean = lambda error_name: np.mean(np.abs(predicted_part[error_name]))
    
    return{'MAPE': error_mean('p'), 'MAE': error_mean('e')}    

calculate_forecast_errors(cmp_df,tamanho)

for err_name, err_value in calculate_forecast_errors(cmp_df, prediction_size=tamanho).items():
    print(err_name, err_value)


# Grafico da Previsão, com o limite superior e inferior

plt.figure(figsize=(17,8))
plt.plot(cmp_df['yhat'])
plt.plot(cmp_df['yhat_lower'])
plt.plot(cmp_df['yhat_upper'])
plt.plot(cmp_df['y'])
plt.xlabel('Tempo')
plt.ylabel('Medias')
plt.grid(False)
plt.show()









