import pandas as pd
import numpy as np
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
from datetime import datetime
## pip install pmdarima
from pmdarima.arima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose

base = pd.read_csv("C:/Users/lcenci1/Downloads/BTC-USD_2.csv")
base

dateParse = lambda dates: datetime.strptime(dates, '%Y-%m-%d')
base = pd.read_csv("C:/Users/lcenci1/Downloads/BTC-USD_2.csv", parse_dates = ['Data'],
                   index_col = 'Data', date_parser = dateParse)




base.index

ts = base['Volume']
ts

ts[1]

ts['2023-01']

ts[datetime(2022,3,2)]

ts['2021-01-01':'2023-01-31']

ts[:'2023-01-31']

ts['2023']

ts.index.max()
ts.index.min()

plt.plot(ts)

#Visualizacao por mes
ts_mes = ts.groupby([lambda x: x.month]).sum()
plt.plot(ts_mes)

ts_datas = ts['2022-01-01':'2022-12-31']
plt.plot(ts_datas)

## decomposicao da série temporal
decomposicao = seasonal_decompose(ts)
decomposicao

## tendencia da Serie temporal
tendencia = decomposicao.trend
tendencia

## sazonalidade
sazonal = decomposicao.seasonal
sazonal

##Aleatoriedade
aleatorio = decomposicao.resid
aleatorio

### Visualizacao de dados
## sazonalidade
plt.plot(sazonal)

## tendencia
plt.plot(tendencia)

##Erro ou Aleatoriedade
plt.plot(aleatorio)

### Ajustar todos em Somente um gráfico
## Série Temporal

plt.subplot(4,1,1)
plt.plot(ts,label = 'Original')
plt.legend(loc = 'best')

##Visulalizacão somente da Tendência

plt.subplot(4,1,2)
plt.plot(tendencia,label = 'Tendência')
plt.legend(loc = 'best')

## Visualizacão somente da sazonalidade

plt.subplot(4,1,3)
plt.plot(sazonal,label = 'Sazonalidade')
plt.legend(loc = 'best')

## Visualizacao da Aleatório

plt.subplot(4,1,4)
plt.plot(aleatorio,label = 'Aleatório')
plt.legend(loc = 'best')
plt.tight_layout()


stepwise_model = auto_arima(base,start_p=1, start_q=1,start_d=0,start_P=1,max_p=6,max_q=6,m=7,seazonal=True,trace = True, stepwise=False)

print(stepwise_model.aic())


treino = base.loc['2014-09-17':'2021-02-03']
treino

teste = base.loc['2021-02-04':]
teste

stepwise_model.fit(treino)

future_forecast = stepwise_model.predict(n_periods=778)
future_forecast

future_forecast = pd.DataFrame(future_forecast,index=teste.index,columns=["Volume"])
future_forecast

pd.concat([teste,future_forecast],axis=1).plot()

pd.concat([base,future_forecast],axis=1).plot(linewidth=3)

