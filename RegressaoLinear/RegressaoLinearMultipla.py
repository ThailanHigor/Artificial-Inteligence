# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 16:37:20 2019

@author: Thailan

Algoritmo para prever o preco de casas
usando regressão linear multipla com os estimators do Tensorflow

Fórmula da regressão linear simples
y = b0 + b1 * x1 + b2 + * x2 + ... + bn * xn
"""

import pandas as pd

colunas_usadas = ['price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']

base = pd.read_csv('house-prices.csv', usecols = colunas_usadas)

from sklearn.preprocessing import MinMaxScaler
scaler_x = MinMaxScaler()
base[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']] = scaler_x.fit_transform(base[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']])
     
scaler_y = MinMaxScaler()
base[['price']] = scaler_y.fit_transform(base[['price']])

#removido a coluna de preco
X = base.drop('price', axis = 1)
y = base.price
       

previsores_colunas = colunas_usadas[1:17]


import tensorflow as tf
tf = tf.compat.v1


colunas = [tf.feature_column.numeric_column(key = c) for c in previsores_colunas]

from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.3)


funcao_treinamento = tf.estimator.inputs.pandas_input_fn(x = X_treinamento, y = y_treinamento,
                                                        batch_size = 32, num_epochs = None, shuffle = True)

funcao_teste = tf.estimator.inputs.pandas_input_fn(x = X_teste, y = y_teste,
                                                   batch_size = 32, num_epochs = 10000, shuffle = False)

regressor = tf.estimator.LinearRegressor(feature_columns=colunas)

regressor.train(input_fn=funcao_treinamento, steps = 10000)

metricas_treinamento = regressor.evaluate(input_fn=funcao_treinamento, steps = 10000)

metricas_teste = regressor.evaluate(input_fn=funcao_teste, steps = 10000)

print(metricas_treinamento)
print(metricas_teste)

funcao_previsao = tf.estimator.inputs.pandas_input_fn(x = X_teste, shuffle = False)
previsoes = regressor.predict(input_fn=funcao_previsao)

valores_previsoes = []
for p in regressor.predict(input_fn=funcao_previsao):
    valores_previsoes.append(p['predictions'])

import numpy as np
valores_previsoes = np.asarray(valores_previsoes).reshape(-1,1)
valores_previsoes = scaler_y.inverse_transform(valores_previsoes)

print("valores previsoes", valores_previsoes)

y_teste2 = y_teste.values.reshape(-1,1)
y_teste2 = scaler_y.inverse_transform(y_teste2)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste2, valores_previsoes)
print("MAE: ", mae)