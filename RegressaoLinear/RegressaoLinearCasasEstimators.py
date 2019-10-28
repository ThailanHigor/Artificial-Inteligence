# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 16:37:20 2019

@author: Thailan

Algoritmo para prever o preco de casas
usando regressão linear simples com os estimators do Tensorflow

Notas:
X = metragem quadrada da casa
y = preço da casa, onde serão feito as previsoes

Fórmula da regressão linear simples
y = b0 + b1 * X
"""

import pandas as pd

base = pd.read_csv('house-prices.csv')

"pegando a coluna metragem quadrada"
"usa-se 5:6 para pegar o lowerbound e já transformar em matriz"
X = base.iloc[:, 5:6].values
y = base.iloc[:, 2:3].values

"escalonando os valores"

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)


import tensorflow as tf
tf = tf.compat.v1

#regressor linear do tensforflow
colunas = [tf.feature_column.numeric_column('x', shape = [1])]
regressor = tf.estimator.LinearRegressor(feature_columns = colunas)

#separando os dados de treinamento e testes em 30%

from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X,y, test_size = 0.3)

funcao_treinamento = tf.estimator.inputs.numpy_input_fn({'x': X_treinamento}, y_treinamento,
                                                        batch_size = 32, num_epochs = None, 
                                                        shuffle = True)

funcao_teste = tf.estimator.inputs.numpy_input_fn({'x': X_teste}, y_teste, batch_size = 32, 
                                                  num_epochs = 1000, shuffle = False)

regressor.train(input_fn = funcao_treinamento, steps = 10000)

metricas_treinamento = regressor.evaluate(input_fn = funcao_treinamento, steps = 10000)
metricas_teste = regressor.evaluate(input_fn = funcao_teste, steps = 10000)

print(metricas_treinamento)
print(metricas_teste)


import numpy as np
novas_casas = np.array([[800],[900],[1000]])
novas_casas = scaler_x.transform(novas_casas)

funcao_previsao = tf.estimator.inputs.numpy_input_fn({'x': novas_casas},
                                                     shuffle = False)
previsoes = regressor.predict(input_fn = funcao_previsao)

for p in previsoes:
    print(scaler_y.inverse_transform(p['predictions']))


