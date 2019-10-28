# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:43:24 2019

@author: Thailan

Algoritmo para prever o preco de casas
usando regressão linear simples

Notas:
X = metragem quadrada da casa
y = preço da casa, onde serão feito as previsoes

Fórmula da regressão linear simples
y = b0 + b1 * X

"""

import pandas as pd

base = pd.read_csv('house-prices.csv')

#print(base.head())
#print(base.count())
#print(base.shape)

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

#plotando gráfico para ver a visualização dos dados

import matplotlib.pyplot as plt
#plt.scatter(X,y)

import numpy as np

np.random.seed(1)
bRandom = np.random.rand(2)

import tensorflow as tf

tf = tf.compat.v1

b0 = tf.Variable(bRandom[0])
b1 = tf.Variable(bRandom[1])

"""
utilizando agora os conceitos de placeholders do tensorflow
pois estamos lidando com muito mais dados e a memoria pode 
nao suportar todos da forma convencional
"""

#define a quantidade de dados em cada pacote
batch_size = 32
xph = tf.placeholder(tf.float64, [batch_size, 1])
yph = tf.placeholder(tf.float64, [batch_size, 1])

y_modelo = b0 + b1 * xph
erro = tf.losses.mean_squared_error(yph, y_modelo)
otimizador = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
treinamento = otimizador.minimize(erro)
init = tf.global_variables_initializer()


"executando o treinamento"
with tf.Session() as sess:
    sess.run(init)
    #print(sess.run(b0))
    #print(sess.run(b1))
    "executando por 10000 épocas"
    for i in range(10000):
        indices = np.random.randint(len(X), size = batch_size)
        feed = { xph: X[indices], yph: y[indices] }
        sess.run(treinamento, feed_dict = feed)
    b0_final, b1_final = sess.run([b0, b1])


"Após treinamento, pode-se realizar previsoes"

previsoes = b0_final + b1_final * X

"plotando no grafico para ver as previsoes"

plt.plot(X, y, 'o')
plt.plot(X, previsoes, color = "red")

"-- Desescalondando os resultados"

y1 = scaler_y.inverse_transform(y)
previsoes1 = scaler_y.inverse_transform(previsoes)


"-- verificando a taxa de erro utilizando mae"
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y1, previsoes1)

"""
Retorno final foi 173392.84
ou seja pode errar esse valor para mais e para menos.
Um Retorno não tao bom assim
"""






























