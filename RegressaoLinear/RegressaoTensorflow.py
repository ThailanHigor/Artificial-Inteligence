# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 14:22:27 2019
@author: Thailan

Algoritmo de regressão linear simples com TensorFlow

Salario de uma pessoa com base na idade

"""

import numpy as np

X = np.array([[18],[23],[28],[33],[38],[43],[48],[53],[58],[63]])
y = np.array([[871],[1132],[1042],[1356],[1488],[1638],[1569],[1754],[1866],[1900]])

""" 
importante:
os valores estão muito grandes, com o tensorflow,
precisamos fazer o escalonamento dos valores.
"""

from sklearn.preprocessing import StandardScaler

scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

"Plotar o gráfico para ver que os pontos estão exatamente nas mesmas posicoes"
import matplotlib.pyplot as plt

" ------------------------------------------------"

#plt.scatter(X,y)
"FORMULA DA REGRESSEÃO LINEAR SIMPLES"
" y = b0 + b1 * x"

"-------------------------------------------------"
"gerando dois numeros aleatorios porem sempre os mesmos"
np.random.seed(0)

"inicio do b0 e do b1"
rand = np.random.rand(2)

import tensorflow as tf

#troca na versão
tf = tf.compat.v1

b0 = tf.Variable(rand[0])
b1 = tf.Variable(rand[1])

erro = tf.losses.mean_squared_error(y, (b0 + b1 * X))
otimizador = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
treinamento = otimizador.minimize(erro)
init = tf.global_variables_initializer()


"executando o treinamento"
with tf.Session() as sess:
    sess.run(init)
    #print(sess.run(b0))
    #print(sess.run(b1))
    "executando por 1000 épocas"
    for i in range(1000):
        sess.run(treinamento)
    b0_final, b1_final = sess.run([b0, b1])

"Após treinamento, pode-se realizar previsoes"

previsoes = b0_final + b1_final * X

"plotando no grafico para ver as previsoes"

plt.plot(X, previsoes, color = "red")
plt.plot(X, y, 'o')

"previsao para uma pessoa de 40 anos de idade"
"antes de jogar na formula temos que escalar o valor na mesma escala de X"
idadeEscalada = scaler_x.transform([[40]])

"para saber o valor em real, faz-se o inverse transform do valor da formula"
previsao = scaler_y.inverse_transform(b0_final + b1_final * idadeEscalada)
print(previsao) 


"-- Desescalondando os resultados"

y1 = scaler_y.inverse_transform(y)
previsoes1 = scaler_y.inverse_transform(previsoes)


"-- verificando a taxa de erro utilizando mae e mse"
from sklearn.metrics import mean_absolute_error,mean_squared_error
mae = mean_absolute_error(y1, previsoes1)
mse = mean_squared_error(y1, previsoes1)

"""
NOTA:
    -mse é recomendado utilizar no treinamento porque ele eleva
    os resultados ao quadrado portanto penalizando numeros grandes
    
    -mae é recomendado utilizar em comparações pois o valor é o real 
    e significa que o algoritmo tem uma taxa de erro de
    X para cima ou X para baixo
"""


