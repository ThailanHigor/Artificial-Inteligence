# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 19:05:06 2019

@author: Thailan

Altorimo de classificacao usando Regressao Logisica

Para determinar se a pessoa ganhará + ou - de 50mil no ano de salario
"""

import pandas as pd

base = pd.read_csv("census.csv")


#base['income'].unique()
X = base.iloc[:, 0:14].values
y = base.iloc[:, 14].values

"necessario transformar os valoes de texto em numeros"

from sklearn.preprocessing import LabelEncoder, StandardScaler
label_encoder = LabelEncoder()

X[:,1] = label_encoder.fit_transform(X[:,1])
X[:,3] = label_encoder.fit_transform(X[:,3])
X[:,5] = label_encoder.fit_transform(X[:,5])
X[:,6] = label_encoder.fit_transform(X[:,6])
X[:,7] = label_encoder.fit_transform(X[:,7])
X[:,8] = label_encoder.fit_transform(X[:,8])
X[:,9] = label_encoder.fit_transform(X[:,9])
X[:,13] = label_encoder.fit_transform(X[:,13])


"Escalonamento dos dados"

scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)

"Divisao da bases treinamento e testes em 30%"

from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size= 0.3)

#Treinamento do modelo
print("treinando ....")
from sklearn.linear_model import LogisticRegression
classificador = LogisticRegression(max_iter = 10000)
classificador.fit(X_treinamento,y_treinamento)

#previsoes usando a base teste
previsoes = classificador.predict(X_teste)

"visualizar a taxa de acerto"

from sklearn.metrics import accuracy_score
taxa_acerto = accuracy_score(y_teste, previsoes)
print(taxa_acerto * 100)

"Retorno 82% de acerto"









