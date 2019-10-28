import numpy as np


X = np.array([[18],[23],[28],[33],[38],[43],[48],[53],[58],[63]])
y = np.array([[871],[1132],[1042],[1356],[1488],[1638],[1569],[1754],[1866],[1900]])


import matplotlib.pyplot as plt

#plt.scatter(X,y)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)

#formula: y = b0 + b1 * x1
# b0
print(regressor.intercept_)

#b1 
print(regressor.coef_)

# previsao usando a formula
previsao1 = regressor.intercept_ + regressor.coef_ * 40

# previsao usando o sklearn
#deu errro
#previsao2 = regressor.predict(40)

previsoes = regressor.predict(X)

# diferencas do orevisto e o valor real
resultado = abs(y - previsoes)

# media das diferencas
# ou seja, o modelo pode ter errado 70 pra mais ou pra menos
#tambem chamado de MAE - MEAN ABSOLUT ERROR
resultadoMedias = abs(y - previsoes).mean()

# -------- utilizando sklearn para calcular o MAE e o MSO
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y, previsoes)

# MSO é maior porque faz o quadrado dos valores.
#mais indicado para o treinamento do algoritmo
# porque ele penaliza os valores muito grandes
mso = mean_squared_error(y, previsoes)

#gerando o grafico

plt.plot(X,y, 'o')
plt.plot(X, previsoes, color="red")
plt.title("Regressão Linear Simples")
plt.xlabel("Idade")
plt.ylabel("Custo")




