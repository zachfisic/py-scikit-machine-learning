import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('./data/position_salaries.csv')
X = df.iloc[:, 1].values.reshape(-1,1)
y = df.iloc[:, -1].values.reshape(-1,1)

# Fit polynomial regression to data
regressor = PolynomialFeatures(degree=4)
X_poly = regressor.fit_transform(X)
l_reg = LinearRegression()
l_reg.fit(X_poly, y)

# Visualizing results
X_grid = np.arange(min(X), max(X), step=0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color="red")
plt.plot(X_grid, l_reg.predict(regressor.fit_transform(X_grid)), color="blue")
plt.title('Polynomial Regression')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()