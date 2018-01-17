import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('./data/position_salaries.csv')
X = df.iloc[:, 1].values.reshape(-1,1)
y = df.iloc[:, -1].values.reshape(-1,1)

# Fit polynomial regression to data
regressor = PolynomialFeatures(degree=2)
X_poly = regressor.fit_transform(X)
l_reg = LinearRegression()
l_reg.fit(X_poly, y)