import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./data/startups.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1:].values

# Encode labels
labelencoder_X = LabelEncoder()
X[:, -1] = labelencoder_X.fit_transform(X[:, -1])
onehotencoder = OneHotEncoder(categorical_features=[-1])
X = onehotencoder.fit_transform(X).toarray()

# Remove single dummy categorical variable
X = X[:, 1:]

# Add necessary constant of ones for stats model
X = np.append(arr=np.ones(shape=(50,1), dtype=np.int8), values=X, axis=1)

"""
Multiple Linear Regression

Build model with optimal number of independent variables using Backward Elimination strategy
"""
sig_lvl = 0.05
def backward_elimination(ivars, dvar, sl):
  n = len(ivars[0])
  for i in range(0, n):
    # Run ordinary least squares algorithm
    regressor_OLS = sm.OLS(endog=dvar, exog=ivars).fit()

    # Remove independent variable with largest p-value (if larger than sig level)
    max_pvalue = max(regressor_OLS.pvalues).astype(float)
    if max_pvalue > sl:
      for j in range(0, n - i):
        if (regressor_OLS.pvalues[j].astype(float) == max_pvalue):
          ivars = np.delete(ivars, j, 1)
  return ivars

X_modeled = backward_elimination(X, y, sig_lvl)