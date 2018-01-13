import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import and split observations
df = pd.read_csv('./data/salaries.csv')
X = df.iloc[:, 0].values
y = df.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Fit linear regression to training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict off test set
y_pred = regressor.predict(X_test)