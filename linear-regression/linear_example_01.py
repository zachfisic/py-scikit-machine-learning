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
X_train = X_train.reshape(-1,1)
y_train = y_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# Fit linear regression to training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict off test set
y_pred = regressor.predict(X_test)

# Visualizing results
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title('Salary vs Exp (training)')
plt.xlabel('years')
plt.ylabel('salary')
plt.show()