import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# Remove dummy categorical variable
X = X[:, 1:]

# Split dataset into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Perform regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict off test set
y_pred = regressor.predict(X_test)
