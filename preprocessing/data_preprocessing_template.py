import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder

df = pd.read_csv('./data/sample.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, 3].values

"""
Impute missing values.

Description: datasets with missing values (e.g. NaN) can be filled via imputation.
Example: average the values in a column to fill rows with 'NaN' value
"""
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


"""
Encode categorical data

Description: data with values that belong to categories (a.k.a labels) can be transformed to integers.
Example: encode location labels into dummy variables
"""
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder_X = OneHotEncoder(categorical_features = [0])
X = onehotencoder_X.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)