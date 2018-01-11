import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer

df = pd.read_csv('./data/sample.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, 3].values

"""
Impute missing values.

Description: datasets with missing values (e.g. NaN) can be filled via imputation. Imputer class must be instantiated and fit to data
Example: average the values in a column to fill rows with 'NaN' value
"""
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

