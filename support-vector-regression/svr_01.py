import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./data/position_salaries.csv')
X = df.iloc[:, 1].values.reshape(-1,1)
y = df.iloc[:, -1].values.reshape(-1,1)

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Start SVR kernel
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

# Visualizing results
plt.scatter(X, y, color="red")
plt.plot(X, regressor.predict(X), color="blue")
plt.title('Polynomial Regression')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

# Run prediction
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))