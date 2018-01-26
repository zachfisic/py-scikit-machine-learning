import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('./data/position_salaries.csv')
X = df.iloc[:, 1].values.reshape(-1,1)
y = df.iloc[:, -1].values.reshape(-1,1)

# Fit Decision Tree Regression to dataset
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

# Visualize results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color="blue")
plt.title('Decision Tree Regression')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

# Run prediction
y_pred = regressor.predict(6.5)
