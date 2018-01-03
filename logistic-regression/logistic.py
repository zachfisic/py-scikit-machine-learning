from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings

# Suppress scipy warning on gelsd driver (osx)
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

#accuracy score
print(model.score(X_test, y_test))

#matrix of model's performance
print(metrics.confusion_matrix(y_test, predictions))