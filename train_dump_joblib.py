import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# save model
joblib.dump(model, "random_forest_iris_model.joblib")
