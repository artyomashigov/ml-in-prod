import pandas as pd
import joblib
from sklearn.datasets import load_iris

# load model
model = joblib.load("random_forest_iris_model.joblib")

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# save to csv for later use
df.to_csv("iris_sample_data.csv", index=False)

# make predictions
pred = model.predict_proba(df)
print(pred)
