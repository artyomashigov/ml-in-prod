# ML in Prodcution Project

## Iris Species Prediction App  🌸

This project demonstrates a simple **machine learning model deployed to production using Streamlit**.

The application predicts the **species of an Iris flower** based on its numerical features (sepal length, sepal width, petal length, petal width) using a trained **Random Forest classifier**.

---

## 🧠 What was done 

- Trained a Random Forest model on the classic Iris dataset using scikit-learn  
- Saved the trained model using `joblib`  
- Built a Streamlit web application for predictions  
- Enabled CSV file upload for batch predictions  
- Deployed the app to Streamlit Cloud  

---

## 📁 Project structure

- `train_dump_joblib.py` – trains the model and saves it as a `.joblib` file  
- `random_forest_iris_model.joblib` – trained ML model  
- `app.py` – Streamlit application for predictions  
- `predictions.py` – helper script for local predictions  
- `iris_sample_data.csv` – example input data  
- `requirements.txt` – project dependencies  

---

## ⚙️ How the app works

1. User uploads a CSV file containing Iris feature values  
2. The saved Random Forest model is loaded  
3. Predictions are generated for each row  
4. Predicted species probabilities are returned  

---

## 🌍 Live application

The app is publicly available here:

👉 **https://iris-model-aa.streamlit.app/**

---

## 🛠 Tech stack

- Python  
- scikit-learn  
- pandas  
- joblib  
- Streamlit  

---

## 📝 Notes

This project focuses on **basic ML production concepts**:
- model persistence  
- reproducible predictions 
- simple UI for non-technical users  

It is intended as a learning and demonstration project.
Please note that the app will become inactive after 12 to 24 hours of inactivity due to using the community version.