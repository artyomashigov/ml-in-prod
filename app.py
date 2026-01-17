import streamlit as st
import pandas as pd
import joblib

st.title("Iris Species Predictor")

with st.sidebar:
    st.header("Data requirements")
    st.caption("To inference the species of iris flower, please upload a CSV file.")
    with st.expander("Data Format Example"):
        st.markdown("- encoding: utf-8")
        st.markdown("- separator: comma")
        st.markdown("- decimal: .")
        st.markdown("- first row: header")
    st.divider()
    st.caption("<p style='text-align:center'>Developed by Artyom Ashigov</p>", unsafe_allow_html=True)

# session state flag
if "started" not in st.session_state:
    st.session_state.started = False

def start_app():
    st.session_state.started = True

st.button("Get started", on_click=start_app)

if st.session_state.started:
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.header("Uploaded Data Sample")
        st.write(df.head())
        model = joblib.load("random_forest_iris_model.joblib")
        predictions = model.predict_proba(df)
        pred_df = pd.DataFrame(predictions, columns=['setosa', 'versicolor', 'virginica'])
        st.header("Predictions")
        st.write(pred_df)
        pred_df = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=pred_df,
            file_name='iris_predictions.csv',
            mime='text/csv',
            key='download-csv'
        )