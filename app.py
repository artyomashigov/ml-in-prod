import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸", layout="wide")

required_columns = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


@st.cache_resource
def load_model():
    return joblib.load("random_forest_iris_model.joblib")


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def validate_input_df(df: pd.DataFrame) -> tuple[bool, list[str], pd.DataFrame]:
    # returns: (is_valid, errors, cleaned_df)
    errors = []
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        errors.append(f"missing required columns: {missing_cols}")

    present_cols = [c for c in required_columns if c in df.columns]
    df = df[present_cols]

    # coerce to numeric to validate types
    for c in present_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if present_cols:
        nan_counts = df[present_cols].isna().sum()
        if int(nan_counts.sum()) > 0:
            errors.append(
                "found missing or non-numeric values. missing counts per column: "
                + ", ".join([f"{k}={int(v)}" for k, v in nan_counts.items() if int(v) > 0])
            )

    return (len(errors) == 0), errors, df


def make_predictions(model, X: pd.DataFrame) -> pd.DataFrame:
    proba = model.predict_proba(X)

    iris_class_names = ["setosa", "versicolor", "virginica"]
    proba_df = pd.DataFrame(proba, columns=iris_class_names)
    proba_df["predicted_class"] = proba_df.idxmax(axis=1)

    return proba_df



# sidebar
with st.sidebar:
    st.header("📌 Data requirements")
    st.caption("Upload a CSV with these columns (exact names):")
    st.code("\n".join(required_columns), language="text")

    with st.expander("✅ Example format"):
        st.markdown("- encoding: utf-8")
        st.markdown("- separator: comma")
        st.markdown("- header row required")

    sample_df = pd.DataFrame(
        [
            [5.1, 3.5, 1.4, 0.2],
            [6.3, 3.3, 4.7, 1.6],
            [6.5, 3.0, 5.8, 2.2],
        ],
        columns=required_columns,
    )
    st.download_button(
        "⬇️ Download sample CSV",
        data=to_csv_bytes(sample_df),
        file_name="iris_sample_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.divider()
    st.caption("<p style='text-align:center'>Developed by Artyom Ashigov</p>", unsafe_allow_html=True)


# main page
st.title("🌸 Iris Species Predictor")

if "started" not in st.session_state:
    st.session_state.started = False

st.button("Get started", on_click=lambda: st.session_state.update({"started": True}))

if st.session_state.started:
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is None:
        st.info("Upload a CSV file to begin.")
        st.stop()

    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"could not read csv: {e}")
        st.stop()

    st.subheader("📄 Uploaded data preview")
    st.dataframe(raw_df.head(15), use_container_width=True)

    is_valid, errors, X = validate_input_df(raw_df)
    if not is_valid:
        st.error("❌ Input validation failed.")
        for err in errors:
            st.write(f"- {err}")
        st.stop()

    st.success("✅ Input validation passed.")

    model = load_model()
    preds_df = make_predictions(model, X)

    st.subheader("✅ Predictions")
    st.dataframe(preds_df.head(30), use_container_width=True)

    st.download_button(
        "⬇️ Download predictions as CSV",
        data=to_csv_bytes(preds_df),
        file_name="iris_predictions.csv",
        mime="text/csv",
    )