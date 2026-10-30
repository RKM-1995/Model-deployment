"""
House Price Prediction using Linear Regression + Streamlit
===========================================================

Run:
    pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib
    streamlit run "Linear regression on house price.py"

Keep house_price.csv in the same folder, or upload a CSV from the app.
The target column must be named: price
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------- Page setup ---------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 House Price Prediction")
st.caption("Linear Regression | Data Exploration | Model Evaluation | Price Prediction")

MODEL_PATH = Path("house_price_model.pkl")
DEFAULT_DATA_PATH = Path("house_price.csv")


# --------------------------- Data loading ---------------------------
@st.cache_data
def read_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    """Read uploaded CSV bytes."""
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))


def load_data() -> pd.DataFrame:
    uploaded_file = st.sidebar.file_uploader("Upload house_price.csv", type=["csv"])

    try:
        if uploaded_file is not None:
            data = read_uploaded_csv(uploaded_file.getvalue())
        elif DEFAULT_DATA_PATH.exists():
            data = pd.read_csv(DEFAULT_DATA_PATH)
            st.sidebar.success(f"Loaded {DEFAULT_DATA_PATH.name}")
        else:
            st.info(
                "Upload your housing CSV using the sidebar. "
                "Alternatively, place `house_price.csv` beside this Python file."
            )
            st.stop()
        return data
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        st.stop()


df_original = load_data()

# Normalize column names and validate target
df_original.columns = df_original.columns.astype(str).str.strip()
if "price" not in df_original.columns:
    st.error("The dataset must contain a target column named `price`.")
    st.write("Columns found:", list(df_original.columns))
    st.stop()

if df_original.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

# Remove rows without target; convert target to numeric
df = df_original.copy()
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df.dropna(subset=["price"])

# Remove duplicate rows and replace infinite values
df = df.replace([np.inf, -np.inf], np.nan).drop_duplicates()
if df.empty:
    st.error("No usable rows remain after cleaning the target column.")
    st.stop()


# --------------------------- Sidebar settings ---------------------------
st.sidebar.header("Model settings")
test_size_percent = st.sidebar.slider("Test data (%)", 10, 40, 20, 5)
remove_outliers = st.sidebar.checkbox("Remove upper IQR outliers from price and area", value=True)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=99999, value=42, step=1)


# --------------------------- Data cleaning ---------------------------
def upper_iqr_filter(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Remove observations above the IQR upper fence for a selected numeric column."""
    if column not in data.columns:
        return data
    values = pd.to_numeric(data[column], errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return data
    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    # Preserve rows where the feature is missing; imputation happens in the pipeline.
    return data[values.isna() | (values <= upper_bound)]


df_model = df.copy()
rows_before_cleaning = len(df_model)
if remove_outliers:
    df_model = upper_iqr_filter(df_model, "price")
    df_model = upper_iqr_filter(df_model, "area")

if len(df_model) < 5:
    st.error("Too few rows remain to train a model. Disable outlier removal or provide more data.")
    st.stop()

X = df_model.drop(columns=["price"])
y = df_model["price"]

if X.shape[1] == 0:
    st.error("The dataset must have at least one feature column in addition to `price`.")
    st.stop()

# Treat numeric-looking columns as numeric when pandas already recognizes them as such.
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

# Build imputing/scaling/encoding steps. Handle scikit-learn versions with or without
# OneHotEncoder(sparse_output=...).
try:
    one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
except TypeError:
    one_hot = OneHotEncoder(handle_unknown="ignore", sparse=True)

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", one_hot),
    ]
)

transformers = []
if numeric_features:
    transformers.append(("num", numeric_pipeline, numeric_features))
if categorical_features:
    transformers.append(("cat", categorical_pipeline, categorical_features))

preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size_percent / 100, random_state=int(random_state)
)

try:
    model_pipeline.fit(X_train, y_train)
    predictions = model_pipeline.predict(X_test)
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.info("Check that feature columns contain usable values and that `price` is numeric.")
    st.stop()

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = float(np.sqrt(mse))
r2 = r2_score(y_test, predictions) if len(y_test) >= 2 else float("nan")


# --------------------------- Tabs ---------------------------
tab_overview, tab_eda, tab_model, tab_predict = st.tabs(
    ["📊 Overview", "🔎 Data Analysis", "🤖 Model Results", "🏡 Predict Price"]
)

with tab_overview:
    st.subheader("Dataset overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows loaded", f"{len(df_original):,}")
    col2.metric("Rows used", f"{len(df_model):,}")
    col3.metric("Input features", f"{X.shape[1]:,}")
    col4.metric("Missing cells", f"{int(df_model.isna().sum().sum()):,}")

    if remove_outliers:
        st.caption(f"Rows removed by the selected upper-IQR filters: {rows_before_cleaning - len(df_model):,}")

    st.markdown("#### Preview")
    st.dataframe(df_model.head(20), use_container_width=True)

    st.markdown("#### Data types and missing values")
    details = pd.DataFrame({
        "data_type": df_model.dtypes.astype(str),
        "missing_values": df_model.isna().sum(),
        "unique_values": df_model.nunique(dropna=True),
    })
    st.dataframe(details, use_container_width=True)

    st.markdown("#### Descriptive statistics")
    st.dataframe(df_model.describe(include="all").T, use_container_width=True)


with tab_eda:
    st.subheader("Exploratory Data Analysis")
    st.markdown("#### Price distribution")
    fig, ax = plt.subplots()
    sns.histplot(df_model["price"], kde=True, ax=ax)
    ax.set_xlabel("Price")
    st.pyplot(fig)
    plt.close(fig)

    numeric_for_plot = [c for c in numeric_features if c in df_model.columns]
    if numeric_for_plot:
        selected_numeric = st.selectbox("Choose a numerical feature", numeric_for_plot)
        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots()
            sns.histplot(df_model[selected_numeric].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {selected_numeric}")
            st.pyplot(fig)
            plt.close(fig)
        with right:
            fig, ax = plt.subplots()
            sns.boxplot(x=df_model[selected_numeric], ax=ax)
            ax.set_title(f"Box plot of {selected_numeric}")
            st.pyplot(fig)
            plt.close(fig)

    if categorical_features:
        st.markdown("#### Categorical feature counts")
        selected_category = st.selectbox("Choose a categorical feature", categorical_features)
        counts = df_model[selected_category].fillna("(missing)").astype(str).value_counts().head(20)
        st.bar_chart(counts)

    st.markdown("#### Correlation between numerical columns")
    numeric_df = df_model.select_dtypes(include=np.number)
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("At least two numerical columns are required for a correlation heatmap.")


with tab_model:
    st.subheader("Test-set evaluation")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{mae:,.2f}")
    m2.metric("MSE", f"{mse:,.2f}")
    m3.metric("RMSE", f"{rmse:,.2f}")
    m4.metric("R² score", f"{r2:.3f}" if np.isfinite(r2) else "N/A")

    st.caption(
        "MAE, RMSE, and MSE are errors in the target's units (MSE is in squared units). "
        "A higher R² is generally better, but evaluate it alongside the errors and the data."
    )

    comparison = pd.DataFrame({
        "Actual price": y_test.to_numpy(),
        "Predicted price": predictions,
        "Absolute error": np.abs(y_test.to_numpy() - predictions),
    }).reset_index(drop=True)
    st.markdown("#### Actual vs predicted prices")
    st.dataframe(comparison, use_container_width=True)

    fig, ax = plt.subplots()
    ax.scatter(comparison["Actual price"], comparison["Predicted price"], alpha=0.7)
    low = min(comparison["Actual price"].min(), comparison["Predicted price"].min())
    high = max(comparison["Actual price"].max(), comparison["Predicted price"].max())
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")
    ax.set_title("Actual vs predicted")
    st.pyplot(fig)
    plt.close(fig)

    if st.button("💾 Save trained model", type="primary"):
        try:
            joblib.dump(model_pipeline, MODEL_PATH)
            st.success(f"Model saved to: {MODEL_PATH.resolve()}")
        except Exception as exc:
            st.error(f"Could not save model: {exc}")

    st.download_button(
        "Download test predictions (CSV)",
        data=comparison.to_csv(index=False).encode("utf-8"),
        file_name="house_price_test_predictions.csv",
        mime="text/csv",
    )


with tab_predict:
    st.subheader("Enter house details")
    st.write("Choose values for the input features below. The model uses the same columns as the uploaded dataset.")

    with st.form("prediction_form"):
        user_values = {}
        input_columns = st.columns(2)

        for idx, column in enumerate(X.columns):
            with input_columns[idx % 2]:
                if column in numeric_features:
                    series = pd.to_numeric(X[column], errors="coerce").dropna()
                    if series.empty:
                        default_value = 0.0
                        min_value = None
                        max_value = None
                    else:
                        default_value = float(series.median())
                        min_value = float(series.min())
                        max_value = float(series.max())

                    user_values[column] = st.number_input(
                        f"{column}",
                        value=default_value,
                        min_value=min_value,
                        max_value=max_value,
                        help=f"Training-data median: {default_value:,.2f}",
                    )
                else:
                    choices = X[column].dropna().astype(str).unique().tolist()
                    choices = sorted(choices)
                    if not choices:
                        choices = ["Unknown"]
                    user_values[column] = st.selectbox(f"{column}", choices)

        submitted = st.form_submit_button("Predict house price", type="primary")

    if submitted:
        input_df = pd.DataFrame([user_values], columns=X.columns)
        try:
            predicted_price = float(model_pipeline.predict(input_df)[0])
            st.success("Prediction complete")
            st.metric("Estimated house price", f"{predicted_price:,.2f}")
            st.caption("This is a model estimate, not a guaranteed market valuation.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


st.sidebar.markdown("---")
st.sidebar.caption("Target column: `price`")
st.sidebar.caption("Model: Linear Regression")
st.sidebar.caption("Preprocessing: imputation, scaling, one-hot encoding")
