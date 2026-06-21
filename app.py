import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction using Linear Regression")
st.markdown("Predict house prices based on Area, Bedrooms and Bathrooms")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]
    df = df.dropna()
    return df

df = load_data()

# ---------------- MODEL ----------------
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Enter House Details")

area = st.sidebar.slider(
    "Living Area (sq ft)",
    500,
    5000,
    1500
)

bedrooms = st.sidebar.slider(
    "Bedrooms",
    1,
    10,
    3
)

bathrooms = st.sidebar.slider(
    "Bathrooms",
    1,
    6,
    2
)

# ---------------- PREDICTION ----------------
st.subheader("House Price Prediction")

if st.button("Predict Price"):

    input_data = np.array(
        [[area, bedrooms, bathrooms]]
    )

    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated House Price: ${prediction:,.2f}"
    )

# ---------------- METRICS ----------------
st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("R² Score", f"{r2:.4f}")

with col2:
    st.metric("MSE", f"{mse:,.2f}")

# ---------------- DATASET ----------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------- PRICE DISTRIBUTION ----------------
st.subheader("Sale Price Distribution")

fig1, ax1 = plt.subplots(figsize=(8,4))

sns.histplot(
    df["SalePrice"],
    bins=30,
    kde=True,
    ax=ax1
)

st.pyplot(fig1)

# ---------------- CORRELATION HEATMAP ----------------
st.subheader("Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(6,4))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax2
)

st.pyplot(fig2)

# ---------------- ACTUAL VS PREDICTED ----------------
st.subheader("Actual vs Predicted Prices")

fig3, ax3 = plt.subplots(figsize=(8,5))

ax3.scatter(
    y_test,
    predictions
)

ax3.set_xlabel("Actual Prices")
ax3.set_ylabel("Predicted Prices")
ax3.set_title("Actual vs Predicted")

st.pyplot(fig3)

# ---------------- MODEL INFO ----------------
st.subheader("Model Information")

st.write("Algorithm Used: Linear Regression")

st.write("Features:")
st.write("- GrLivArea (Square Footage)")
st.write("- BedroomAbvGr (Bedrooms)")
st.write("- FullBath (Bathrooms)")

st.write("Target:")
st.write("- SalePrice")