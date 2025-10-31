import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 🎯 Title
st.title("🏡 House Price Prediction Dashboard")

st.write("""
Use this simple ML model to estimate house prices based on features like rooms, area, and age.
""")

# 📥 Upload data (or use default)
uploaded_file = st.file_uploader("Upload a CSV file (optional)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    # Example dataset
    df = pd.DataFrame({
        "area_sqft": np.random.randint(800, 4000, 100),
        "bedrooms": np.random.randint(1, 6, 100),
        "bathrooms": np.random.randint(1, 4, 100),
        "age_years": np.random.randint(0, 30, 100),
        "price": np.random.randint(100000, 1000000, 100)
    })

st.write("### Dataset preview", df.head())

# 📊 Feature selection
X = df.drop("price", axis=1)
y = df["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate model
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
st.write(f"**Model MAE:** ${mae:,.0f}")

# 🧮 User input for prediction
st.subheader("Predict a new house price")
area = st.number_input("Area (sqft)", min_value=500, max_value=10000, value=1500)
bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 5, 2)
age = st.slider("Age of house (years)", 0, 50, 10)

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        "area_sqft": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "age_years": [age]
    })
    prediction = model.predict(input_data)[0]
    st.success(f"🏠 Estimated Price: ${prediction:,.0f}")
