import streamlit as st
import pandas as pd
import joblib
import requests
import numpy as np

st.set_page_config(page_title="Real Estate AI", page_icon="🏠")
st.title("🏠 Real Estate Price Predictor")
st.write("Predict a house price and understand the model's prediction.")

@st.cache_resource
def load_model():
    return joblib.load("models/ridge_model.joblib")

model = load_model()

st.subheader("Enter House Details")

with st.form("house_form"):
    sqft_living = st.number_input("Living area (sqft)", min_value=100, value=1500)
    sqft_lot = st.number_input("Lot area (sqft)", min_value=100, value=5000)
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=15.0, value=2.0, step=0.25)
    floors = st.number_input("Floors", min_value=1.0, max_value=4.0, value=1.0, step=0.5)
    view = st.selectbox("View quality (0–4)", [0, 1, 2, 3, 4])
    condition = st.selectbox("House condition (1–5)", [1, 2, 3, 4, 5])
    waterfront = st.selectbox("Waterfront", [0, 1])
    sqft_above = st.number_input("Above-ground area (sqft)", min_value=0, value=1200)
    sqft_basement = st.number_input("Basement area (sqft)", min_value=0, value=300)
    house_age = st.number_input("House age (years)", min_value=0, value=20)
    has_basement = st.selectbox("Has basement?", [0, 1])
    was_renovated = st.selectbox("Was renovated?", [0, 1])
    zipcode = st.text_input("Zipcode", value="98118")

    submitted = st.form_submit_button("Predict Price")

if submitted:
    # Apply the same log transformation used during training.
    house = pd.DataFrame([{
        "sqft_living": np.log1p(sqft_living),
        "sqft_lot": np.log1p(sqft_lot),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "floors": floors,
        "view": view,
        "condition": condition,
        "waterfront": waterfront,
        "sqft_above": np.log1p(sqft_above),
        "sqft_basement": np.log1p(sqft_basement),
        "house_age": house_age,
        "has_basement": has_basement,
        "was_renovated": was_renovated,
        "zipcode": str(zipcode)
    }])

    predicted_price = float(np.expm1(model.predict(house)[0]))

    st.subheader("Prediction")
    st.success(f"Estimated house price: ${predicted_price:,.0f}")

    if st.button("Explain Prediction with AI"):
        prompt = f"""
        Explain this real-estate model prediction in simple English.
        Predicted price: ${predicted_price:,.0f}
        House features: {house.to_dict(orient='records')[0]}

        Do not invent facts or claim causation.
        Explain that this is a machine-learning estimate, not a guaranteed price.
        """

        try:
            with st.spinner("Generating explanation with local AI..."):
                response = requests.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": "llama3.2:3b",
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False
                    },
                    timeout=180
                )
                response.raise_for_status()
                explanation = response.json()["message"]["content"]

            st.subheader("AI Explanation")
            st.write(explanation)

        except requests.RequestException as e:
            st.error(
                "Could not connect to Ollama. Make sure Ollama is running "
                "and llama3.2:3b is installed."
            )
            st.caption(str(e))