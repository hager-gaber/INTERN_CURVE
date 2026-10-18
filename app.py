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

FRIENDLY_NAMES = {
    "sqft_living": "living area",
    "sqft_above": "above-ground area",
    "sqft_lot": "lot size",
    "sqft_basement": "basement size",
    "bedrooms": "number of bedrooms",
    "bathrooms": "number of bathrooms",
    "floors": "number of floors",
    "view": "view quality",
    "condition": "house condition",
    "waterfront": "waterfront location",
    "house_age": "house age",
    "has_basement": "having a basement",
    "was_renovated": "renovation status",
    "zipcode": "neighborhood",
}


def get_contributions(model, house: pd.DataFrame) -> pd.DataFrame:
    """Per-feature contribution in log-price space (coef * scaled value)."""
    pre = model.named_steps["pre"]
    ridge = model.named_steps["model"]

    Xt = pre.transform(house)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = np.asarray(Xt).ravel()

    df = pd.DataFrame({
        "feature": pre.get_feature_names_out(),
        "contribution": Xt * ridge.coef_,
    })
    df["feature"] = (
        df["feature"]
        .str.replace("num__", "", regex=False)
        .str.replace(r"^cat__zipcode_.*", "zipcode", regex=True)
    )
    df = df.groupby("feature", as_index=False)["contribution"].sum()
    df["abs"] = df["contribution"].abs()
    return df.sort_values("abs", ascending=False).drop(columns="abs")


def build_summary(contrib_df, top_n=5):
    """Factual summary written by code (no LLM, no invention)."""
    top = contrib_df.head(top_n)
    ups = [FRIENDLY_NAMES.get(f, f) for f in top[top["contribution"] > 0]["feature"]]
    downs = [FRIENDLY_NAMES.get(f, f) for f in top[top["contribution"] < 0]["feature"]]

    parts = []
    if ups:
        parts.append(f"**Raised the estimate:** {', '.join(ups)}.")
    if downs:
        parts.append(f"**Lowered the estimate:** {', '.join(downs)}.")
    parts.append("These effects are measured against an average house in the data "
                 "the model learned from. They show patterns in the data, "
                 "not judgments about the house.")
    return "\n\n".join(parts)


def describe_contributions(contrib_df, top_n=5):
    lines = []
    for r in contrib_df.head(top_n).itertuples():
        name = FRIENDLY_NAMES.get(r.feature, r.feature)
        direction = "raised the estimate" if r.contribution > 0 else "lowered the estimate"
        lines.append(f"- {name}: {direction}")
    return "\n".join(lines)


def build_prompt(price, top_contrib):
    drivers = describe_contributions(top_contrib)
    return f"""
Write a short, friendly explanation for a home buyer or seller with no technical background.

Estimated price: \\${price:,.0f}

What affected the estimate (compared to an average house in our data):
{drivers}

Write 3 short paragraphs:
1. State the estimated price.
2. Explain which factors raised the estimate and which lowered it, in plain words.
3. Remind the reader this is an automated estimate, not a guaranteed price or a professional appraisal.

Tone: warm, friendly and encouraging. You may use light expressions such as
"a nice boost" or "a slight drag", but ONLY about how a factor affected the estimate.

Strict rules:
- Use only the information above.
- Never make claims about the condition, quality or desirability of the house or neighborhood.
- Never say the house has flaws or problems.
- Do not explain WHY a factor had its effect; only say whether it raised or lowered the estimate.
- Do not mention any numbers except the price.
"""


st.subheader("Enter House Details")

with st.form("house_form"):
    sqft_living = st.number_input("Living area (sqft)", min_value=100, value=1500)
    sqft_lot = st.number_input("Lot area (sqft)", min_value=100, value=5000)
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=15.0, value=2.0, step=0.25)
    floors = st.number_input("Floors", min_value=1.0, max_value=4.0, value=1.0, step=0.5)
    view = st.selectbox("View quality (0–4)", [0, 1, 2, 3, 4])
    condition = st.selectbox("House condition (1–5)", [1, 2, 3, 4, 5], index=2)
    waterfront = st.selectbox("Waterfront", ["No", "Yes"]) == "Yes"
    sqft_above = st.number_input("Above-ground area (sqft)", min_value=0, value=1200)
    sqft_basement = st.number_input("Basement area (sqft, 0 if none)", min_value=0, value=300)
    house_age = st.number_input("House age (years)", min_value=0, value=20)
    was_renovated = st.selectbox("Was renovated?", ["No", "Yes"]) == "Yes"
    zipcode = st.text_input("Zipcode", value="98118")

    submitted = st.form_submit_button("Predict Price")

if submitted:
    if sqft_above + sqft_basement != sqft_living:
        st.warning("Heads up: above-ground area + basement area doesn't equal "
                   "the living area. Please double-check your numbers.")

    raw = {
        "sqft_living": sqft_living, "sqft_lot": sqft_lot, "bedrooms": bedrooms,
        "bathrooms": bathrooms, "floors": floors, "view": view,
        "condition": condition, "waterfront": int(waterfront),
        "sqft_above": sqft_above, "sqft_basement": sqft_basement,
        "house_age": house_age,
        "has_basement": int(sqft_basement > 0),   # computed, not typed
        "was_renovated": int(was_renovated),
        "zipcode": str(zipcode),
    }

    model_input = raw.copy()
    for c in ["sqft_living", "sqft_lot", "sqft_above", "sqft_basement"]:
        model_input[c] = np.log1p(raw[c])
    house = pd.DataFrame([model_input])

    predicted_price = float(np.expm1(model.predict(house)[0]))
    contributions = get_contributions(model, house)

    st.session_state["result"] = {
        "price": predicted_price,
        "contributions": contributions,
    }
    st.session_state.pop("explanation", None)

if "result" in st.session_state:
    res = st.session_state["result"]

    st.subheader("Prediction")
    st.success(f"Estimated house price: \\${res['price']:,.0f}")

    low, high = res["price"] * 0.85, res["price"] * 1.15
    st.caption(f"Typical range: \\${low:,.0f} – \\${high:,.0f} "
           "(the model's typical error is around 11%, so the real price can differ).")

    st.subheader("What drove this estimate?")
    top = res["contributions"].head(8).copy()
    top["feature"] = top["feature"].map(FRIENDLY_NAMES).fillna(top["feature"])
    st.bar_chart(top.set_index("feature")["contribution"])
    st.caption("Bars above zero raised the estimate; bars below zero lowered it, "
               "compared to an average house.")

    st.subheader("Summary")
    st.write(build_summary(res["contributions"]))

    if st.button("Explain Prediction with AI"):
        prompt = build_prompt(res["price"], res["contributions"])
        try:
            with st.spinner("Generating explanation with local AI..."):
                r = requests.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": "llama3.2:3b",
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {"temperature": 0.2, "num_predict": 350},
                    },
                    timeout=180,
                )
                r.raise_for_status()
                st.session_state["explanation"] = r.json()["message"]["content"]
        except requests.RequestException as e:
            st.error("Could not connect to Ollama. Make sure Ollama is running "
                     "and llama3.2:3b is installed.")
            st.caption(str(e))

    if "explanation" in st.session_state:
        st.subheader("AI Explanation")
        st.write(st.session_state["explanation"])
        st.caption("AI-written text. It may be imperfect; the Summary above is "
                   "generated directly from the model's numbers.")