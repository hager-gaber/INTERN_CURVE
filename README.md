# Real Estate Price Predictor

A Streamlit app that predicts a house price and explains what pushed the estimate up or down, in plain language.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires `models/ridge_model.joblib` (created by the training notebook).
For the optional AI explanation, install [Ollama](https://ollama.com) and run `ollama pull llama3.2:3b`.

## How it works

- **Model:** Ridge regression on log-price (median error 11.1%, within 10% for 46% of houses; better than Random Forest and Gradient Boosting).
- **Chart and Summary:** generated directly from the model's numbers, so they contain no invented facts.
- **AI explanation:** a local LLM rewrites the summary in friendly language.

## Limitations

This is an estimate, not a professional appraisal. Typical error is around 11%, and the model can overestimate small houses in expensive neighborhoods.
