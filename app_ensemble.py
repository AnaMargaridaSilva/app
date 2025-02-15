import streamlit as st
from transformers import pipeline
import torch
import numpy as np
from huggingface_hub import login

hf_token = st.secrets["HUGGINGFACE_TOKEN"]
login(token=hf_token)

model_names = [
    "anamargarida/ensemble/model_seed_42",
    "anamargarida/ensemble/model_seed_123",
    "anamargarida/ensemble/model_seed_2024",
    "anamargarida/ensemble/model_seed_777",
    "anamargarida/ensemble/model_seed_999"
]

# Set device for inference
device = 0 if torch.cuda.is_available() else -1

# Load all models into pipelines
@st.cache_resource
def load_models():
    return [pipeline("text-classification", model=model_name, device=device) for model_name in model_names]

models = load_models()


st.title("Signal Detection with an Ensemble of Models")
st.write("Enter text below, and the ensemble model will classify it using majority voting.")

# User input
input_text = st.text_area("Enter a sentence for classification")

if st.button("Classify"):
    if input_text.strip():
        predictions = []
        
        # Get predictions from all models
        for classifier in models:
            result = classifier(input_text)
            predictions.append(result[0]["label"])  # Extract label from pipeline output
        
        # Majority voting
        final_prediction = max(set(predictions), key=predictions.count)

        # Display results
        st.write(f"### Ensemble Prediction: **{final_prediction}**")
        st.write(f"Model Predictions: {predictions}")

    else:
        st.warning("Please enter some text for classification.")
