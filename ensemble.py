import streamlit as st
from transformers import pipeline
import torch
import numpy as np
from huggingface_hub import login
import statistics

hf_token = st.secrets["HUGGINGFACE_TOKEN"]
login(token=hf_token)

model_names = [
    "anamargarida/ensemble_model_seed_42",
    "anamargarida/ensemble_model_seed_123",
    "anamargarida/ensemble_model_seed_2024"
]

# Set device for inference
device = 0 if torch.cuda.is_available() else -1

# Load all models into pipelines
@st.cache_resource
def load_models():
    return [pipeline("text-classification", model=model_name, device=device) for model_name in model_names]

models = load_models()

st.title("Signal Detection with an Ensemble")
st.write("Enter text below, and the ensemble model will classify it using majority voting.")

# User input
input_text = st.text_area("Enter a sentence for classification")

if st.button("Classify"):
    if input_text.strip():
        predictions = []
        scores = []

        # Get predictions and scores from all models
        for classifier in models:
            result = classifier(input_text)[0]  # Get first result
            predictions.append(result["label"])  # Extract label
            scores.append(result["score"])  # Extract confidence score

        # Use mode to get the most frequent label
        try:
            final_prediction = statistics.mode(predictions)  # Returns most common label
        except statistics.StatisticsError:
            # If there's a tie, choose the label with the highest average confidence
            unique_labels = set(predictions)
            label_avg_scores = {label: np.mean([scores[i] for i in range(len(predictions)) if predictions[i] == label]) for label in unique_labels}
            final_prediction = max(label_avg_scores, key=label_avg_scores.get)  # Pick label with highest average confidence

        # Display results
        st.write(f"### Ensemble Prediction (Majority Voting): **{final_prediction}**")
        
        # Show individual model results
        for i, (model, label, score) in enumerate(zip(model_names, predictions, scores)):
            st.write(f"**{model}:** {label} (Score: {score:.4f})")

    else:
        st.warning("Please enter some text for classification.")
