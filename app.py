import streamlit as st
from transformers import pipeline
import torch
from huggingface_hub import login

hf_token = st.secrets["HUGGINGFACE_TOKEN"]
login(token=hf_token)

#!git config --global user.name "anamargarida"
#!git config --global user.email "anamargaridasilva320@gmail.com"

model_name = "anamargarida/my_model_larger_dataset"
device = 0 if torch.cuda.is_available() else -1 
classifier = pipeline('text-classification', model=model_name, device=device)

# Streamlit interface
st.title("Signal Detection with One Model")
input_text = st.text_area("Enter a sentence for classification")

if st.button("Classify"):
    result = classifier(input_text)
    st.write(result)
