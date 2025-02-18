import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

hf_token = st.secrets["HUGGINGFACE_TOKEN"]
login(token=hf_token)

# Load model & tokenizer once (cached for efficiency)
@st.cache_resource
def load_model():
    model_name = "anamargarida/Extraction_withseed777"
    
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")  
    
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

# Function to extract cause, effect, and signal arguments
def extract_arguments(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract start/end logits for each argument type
    start_cause_logits = outputs["start_arg0_logits"]
    end_cause_logits = outputs["end_arg0_logits"]
    start_effect_logits = outputs["start_arg1_logits"]
    end_effect_logits = outputs["end_arg1_logits"]
    start_signal_logits = outputs["start_sig_logits"]
    end_signal_logits = outputs["end_sig_logits"]

    # Get start/end token indices
    start_cause = start_cause_logits.argmax().item()
    end_cause = end_cause_logits.argmax().item()
    start_effect = start_effect_logits.argmax().item()
    end_effect = end_effect_logits.argmax().item()
    start_signal = start_signal_logits.argmax().item()
    end_signal = end_signal_logits.argmax().item()

    # Convert token indices to words
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    cause = tokenizer.convert_tokens_to_string(tokens[start_cause:end_cause+1])
    effect = tokenizer.convert_tokens_to_string(tokens[start_effect:end_effect+1])
    signal = tokenizer.convert_tokens_to_string(tokens[start_signal:end_signal+1])

    return cause, effect, signal

# Streamlit UI
st.title("Causal Argument Extraction")
st.write("Enter a sentence, and the model will extract the **cause**, **effect**, and **causal signal**.")

# Text input
user_input = st.text_area("Enter text:", "Burning fossil fuels causes global warming due to carbon emissions.")

# Load model & tokenizer
tokenizer, model = load_model()

if st.button("Extract Arguments"):
    if user_input:
        cause, effect, signal = extract_arguments(user_input, tokenizer, model)

        # Highlight extracted arguments in the original text
        highlighted_text = user_input
        if cause:
            highlighted_text = highlighted_text.replace(cause, f"<span style='color:blue; font-weight:bold;'>{cause}</span>")
        if effect:
            highlighted_text = highlighted_text.replace(effect, f"<span style='color:green; font-weight:bold;'>{effect}</span>")
        if signal:
            highlighted_text = highlighted_text.replace(signal, f"<span style='color:red; font-weight:bold;'>{signal}</span>")

        st.markdown(f"**Extracted Arguments:**<br>{highlighted_text}", unsafe_allow_html=True)
    else:
        st.warning("Please enter some text before extracting.")
