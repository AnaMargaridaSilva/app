import streamlit as st
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from huggingface_hub import login
from ST2ModelV2 import ST2ModelV2 



hf_token = st.secrets["HUGGINGFACE_TOKEN"]
login(token=hf_token)

# Load model & tokenizer once (cached for efficiency)
@st.cache_resource
@st.cache_resource
def load_model():
    
    model_name = "anamargarida/Extraction_withseed777"  # Example, update if needed
    config = AutoConfig.from_pretrained("roberta-large")
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    
    
    # Instantiate the custom model
    model = ST2ModelV2(model_name, config)
    
    model.eval()
    return tokenizer, model


# Function to extract cause, effect, and signal arguments
def extract_arguments(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        
    # Debugging: Check what the model returns
    st.write("Model Output:", outputs)  # Full output to examine
    st.write("Model output keys:", outputs.keys())  # Check available keys

    # Extract start/end logits for each argument type
    start_cause_logits = outputs.get("start_arg0_logits", None)
    end_cause_logits = outputs.get("end_arg0_logits", None)
    start_effect_logits = outputs.get("start_arg1_logits", None)
    end_effect_logits = outputs.get("end_arg1_logits", None)
    start_signal_logits = outputs.get("start_sig_logits", None)
    end_signal_logits = outputs.get("end_sig_logits", None)

    if start_cause_logits is not None:
        st.write("Start Cause Logits Shape:", start_cause_logits.shape)
    if end_cause_logits is not None:
        st.write("End Cause Logits Shape:", end_cause_logits.shape)

    # Get start/end token indices
    start_cause = start_cause_logits.argmax().item() if start_cause_logits is not None else None
    end_cause = end_cause_logits.argmax().item() if end_cause_logits is not None else None
    start_effect = start_effect_logits.argmax().item() if start_effect_logits is not None else None
    end_effect = end_effect_logits.argmax().item() if end_effect_logits is not None else None
    start_signal = start_signal_logits.argmax().item() if start_signal_logits is not None else None
    end_signal = end_signal_logits.argmax().item() if end_signal_logits is not None else None

    st.write(f"Start Cause: {start_cause}, End Cause: {end_cause}")
    st.write(f"Start Effect: {start_effect}, End Effect: {end_effect}")
    st.write(f"Start Signal: {start_signal}, End Signal: {end_signal}")

    # Convert token indices to words
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    cause = tokenizer.convert_tokens_to_string(tokens[start_cause:end_cause+1]) if start_cause is not None and end_cause is not None else ""
    effect = tokenizer.convert_tokens_to_string(tokens[start_effect:end_effect+1]) if start_effect is not None and end_effect is not None else ""
    signal = tokenizer.convert_tokens_to_string(tokens[start_signal:end_signal+1]) if start_signal is not None and end_signal is not None else ""

    return cause, effect, signal


"""
# Function to extract cause, effect, and signal arguments
def extract_arguments(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    
    print(outputs.keys())  # See what the model returns
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
st.write("Enter a sentence, and the model will extract the **cause**, **effect**, and **signal**.")
st.write("Model output keys:", outputs.keys())

"""
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

