import streamlit as st
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, AutoModel
from ST2ModelV2 import ST2ModelV2
from huggingface_hub import login

hf_token = st.secrets["HUGGINGFACE_TOKEN"]
login(token=hf_token)

# Load model & tokenizer once (cached for efficiency)
@st.cache_resource
def load_model():
    model_name = "anamargarida/Extraction2"  # Update with your actual model name
    
    # Load the configuration and tokenizer from Hugging Face Hub
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define arguments as needed
    class Args:
        def __init__(self):
            self.model_name_or_path = model_name
            self.dropout = 0.1
            self.mlp = False
            self.add_signal_bias = False
            self.signal_classification = False
            self.pretrained_signal_detector = False

    args = Args()

    # Load the model directly from Hugging Face
    model = ST2ModelV2.from_pretrained(model_name, args=args)

    model.eval()  # Set model to evaluation mode
    return tokenizer, model

# Load the model and tokenizer
tokenizer, model = load_model()

# Now you can use the model to predict text input from the user
st.title("Causal Relation Extraction")

input_text = st.text_area("Enter your text here:", height=300)

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


if st.button("Extract Arguments"):
    if input_text:
        cause, effect, signal = extract_arguments(input_text, tokenizer, model)

        # Highlight extracted arguments in the original text
        highlighted_text = input_text
        if cause:
            highlighted_text = highlighted_text.replace(cause, f"<span style='color:blue; font-weight:bold;'>{cause}</span>")
        if effect:
            highlighted_text = highlighted_text.replace(effect, f"<span style='color:green; font-weight:bold;'>{effect}</span>")
        if signal:
            highlighted_text = highlighted_text.replace(signal, f"<span style='color:red; font-weight:bold;'>{signal}</span>")

        st.markdown(f"**Extracted Arguments:**<br>{highlighted_text}", unsafe_allow_html=True)
    else:
        st.warning("Please enter some text before extracting.")

"""
if st.button("Extract"):
    if input_text:
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Perform inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        
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
"""

