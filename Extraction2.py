import streamlit as st
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, AutoModel
from ST2ModelV2 import ST2ModelV2
from huggingface_hub import login
import re

hf_token = st.secrets["HUGGINGFACE_TOKEN"]
login(token=hf_token)

# Load model & tokenizer once (cached for efficiency)
@st.cache_resource
def load_model():
    
    model_name = "anamargarida/Extraction3"  
    
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    
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


st.title("Causal Relation Extraction")
input_text = st.text_area("Enter your text here:", height=300)

# Function to extract cause, effect, and signal arguments
def extract_arguments(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        
    # Debugging: Check what the model returns
    # st.write("Model Output:", outputs)  # Full output to examine
    # st.write("Model output keys:", outputs.keys())  # Check available keys

    # Extract start/end logits for each argument type
    start_cause_logits = outputs.get("start_arg0_logits", None)
    end_cause_logits = outputs.get("end_arg0_logits", None)
    start_effect_logits = outputs.get("start_arg1_logits", None)
    end_effect_logits = outputs.get("end_arg1_logits", None)
    start_signal_logits = outputs.get("start_sig_logits", None)
    end_signal_logits = outputs.get("end_sig_logits", None)

    
    # if start_cause_logits is not None:
        #st.write("Start Cause Logits Shape:", start_cause_logits.shape)
    #if end_cause_logits is not None:
        #st.write("End Cause Logits Shape:", end_cause_logits.shape)

    
    
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



def highlight_text(original_text, span, color):
    """Replace the extracted span with a highlighted version in a new sentence."""
    if span:
        return re.sub(re.escape(span), f"<span style='color:{color}; font-weight:bold;'>{span}</span>", original_text, flags=re.IGNORECASE)
    return original_text  # Return unchanged text if no span is found

if st.button("Extract"):
    if input_text:
        cause, effect, signal = extract_arguments(input_text, tokenizer, model)

        # Generate separate sentences for each highlight
        cause_sentence = highlight_text(input_text, cause, "blue")  # Cause in blue
        effect_sentence = highlight_text(input_text, effect, "green")  # Effect in green
        signal_sentence = highlight_text(input_text, signal, "red")  # Signal in red

        # Display sentences separately
        st.markdown(f"**Cause Highlighted:**<br>{cause_sentence}", unsafe_allow_html=True)
        st.markdown(f"**Effect Highlighted:**<br>{effect_sentence}", unsafe_allow_html=True)
        st.markdown(f"**Signal Highlighted:**<br>{signal_sentence}", unsafe_allow_html=True)
    else:
        st.warning("Please enter some text before extracting.")

def mark_text(original_text, span, color):
    """Replace extracted span with a colored background marker."""
    if span:
        return re.sub(re.escape(span), f"<mark style='background-color:{color}; padding:2px; border-radius:4px;'>{span}</mark>", original_text, flags=re.IGNORECASE)
    return original_text  # Return unchanged text if no span is found

if st.button("Extract"):
    if input_text:
        cause, effect, signal = extract_arguments(input_text, tokenizer, model)

        cause_text = mark_text(input_text, cause, "#FFD700")  # Gold for cause
        effect_text = mark_text(input_text, effect, "#90EE90")  # Light green for effect
        signal_text = mark_text(input_text, signal, "#FF6347")  # Tomato red for signal

        st.markdown(f"**Cause Marked:**<br>{cause_text}", unsafe_allow_html=True)
        st.markdown(f"**Effect Marked:**<br>{effect_text}", unsafe_allow_html=True)
        st.markdown(f"**Signal Marked:**<br>{signal_text}", unsafe_allow_html=True)

        st.markdown("""
        ### **Legend: Color Markers for Extracted Components**
        <table style="border-collapse: collapse; width: 100%;">
            <tr>
                <th style="text-align: left; padding: 5px; border: 1px solid black;">Component</th>
                <th style="text-align: left; padding: 5px; border: 1px solid black;">Color</th>
            </tr>
            <tr>
                <td style="padding: 5px; border: 1px solid black;">Cause</td>
                <td style="background-color: #FFD700; padding: 5px; border-radius: 4px;">🟡 Gold (#FFD700)</td>
            </tr>
            <tr>
                <td style="padding: 5px; border: 1px solid black;">Effect</td>
                <td style="background-color: #90EE90; padding: 5px; border-radius: 4px;">🟢 Light Green (#90EE90)</td>
            </tr>
            <tr>
                <td style="padding: 5px; border: 1px solid black;">Signal</td>
                <td style="background-color: #FF6347; padding: 5px; border-radius: 4px;">🔴 Tomato Red (#FF6347)</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please enter some text before extracting.")



#if st.button("Extract"):
    #if input_text:
        # Tokenize input
        #inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        #input_ids = inputs["input_ids"]
        #attention_mask = inputs["attention_mask"]

        # Perform inference
        #with torch.no_grad():
            #outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        
