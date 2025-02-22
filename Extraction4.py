import streamlit as st
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, AutoModel
from ST2ModelV2 import ST2ModelV2
from huggingface_hub import login
import re
import copy

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
            self.model_name = model_name
            self.dropout = 0.1
            self.signal_classification = True
            self.pretrained_signal_detector = False
        
    args = Args()

    # Load the model directly from Hugging Face
    model = ST2ModelV2.from_pretrained(model_name, args=args)

    model.eval()  # Set model to evaluation mode
    return tokenizer, model

# Load the model and tokenizer
tokenizer, model = load_model()


def extract_arguments(text, tokenizer, model, beam_search=True):
    model_name = "anamargarida/Extraction3"  
    class Args:
        def __init__(self):
            self.model_name = model_name
            self.dropout = 0.1
            self.signal_classification = True
            self.pretrained_signal_detector = False
        
    args = Args()

    
    inputs = tokenizer(text, return_tensors="pt")
    

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits
    start_cause_logits = outputs["start_arg0_logits"][0]
    end_cause_logits = outputs["end_arg0_logits"][0]
    start_effect_logits = outputs["start_arg1_logits"][0]
    end_effect_logits = outputs["end_arg1_logits"][0]
    start_signal_logits = outputs.get("start_sig_logits", None)
    end_signal_logits = outputs.get("end_sig_logits", None)

    # Beam Search for position selection
    if beam_search:
        indices1, indices2, _, _, _ = model.beam_search_position_selector(
            start_cause_logits=start_cause_logits,
            end_cause_logits=end_cause_logits,
            start_effect_logits=start_effect_logits,
            end_effect_logits=end_effect_logits,
            topk=5
        )
        start_cause, end_cause, start_effect, end_effect = indices1
    else:
        

        start_cause = start_cause_logits.argmax().item()
        end_cause = end_cause_logits.argmax().item()
        start_effect = start_effect_logits.argmax().item()
        end_effect = end_effect_logits.argmax().item()

    # Signal classification check
    has_signal = True  # Default to True
    if args.signal_classification:
        if not args.pretrained_signal_detector:
            has_signal = outputs["signal_classification_logits"][0].argmax().item()
        else:
            has_signal = signal_detector.predict(text=text)  # External detector

    # Handle signal start/end indices
    if has_signal and start_signal_logits is not None and end_signal_logits is not None:
        start_signal = start_signal_logits.argmax().item()
        end_signal_logits[:start_signal] = -1e4
        end_signal_logits[start_signal + 5:] = -1e4
        end_signal = end_signal_logits.argmax().item()
    else:
        start_signal, end_signal = None, None

    # Convert token indices to words
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    cause = tokenizer.convert_tokens_to_string(tokens[start_cause:end_cause+1]) if start_cause is not None and end_cause is not None else ""
    effect = tokenizer.convert_tokens_to_string(tokens[start_effect:end_effect+1]) if start_effect is not None and end_effect is not None else ""
    signal = tokenizer.convert_tokens_to_string(tokens[start_signal:end_signal+1]) if start_signal is not None and end_signal is not None else ""

    return cause, effect, signal




st.title("Causal Relation Extraction")
input_text = st.text_area("Enter your text here:", height=300)

if st.button("Extract1"):
    if input_text:
        cause, effect, signal = extract_arguments(input_text, tokenizer, model, beam_search=True)

        cause_text = mark_text(input_text, cause, "#FFD700")  # Gold for cause
        effect_text = mark_text(input_text, effect, "#90EE90")  # Light green for effect
        signal_text = mark_text(input_text, signal, "#FF6347")  # Tomato red for signal

        st.markdown(f"**Cause Marked:**<br>{cause_text}", unsafe_allow_html=True)
        st.markdown(f"**Effect Marked:**<br>{effect_text}", unsafe_allow_html=True)
        st.markdown(f"**Signal Marked:**<br>{signal_text}", unsafe_allow_html=True)

