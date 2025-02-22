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
            self.model_name = model_name
            self.dropout = 0.1
            self.signal_classification = False
            self.pretrained_signal_detector = False
            self.beam_search = True  # Enable beam search
            self.topk = 2

    args = Args()

    # Load the model directly from Hugging Face
    model = ST2ModelV2.from_pretrained(model_name, args=args)

    model.eval()  # Set model to evaluation mode
    return tokenizer, model

# Load the model and tokenizer
tokenizer, model = load_model()






import copy

def extract_arguments(text, tokenizer, model, args):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
    word_ids = [idx for idx, offset in enumerate(inputs["offset_mapping"][0]) if offset != (0, 0)]
    attention_mask = inputs["attention_mask"][0]

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits
    start_cause_logits = outputs["start_arg0_logits"][0]
    end_cause_logits = outputs["end_arg0_logits"][0]
    start_effect_logits = outputs["start_arg1_logits"][0]
    end_effect_logits = outputs["end_arg1_logits"][0]
    start_signal_logits = outputs["start_sig_logits"][0] if "start_sig_logits" in outputs else None
    end_signal_logits = outputs["end_sig_logits"][0] if "end_sig_logits" in outputs else None

    # Apply Beam Search if enabled
    if args.beam_search:
        indices1, indices2, _, _, _ = model.beam_search_position_selector(
            start_cause_logits=start_cause_logits,
            end_cause_logits=end_cause_logits,
            start_effect_logits=start_effect_logits,
            end_effect_logits=end_effect_logits,
            attention_mask=attention_mask,
            word_ids=word_ids,
            topk=args.topk,
        )
        start_cause, end_cause, start_effect, end_effect = indices1
    else:
        # Mask out padding tokens for standard argmax selection
        for logits in [start_cause_logits, end_cause_logits, start_effect_logits, end_effect_logits]:
            logits -= (1 - attention_mask) * 1e4

        start_cause = start_cause_logits.argmax().item()
        end_cause = end_cause_logits.argmax().item()
        start_effect = start_effect_logits.argmax().item()
        end_effect = end_effect_logits.argmax().item()

    # Signal classification check
    has_signal = True  # Default to True for now
    if args.signal_classification:
        if not args.pretrained_signal_detector:
            has_signal = outputs["signal_classification_logits"][0].argmax().item()
        else:
            has_signal = signal_detector.predict(text=text)  # External detector

    if has_signal and start_signal_logits is not None and end_signal_logits is not None:
        start_signal = start_signal_logits.argmax().item()
        end_signal_logits[:start_signal] = -1e4
        end_signal_logits[start_signal + 5:] = -1e4
        end_signal = end_signal_logits.argmax().item()
    else:
        start_signal, end_signal = None, None

    # Convert token IDs back to words
    space_splitted_tokens = text.split()

    def wrap_tokens(start, end, tag):
        """Helper function to wrap extracted tokens with tags."""
        if start is not None and end is not None and start < len(space_splitted_tokens) and end < len(space_splitted_tokens):
            space_splitted_tokens[start] = f"<{tag}>" + space_splitted_tokens[start]
            space_splitted_tokens[end] = space_splitted_tokens[end] + f"</{tag}>"

    # Wrap detected entities
    wrap_tokens(start_cause, end_cause, "ARG0")
    wrap_tokens(start_effect, end_effect, "ARG1")
    if has_signal:
        wrap_tokens(start_signal, end_signal, "SIG0")

    result_text = " ".join(space_splitted_tokens)
    return result_text



st.title("Causal Relation Extraction")
input_text = st.text_area("Enter your text here:", height=300)

if st.button("Extract"):
    highlighted_text = extract_arguments(input_text, tokenizer, model, args)
    st.markdown(f"**Extracted Text:** {highlighted_text}")
