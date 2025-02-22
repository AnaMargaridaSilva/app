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
     
    class Args:
        def __init__(self):
            self.signal_classification = True
            self.pretrained_signal_detector = False
        
    args = Args()
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)

    #st.write("Model output keys:", outputs.keys())

    # Extract logits
    start_cause_logits = outputs["start_arg0_logits"][0]
    end_cause_logits = outputs["end_arg0_logits"][0]
    start_effect_logits = outputs["start_arg1_logits"][0]
    end_effect_logits = outputs["end_arg1_logits"][0]
    start_signal_logits = outputs.get("start_sig_logits", None)
    end_signal_logits = outputs.get("end_sig_logits", None)

    #st.write("start_cause_logits", start_cause_logits)
    #st.write("end_cause_logits", end_cause_logits)
    #st.write("start_effect_logits", start_effect_logits)
    #st.write("end_effect_logits", end_effect_logits)
    #st.write("start_signal_logits", start_signal_logits)
    #st.write("end_signal_logits", end_signal_logits)


    # Set the first and last token logits to a very low value to ignore them
    start_cause_logits[0] = -1e-4
    end_cause_logits[0] = -1e-4
    start_effect_logits[0] = -1e-4
    end_effect_logits[0] = -1e-4
    start_cause_logits[len(inputs["input_ids"][0]) - 1] = -1e-4
    end_cause_logits[len(inputs["input_ids"][0]) - 1] = -1e-4
    start_effect_logits[len(inputs["input_ids"][0]) - 1] = -1e-4
    end_effect_logits[len(inputs["input_ids"][0]) - 1] = -1e-4

    st.write("start_cause_logits", start_cause_logits)
    st.write("end_cause_logits", end_cause_logits)
    st.write("start_effect_logits", start_effect_logits)
    st.write("end_effect_logits", end_effect_logits)
    st.write("start_signal_logits", start_signal_logits)
    st.write("end_signal_logits", end_signal_logits)
    
    # Beam Search for position selection
    if beam_search:
        indices1, indices2, _, _, _ = model.beam_search_position_selector(
            start_cause_logits=start_cause_logits,
            end_cause_logits=end_cause_logits,
            start_effect_logits=start_effect_logits,
            end_effect_logits=end_effect_logits,
            topk=5
        )
        start_cause1, end_cause1, start_effect1, end_effect1 = indices1
        start_cause2, end_cause2, start_effect2, end_effect2 = indices2
    else:
        start_cause1 = start_cause_logits.argmax().item()
        end_cause1 = end_cause_logits.argmax().item()
        start_effect1 = start_effect_logits.argmax().item()
        end_effect1 = end_effect_logits.argmax().item()

        start_cause2, end_cause2, start_effect2, end_effect2 = None, None, None, None

    # Signal classification check
    has_signal = outputs.get("signal_classification_logits", None)
    if has_signal is not None:
        has_signal = has_signal[0].argmax().item()  

    start_signal1, end_signal1, start_signal2, end_signal2 = None, None, None, None
    if has_signal and start_signal_logits is not None and end_signal_logits is not None:
        start_signal1 = start_signal_logits.argmax().item()
        end_signal_logits[:start_signal1] = -1e4
        end_signal_logits[start_signal1 + 5:] = -1e4
        end_signal1 = end_signal_logits.argmax().item()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    token_ids = inputs["input_ids"][0]

    st.write("Token Positions, IDs, and Corresponding Tokens:")
    for position, (token_id, token) in enumerate(zip(token_ids, tokens)):
        st.write(f"Position: {position}, ID: {token_id}, Token: {token}")

    def extract_span(start, end):
        return tokenizer.convert_tokens_to_string(tokens[start:end+1]) if start is not None and end is not None else ""

    cause1 = extract_span(start_cause1, end_cause1)
    cause2 = extract_span(start_cause2, end_cause2)
    effect1 = extract_span(start_effect1, end_effect1)
    effect2 = extract_span(start_effect2, end_effect2)
    signal1 = extract_span(start_signal1, end_signal1)
    signal2 = extract_span(start_signal2, end_signal2)

    list1 = [start_cause1, end_cause1, start_effect1, end_effect1, start_signal1, end_signal1]
    list2 = [start_cause2, end_cause2, start_effect2, end_effect2, start_signal2, end_signal2]
    return cause1, cause2, effect1, effect2, signal1, signal2, list1, list2

def mark_text(original_text, span, color):
    """Replace extracted span with a colored background marker."""
    if span:
        return re.sub(re.escape(span), f"<mark style='background-color:{color}; padding:2px; border-radius:4px;'>{span}</mark>", original_text, flags=re.IGNORECASE)
    return original_text  # Return unchanged text if no span is found

st.title("Causal Relation Extraction")
input_text = st.text_area("Enter your text here:", height=300)
beam_search = st.radio("Enable Beam Search?", ('No', 'Yes')) == 'Yes'


if st.button("Extract1"):
    if input_text:
        cause1, cause2, effect1, effect2, signal1, signal2, list1, list2 = extract_arguments(input_text, tokenizer, model, beam_search=beam_search)

        cause_text1 = mark_text(input_text, cause1, "#FFD700")  # Gold for cause
        effect_text1 = mark_text(input_text, effect1, "#90EE90")  # Light green for effect
        signal_text1 = mark_text(input_text, signal1, "#FF6347")  # Tomato red for signal

        st.markdown(f"<span style='font-size: 24px;'><strong>Relation 1:</strong></span>", unsafe_allow_html=True)
        st.markdown(f"**Cause:**<br>{cause_text1}", unsafe_allow_html=True)
        st.markdown(f"**Effect:**<br>{effect_text1}", unsafe_allow_html=True)
        st.markdown(f"**Signal:**<br>{signal_text1}", unsafe_allow_html=True)

        st.write("List 1:", list1)

        if beam_search:

            cause_text2 = mark_text(input_text, cause2, "#FFD700")  # Gold for cause
            effect_text2 = mark_text(input_text, effect2, "#90EE90")  # Light green for effect
            signal_text2 = mark_text(input_text, signal2, "#FF6347")  # Tomato red for signal
    
            st.markdown(f"<span style='font-size: 24px;'><strong>Relation 2:</strong></span>", unsafe_allow_html=True)
            st.markdown(f"**Cause:**<br>{cause_text2}", unsafe_allow_html=True)
            st.markdown(f"**Effect:**<br>{effect_text2}", unsafe_allow_html=True)
            st.markdown(f"**Signal:**<br>{signal_text2}", unsafe_allow_html=True)

            st.write("List 2:", list2)
    else:
        st.warning("Please enter some text before extracting.")
