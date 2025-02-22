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
    model_name = "anamargarida/Extraction2"  # Update if needed
    
    # Load the configuration and tokenizer from Hugging Face Hub
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create an args object with necessary parameters
    class Args:
        def __init__(self):
            self.model_name_or_path = model_name
            self.dropout = 0.1  # Example dropout value
            self.mlp = False
            self.add_signal_bias = False
            self.signal_classification = False
            self.pretrained_signal_detector = False

    args = Args()

    # Instantiate the model with config
    model = ST2ModelV2(args, config)

    # Load model weights
    model_path = "path_to_model_weights.pth"  # Update this path
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    model.eval()  # Set model to evaluation mode
    return tokenizer, model

# Load the model and tokenizer
tokenizer, model = load_model()

# Now you can use the model to predict text input from the user
st.title("Text Extraction Model")

input_text = st.text_area("Enter your text here:", height=300)

if st.button("Extract"):
    if input_text:
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Perform inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract and display the logits (adjust as needed for your use case)
        start_arg0_logits = outputs["start_arg0_logits"]
        end_arg0_logits = outputs["end_arg0_logits"]
        
        # Display results
        st.write("Start Arg0 logits:", start_arg0_logits)
        st.write("End Arg0 logits:", end_arg0_logits)

    
   
