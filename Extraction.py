import streamlit as st
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer
from ST2ModelV2 import ST2ModelV2

# Load model & tokenizer once (cached for efficiency)
@st.cache_resource
def load_model():
    model_name = "anamargarida/Extraction"  # Example model name, update if needed
    
    # Load the configuration and tokenizer from Hugging Face Hub
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load the weights from the safetensors file
    safetensor_path = "path_to_your_model/model.safetensor"  # Adjust the path to your model
    state_dict = load_file(safetensor_path)  # Load model weights from safetensors

    # Pass the necessary args to the model constructor (example: args with dropout)
    args = type('', (), {})()  # Creating a dummy object for args
    args.model_name_or_path = model_name
    args.dropout = 0.1  # Example dropout value, you can adjust as necessary
    args.mlp = False  # Adjust according to your needs
    args.add_signal_bias = False  # Adjust as needed
    args.signal_classification = False  # Adjust as needed
    args.pretrained_signal_detector = False  # Adjust as needed
    
    # Instantiate the model with config
    model = ST2ModelV2(args, config)  # Assuming the model accepts the config
    
    # Load the model weights into your model
    model.load_state_dict(state_dict)

    model.eval()  # Set the model to evaluation mode
    
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

    
   
