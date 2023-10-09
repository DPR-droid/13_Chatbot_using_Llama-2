# https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/

import streamlit as st
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Added for local install
import os

# Added to remove warnings
import warnings
warnings.filterwarnings('ignore')


# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")


# Define a dictionary with model names and their associated paths
model_paths = {
    'Python-13B-V1.0-GPTQ': 'TheBloke/WizardCoder-Python-13B-V1.0-GPTQ',
    'Llama-2-7b-Chat-GPTQ': 'TheBloke/Llama-2-7b-Chat-GPTQ'
}

# Define a function to download and save a model
def download_model(selected_model):
    model_name = model_paths[selected_model]
    output_model_dir = f"models/{selected_model}"

    # Define the required files
    required_files = ['pytorch_model.bin', 'tokenizer.json', 'tokenizer.model']

    # Check if the model is already downloaded
    if os.path.exists(output_model_dir):
        # Check if all required files exist in the directory
        if all(os.path.exists(os.path.join(output_model_dir, file)) for file in required_files):
            st.sidebar.success(f"{selected_model} is already downloaded.")
            return
        else:
            st.sidebar.warning(f"{selected_model} is partially downloaded. Some files are missing.")

    # Download and save the model locally
    st.sidebar.info(f"Downloading {selected_model}...")
    os.makedirs(output_model_dir, exist_ok=True)
    loaded_model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save the model and tokenizer to the output directory
    loaded_model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    st.sidebar.success(f"{selected_model} has been downloaded.")

# Create a Streamlit sidebar
st.sidebar.title('Models and Parameters')

# Create a dropdown to select 
selected_model = st.sidebar.selectbox('Choose a model', list(model_paths.keys()))

# Create sliders for model parameters (temperature, top_p, max_length)
temperature = st.sidebar.slider('Temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
top_p = st.sidebar.slider('Top P', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
max_length = st.sidebar.slider('Max Length', min_value=32, max_value=128, value=120, step=8)

# Create a button to download the selected model
if st.sidebar.button('Download Model'):
    download_model(selected_model)

