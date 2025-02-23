import streamlit as st
from transformers import AutoTokenizer
import sentencepiece as spm
import numpy as np

def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

def tokenize_text(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, token_ids

# Streamlit UI
st.title("🔠 Tokenizer Playground")
st.write("Compare different tokenizers and understand how text is processed in LLMs.")

# Select tokenizer model
model_options = ["bert-base-uncased", "gpt2", "xlm-roberta-base"]
model_name = st.selectbox("Choose a tokenizer model:", model_options)

tokenizer = load_tokenizer(model_name)

# User input text
user_text = st.text_area("Enter text to tokenize:", "Hello, how are you?")

if user_text:
    tokens, token_ids = tokenize_text(tokenizer, user_text)
    
    st.subheader("Tokenized Output")
    st.write("**Tokens:**", tokens)
    st.write("**Token IDs:**", token_ids)
    
    # Visualization
    st.subheader("Visualization")
    st.write("Token breakdown:")
    token_table = np.array([tokens, token_ids]).T
    st.table(token_table)
    
    st.write("Byte-level details (for BPE models):")
    byte_tokens = [token.encode('utf-8') for token in tokens]
    st.write(byte_tokens)

st.write("Try different models and see how tokenization changes!'")
