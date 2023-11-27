import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@st.cache_resource()
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b")
    model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b")
    model.eval()
    return tokenizer, model
tokenizer, model = load_model()

def predict_next_word(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs.logits

    probs = torch.nn.functional.softmax(predictions[0, -1, :], dim=-1)
    top_k = probs.topk(5)
    top_k_words = [tokenizer.decode([idx]) for idx in top_k.indices]
    top_k_probs = top_k.values.tolist()

    results = {word: prob for word, prob in zip(top_k_words, top_k_probs)}
    return results

st.title("Next Word Prediction Demo")
user_input = st.text_area("Enter your text", "The future of AI is")
if st.button("Predict Next Word"):
    results = predict_next_word(user_input)
    for word, prob in results.items():
        st.write(f"{word}: {prob:.5f}")

st.write("Top 5 probable next words and their probabilities are displayed.")