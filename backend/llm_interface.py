import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"

@st.cache_resource
def get_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

pipe = get_pipeline()

def generate_answer(context, query, history=None):
    if history:
        history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in history[-2:]])
    else:
        history_text = ""

    prompt = f"{history_text}\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    output = pipe(prompt, max_new_tokens=256)[0]["generated_text"]
    return output.strip()