# app_transformer.py
import streamlit as st
from transformers import pipeline
from transformers.utils import logging as hf_logging

# Reduce transformers info spam
hf_logging.set_verbosity_error()

st.set_page_config(page_title="Text Summarizer by Akshat & Aditya", layout="centered")
st.title("Text Summarizer by Akshat & Aditya")

st.markdown(
    "Paste text below and click **Summarize**. "
    "**First run** may take time to download the model (one-time)."
)   

# Controls
text = st.text_area("Paste your text here", height=300)
min_length = st.slider("Min tokens in summary", 5, 100, 20)
max_length = st.slider("Max tokens in summary", 20, 200, 80)
sentences_fallback = st.checkbox("Use fast extractive fallback for very short input", value=True)

@st.cache_resource(show_spinner=False)
def get_summarizer():
    # Use a smaller/faster model; change if you prefer other model
    model_name = "sshleifer/distilbart-cnn-12-6"
    return pipeline("summarization", model=model_name, device=-1)  # device=-1 ensures CPU

def extractive_fallback(text, num_sentences=3):
    # Very tiny extractive fallback without extra deps
    # Splits by sentences and returns first N (fast)
    import re
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(sents[:num_sentences]) if sents else text

if st.button("Summarize"):
    if not text.strip():
        st.warning("Please paste some text first.")
    else:
        # quick shortcut for tiny texts
        if len(text.split()) < 30 and sentences_fallback:
            st.info("Short input — using fast extractive fallback.")
            out = extractive_fallback(text, num_sentences=2)
            st.subheader("Summary")
            st.write(out)
            st.download_button("Download summary (.txt)", out, file_name="summary.txt")
        else:
            with st.spinner("Loading model (if first time) and generating summary..."):
                summarizer = get_summarizer()
                # HuggingFace pipeline expects not-too-long inputs; chunk if very long
                try:
                    result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
                    summary = result[0]["summary_text"]
                except ValueError:
                    # fallback if input too long: truncate smartly
                    prompt = " ".join(text.split()[:1000])
                    result = summarizer(prompt, max_length=max_length, min_length=min_length, do_sample=False)
                    summary = result[0]["summary_text"]
            st.subheader("Summary")
            st.write(summary)
            st.download_button("Download summary (.txt)", summary, file_name="summary.txt")
            st.code(summary, language="text")
