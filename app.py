"""Streamlit demo for the trained WELFake headline TextCNN."""
from pathlib import Path
import re
import numpy as np
import streamlit as st

from train_fake_news import TextCNN, encode

ROOT = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    words = np.load(ROOT / "vocabulary.npz", allow_pickle=False)["words"].tolist()
    vocab = {word: i for i, word in enumerate(words)}
    archive = np.load(ROOT / "cnn_checkpoint.npz", allow_pickle=False)
    model = TextCNN(len(vocab))
    model.p = {key: archive[key] for key in archive.files}
    return model, vocab

st.set_page_config(page_title="News Headline Classifier", page_icon="📰", layout="centered")
st.title("News Headline Classifier")
st.caption("A demonstration of a CNN trained on WELFake headline labels")
st.warning("This predicts dataset-style labels. It does not fact-check claims or determine whether a headline is true.")

headline = st.text_area("Enter a headline", height=110, placeholder="Type or paste a news headline...")
if st.button("Classify headline", type="primary", disabled=not headline.strip()):
    model, vocab = load_model()
    x = encode([headline], vocab, length=48)
    probability_real = float(model.predict_proba(x)[0])
    prediction = "Likely REAL label" if probability_real >= 0.5 else "Likely FAKE label"
    st.subheader(prediction)
    st.progress(probability_real, text=f"Model probability for REAL label: {probability_real:.1%}")
    st.caption("The threshold is 50%. Probabilities are model scores, not calibrated truth probabilities.")

with st.expander("About this model"):
    st.write("The model is a small TextCNN trained for 8 epochs on WELFake. It uses headline text only, with a maximum of 48 tokens. WELFake contains historical data from multiple sources and may encode publisher, topic, and time-period patterns.")
    st.write("Use this only as an academic demonstration. Verify news through reliable sources and independent fact-checking.")
