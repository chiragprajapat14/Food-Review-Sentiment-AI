import pickle

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAXLEN = 88  # must match the padding length used in training

st.set_page_config(page_title="Food Review AI", page_icon="🍔", layout="centered")


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("food_sentiment_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


model, token = load_assets()

st.title("🍔 Food Delivery Review Sentiment Analyzer")
st.markdown("LSTM model trained on **465K Zomato & Swiggy** Play Store reviews (English only).")

user_text = st.text_area(
    "Enter your food review in English:",
    placeholder="e.g., Pizza was very hot and delicious!",
    height=120,
)

if st.button("Analyze Sentiment", type="primary"):
    if not user_text.strip():
        st.warning("Please enter a review first.")
    else:
        seq = token.texts_to_sequences([user_text])
        # same padding settings as training (post-padding, default truncation)
        padded = pad_sequences(seq, padding="post", maxlen=MAXLEN)
        prob = float(model.predict(padded, verbose=0)[0][0])  # [0][0] -> plain number

        if prob >= 0.5:
            st.success(f"😊 Positive review (confidence: {prob * 100:.1f}%)")
            st.info("Auto-reply: Thank you so much for ordering! We are glad you enjoyed it.")
        else:
            st.error(f"😞 Negative review (confidence: {(1 - prob) * 100:.1f}%)")
            st.warning("Auto-reply: We are sorry about your experience and will look into it.")
