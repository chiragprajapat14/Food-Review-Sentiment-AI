import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Food Review AI", page_icon="🍔", layout="centered")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("food_sentiment_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, token = load_assets()

st.title("🍔 Food Delivery Review Sentiment Analyzer")
st.markdown("Trained on **4.6 Lakh+ Zomato & Swiggy** customer reviews.")

user_text = st.text_area(
    "Enter your food review in English:",
    placeholder="e.g., Pizza was very hot and delicious!",
    height=120
)

if st.button("Analyze Sentiment", type="primary"):
    if not user_text.strip():
        st.warning("⚠️ Please enter a review first!")
    else:
        seq = token.texts_to_sequences([user_text.lower()])
        padded = pad_sequences(seq, padding="post", maxlen=88, truncating="post")
        
        # Fix: 2D array ko single float number mein convert kiya hai
        score = float(model.predict(padded, verbose=0)[0][0])
        
        if score >= 0.5:
            st.success(f"😊 Positive Review (Confidence: {score*100:.2f}%)")
            st.info("💬 Auto-Reply: Thank you so much for ordering! Keep loving our service.")
        else:
            st.error(f"😞 Negative Review (Confidence: {(1-score)*100:.2f}%)")
            st.warning("💬 Auto-Reply: We are extremely sorry. Your issue has been escalated to support team.")

