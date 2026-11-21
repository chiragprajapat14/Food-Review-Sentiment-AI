# 🍔 Food Delivery Review Sentiment Analyzer

An end-to-end **Natural Language Processing (NLP)** and **Deep Learning** web application that classifies customer food delivery reviews into **Positive** or **Negative** sentiments and generates automated customer support responses.

## 📊 Dataset & Model Overview
* **Dataset:** 465,000+ real-world Indian food delivery reviews combined from **Zomato & Swiggy** datasets.
* **Model Architecture:** Sequential Deep Learning model built with **TensorFlow / Keras** (`Embedding` + `LSTM` + `Dense` layers).
* **Performance:** Achieved **94% Test Accuracy** with strong F1-scores across both Positive (`0.90`) and Negative (`0.95`) classes.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning & NLP:** TensorFlow, Keras (`Tokenizer`, `pad_sequences`), NumPy, Pandas
* **Evaluation:** Scikit-Learn (Classification Report & Confusion Matrix), Seaborn, Matplotlib
* **Deployment:** Streamlit Community Cloud

## 🚀 Features
* Real-time sentiment prediction with confidence score.
* Automated customer support response generation based on review sentiment.
* Lightweight, interactive web interface built using Streamlit.

## 📂 Project Structure
```text
├── app.py                        # Streamlit web application
├── food_sentiment_model.keras    # Trained LSTM model
├── tokenizer.pkl                 # Fitted Keras text tokenizer
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 💻 Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/chiragprajapat14/Food-Review-Sentiment-AI.git
   cd Food-Review-Sentiment-AI
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
