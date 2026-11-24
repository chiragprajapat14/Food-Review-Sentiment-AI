# 🍔 Food Delivery Review Sentiment Analyzer

An LSTM deep-learning model that classifies Zomato and Swiggy Play Store reviews as **positive** or **negative**, deployed as a Streamlit web app.

**Live demo:** [food-review-sentiment-ai-hwlrwhgrnwrvxw2zrpbgsq.streamlit.app](https://food-review-sentiment-ai-hwlrwhgrnwrvxw2zrpbgsq.streamlit.app/)

> Free Streamlit apps go to sleep when nobody uses them. If you see a "wake up" button, click it and wait a minute.

## Results

Evaluated on a held-out 20% split (93,132 reviews) of the dataset:

| Model | Test accuracy | Macro F1 |
|---|---|---|
| Majority class (always "negative") | 50.2% | n/a |
| SimpleRNN | 91.7% | 0.92 |
| **LSTM (deployed)** | **93.7%** | **0.94** |

LSTM per-class results:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Negative (0) | 0.92 | 0.96 | 0.94 |
| Positive (1) | 0.96 | 0.91 | 0.93 |

The SimpleRNN was unstable during training (validation accuracy dropped to about 67% in one epoch), which is one reason the LSTM was chosen. All numbers come from a single training run.

## How it works

1. **Data:** reviews and star ratings from the Kaggle dataset below.
2. **Labels:** a review with a rating of 3 or higher is labelled positive (1), otherwise negative (0).
3. **Text preparation:** Keras `Tokenizer` (10,000 words, out-of-vocabulary token), then padding to 88 tokens.
4. **Model:** `Embedding(10000, 16)` → `LSTM(32, dropout=0.2)` → `Dense(16, relu)` → `Dense(1, sigmoid)`. Adam optimizer, binary cross-entropy, 10 epochs.
5. **App:** the Streamlit app loads the saved model and tokenizer, predicts the sentiment of a typed review, shows a confidence score, and displays a templated reply.

## Dataset

[Zomato and Swiggy Play Store Reviews](https://www.kaggle.com/datasets/saloni1712/zomato-and-swiggy-play-store-reviews) on Kaggle: 465,660 reviews (Swiggy 200,791 and Zomato 264,869). The data is not stored in this repository. The notebook downloads it with `kagglehub`.

## Project structure

```
.
├── app.py                       # Streamlit app
├── food_sentiment_model.keras   # trained LSTM model
├── tokenizer.pkl                # fitted Keras tokenizer
├── requirements.txt
├── notebook.ipynb               # data loading, training, evaluation
└── README.md
```

## Run locally

```bash
git clone https://github.com/chiragprajapat14/Food-Review-Sentiment-AI.git
cd Food-Review-Sentiment-AI
pip install -r requirements.txt
streamlit run app.py
```

## Retrain the model

Open `notebook.ipynb` in Google Colab and run all cells from top to bottom. The last cell saves `food_sentiment_model.keras` and `tokenizer.pkl`.

## Limitations

- **Noisy labels.** Sentiment comes from star ratings, not human annotation. Sarcasm and mismatched ratings add label noise, and 3-star reviews are counted as positive.
- **English only.** Hinglish and other languages are not supported.
- **Duplicates not removed.** The dataset contains 12,455 duplicate review texts (about 2.7%), so a few short reviews can appear in both the training and test sets.
- **Tokenizer fitted on all data.** It was fitted before the train/test split, which is a small leak. It should be fitted on the training set only.
- **No simple baseline yet.** The LSTM has not been compared with a TF-IDF plus Logistic Regression model.
- **Templated replies.** The auto-reply is a fixed message per sentiment. It does not trigger any workflow.
- **Random split.** The 80/20 split is random and not stratified, which is acceptable here because the classes are nearly balanced.

## Future work

- Remove duplicates and fit the tokenizer on training data only, then re-evaluate.
- Add a TF-IDF plus Logistic Regression baseline.
- Add early stopping and error analysis on misclassified reviews.
- Add a neutral class or drop 3-star reviews.
- Support Hinglish reviews.

## Author

**Chirag Prajapat** · [LinkedIn](https://www.linkedin.com/in/chirag-prajapat14) · [GitHub](https://github.com/chiragprajapat14)
