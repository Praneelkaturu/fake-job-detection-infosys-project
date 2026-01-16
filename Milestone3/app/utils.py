# app/utils.py
import joblib
import time
import numpy as np
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertForSequenceClassification
from scipy.special import softmax

# ------------------- Paths -------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ------------------- Load classical ML models -------------------
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
best_lr = joblib.load(os.path.join(MODEL_DIR, "logistic_regression_v1.pkl"))

# ------------------- Load BiLSTM model -------------------
bilstm = load_model(os.path.join(MODEL_DIR, "bilstm_model_v1.keras"))

# ❗ IMPORTANT: tokenizer must be saved during training
TOKENIZER_PATH = os.path.join(MODEL_DIR, "bilstm_tokenizer.pkl")
tokenizer = None
if os.path.exists(TOKENIZER_PATH):
    tokenizer = joblib.load(TOKENIZER_PATH)

MAX_LEN = 200

# ------------------- Load BERT -------------------
BERT_DIR = os.path.join(MODEL_DIR, "bert_model_v1")
bert_tokenizer = BertTokenizer.from_pretrained(BERT_DIR)
bert = TFBertForSequenceClassification.from_pretrained(
    BERT_DIR,
    id2label={0: "Real", 1: "Fake"},
    label2id={"Real": 0, "Fake": 1}
)

# ------------------- Prediction function -------------------
def predict_job(text: str, model: str = "BERT"):
    start_time = time.time()

    if model == "BERT":
        enc = bert_tokenizer(
            [text],
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="tf"
        )

        outputs = bert(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            training=False
        )

        logits = outputs.logits.numpy()[0]
        probs = softmax(logits)

        pred = int(np.argmax(probs))
        confidence = float(np.max(probs))

    elif model == "LogisticRegression":
        vec = tfidf.transform([text])
        pred = int(best_lr.predict(vec)[0])
        confidence = float(np.max(best_lr.predict_proba(vec)))

    elif model == "BiLSTM":
        if tokenizer is None:
            raise RuntimeError("BiLSTM tokenizer not found. Save tokenizer during training.")

        seq = tokenizer.texts_to_sequences([text])
        pad_seq = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
        prob = bilstm.predict(pad_seq, verbose=0).ravel()[0]

        pred = int(prob > 0.5)
        confidence = float(prob if pred == 1 else 1 - prob)

    else:
        raise ValueError("Unsupported model type")

    return {
        "prediction": "Fake" if pred == 1 else "Real",
        "confidence": round(confidence * 100, 2),
        "processing_time": round(time.time() - start_time, 3),
        "timestamp": datetime.now().isoformat()
    }
