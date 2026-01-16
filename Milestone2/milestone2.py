import pandas as pd
import numpy as np
import time
import json
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Use transformers <5.0 for TensorFlow compatibility
from transformers import BertTokenizer, TFBertForSequenceClassification

# ------------------- Settings -------------------
RUN_BERT = True
DATA_FILE = r"C:\Users\Vijay\cleaned_file.csv"

# ------------------- Load Data -------------------
df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

if 'text_combined' not in df.columns:
    text_columns = [col for col in df.columns if col not in ['fraudulent']]
    if len(text_columns) == 0:
        raise ValueError("No text columns found to combine.")
    df['text_combined'] = df[text_columns].astype(str).agg(' '.join, axis=1)

X = df['text_combined'].astype(str)
y = df['fraudulent'].astype(int)

# ------------------- Train/Test Split -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------- TF-IDF -------------------
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

# ------------------- Logistic Regression -------------------
lr_params = {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "solver": ["liblinear"]}
lr = LogisticRegression()
lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring="f1")

start = time.time()
lr_grid.fit(X_train_tfidf, y_train)
lr_time = time.time() - start

best_lr = lr_grid.best_estimator_
joblib.dump(best_lr, "logistic_regression_v1.pkl")

lr_preds = best_lr.predict(X_test_tfidf)
lr_probs = best_lr.predict_proba(X_test_tfidf)[:, 1]

lr_acc = accuracy_score(y_test, lr_preds)
lr_auc = roc_auc_score(y_test, lr_probs)
lr_cv = cross_val_score(best_lr, X_train_tfidf, y_train, cv=5, scoring="f1")

print("\nLogistic Regression Report")
print(classification_report(y_test, lr_preds))
print(confusion_matrix(y_test, lr_preds))

# ------------------- Random Forest -------------------
rf_params = {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}
rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring="f1")
rf_grid.fit(X_train_tfidf, y_train)

best_rf = rf_grid.best_estimator_
joblib.dump(best_rf, "random_forest_v1.pkl")

rf_preds = best_rf.predict(X_test_tfidf)
rf_probs = best_rf.predict_proba(X_test_tfidf)[:, 1]

rf_acc = accuracy_score(y_test, rf_preds)
rf_auc = roc_auc_score(y_test, rf_probs)
rf_cv = cross_val_score(best_rf, X_train_tfidf, y_train, cv=5, scoring="f1")

print("\nRandom Forest Report")
print(classification_report(y_test, rf_preds))
print(confusion_matrix(y_test, rf_preds))

# Feature importance
feature_importances = pd.Series(best_rf.feature_importances_, index=tfidf.get_feature_names_out())
plt.figure(figsize=(8,6))
feature_importances.nlargest(20).plot(kind='barh')
plt.title("Top 20 Feature Importances - Random Forest")
plt.show()

# ------------------- BiLSTM -------------------
y_train_lstm = y_train.to_numpy().astype(np.float32).reshape(-1, 1)
y_test_lstm  = y_test.to_numpy().astype(np.float32).reshape(-1, 1)

tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = 200
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post")

vocab_size = len(tokenizer.word_index) + 1

bilstm = Sequential([
    Embedding(vocab_size, 128, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dense(1, activation="sigmoid")
])

bilstm.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history = bilstm.fit(
    X_train_pad,
    y_train_lstm,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=[early_stop]
)

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title("BiLSTM Loss")
plt.legend()
plt.show()

dl_loss, dl_acc = bilstm.evaluate(X_test_pad, y_test_lstm)
dl_probs = bilstm.predict(X_test_pad).ravel()
dl_auc = roc_auc_score(y_test_lstm, dl_probs)
dl_preds = (dl_probs > 0.5).astype(int)

print("\nBiLSTM Report")
print(classification_report(y_test_lstm, dl_preds))
print(confusion_matrix(y_test_lstm, dl_preds))

bilstm.save("bilstm_model_v1.keras")

# ------------------- BERT -------------------
bert_acc = None
bert_auc = None

if RUN_BERT:
    # Labels must be 1D int array for BERT
    y_train_bert = y_train.to_numpy().astype(np.int32)
    y_test_bert  = y_test.to_numpy().astype(np.int32)

    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = TFBertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    train_enc = bert_tokenizer(
        X_train.tolist(),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="tf"
    )
    test_enc = bert_tokenizer(
        X_test.tolist(),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="tf"
    )

    bert.compile(
        optimizer=Adam(learning_rate=2e-5),
        metrics=["accuracy"]
    )

    bert.fit(
        x={"input_ids": train_enc['input_ids'], "attention_mask": train_enc['attention_mask']},
        y=y_train_bert,
        validation_split=0.2,
        epochs=3,
        batch_size=16
    )

    bert_eval = bert.evaluate(
        x={"input_ids": test_enc['input_ids'], "attention_mask": test_enc['attention_mask']},
        y=y_test_bert
    )

    bert_logits = bert.predict(
        {"input_ids": test_enc['input_ids'], "attention_mask": test_enc['attention_mask']}
    ).logits
    bert_preds = bert_logits.argmax(axis=1)

    bert_acc = bert_eval[1]
    bert_auc = roc_auc_score(y_test_bert, bert_logits[:, 1])

    print("\nBERT Report")
    print(classification_report(y_test_bert, bert_preds))
    print(confusion_matrix(y_test_bert, bert_preds))

    bert.save_pretrained("bert_model_v1")
    bert_tokenizer.save_pretrained("bert_model_v1")

# ------------------- Model Comparison -------------------
comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "BiLSTM", "BERT"],
    "Accuracy": [lr_acc, rf_acc, dl_acc, bert_acc],
    "ROC_AUC": [lr_auc, rf_auc, dl_auc, bert_auc],
    "F1": [
        lr_cv.mean(),
        rf_cv.mean(),
        f1_score(y_test_lstm, dl_preds),
        f1_score(y_test_bert, bert_preds) if bert_acc else None
    ]
})

comparison.to_csv("model_comparison.csv", index=False)
print(comparison)

# ------------------- Metadata -------------------
metadata = {
    "selected_model": "BERT",
    "version": "v1.0",
    "timestamp": datetime.now().isoformat()
}

with open("model_metadata_v1.json", "w") as f:
    json.dump(metadata, f, indent=4)

# ------------------- Prediction Function -------------------
def predict_job(text):
    if RUN_BERT and bert_acc is not None:
        enc = bert_tokenizer(
            [text],
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="tf"
        )
        logits = bert.predict({"input_ids": enc['input_ids'], "attention_mask": enc['attention_mask']}).logits.numpy()
        return int(logits.argmax())
    vec = tfidf.transform([text])
    return int(best_lr.predict(vec)[0])
