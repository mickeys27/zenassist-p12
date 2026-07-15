"""
Entraîne le pipeline TF-IDF + Naive Bayes multinomial (meilleur modèle
identifié dans ZenAssist_V5.ipynb) et exporte :
  - dist/model.pkl     : pipeline scikit-learn complet (vectorizer + modèle)
  - dist/metrics.json  : métriques d'évaluation sur le jeu de test
"""

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_PATH = "datasetsample.csv"
OUTPUT_DIR = Path("dist")

# --- 1.1 / 1.2 Chargement et nettoyage -------------------------------------
df = pd.read_csv(DATA_PATH, dtype={"Consumer Claim": "str"}, low_memory=False)
df_clean = df.dropna(subset=["Consumer Claim"]).copy()

# --- 1.2 Unification de la taxonomie CFPB -----------------------------------
TAG_MAPPING = {
    "Bank account or service": "Checking or savings account",
    "Credit card": "Credit card or prepaid card",
    "Prepaid card": "Credit card or prepaid card",
    "Credit reporting": "Credit reporting, credit repair services, or other personal consumer reports",
    "Money transfers": "Money transfer, virtual currency, or money service",
    "Virtual currency": "Money transfer, virtual currency, or money service",
    "Payday loan": "Payday loan, title loan, or personal loan",
}
df_clean["Tag"] = df_clean["Tag"].replace(TAG_MAPPING)

# --- 1.4 Split train/test ----------------------------------------------------
X = df_clean["Consumer Claim"]
y = df_clean["Tag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3.1 / 3.2 Pipeline TF-IDF + Naive Bayes ---------------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
    )),
    ("clf", MultinomialNB()),
])

print("Entrainement du pipeline (TF-IDF + Naive Bayes)...")
t0 = time.perf_counter()
pipeline.fit(X_train, y_train)
fit_time_s = time.perf_counter() - t0
print(f"Entrainement termine en {fit_time_s:.1f}s")

# --- Evaluation ---------------------------------------------------------------
t0 = time.perf_counter()
y_pred = pipeline.predict(X_test)
pred_time_ms = (time.perf_counter() - t0) / len(X_test) * 1000

accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

print(f"Accuracy : {accuracy:.4f}")
print(f"F1 macro : {f1_macro:.4f}")
print(f"Prediction : {pred_time_ms:.4f} ms/reclamation")

# --- Export --------------------------------------------------------------------
OUTPUT_DIR.mkdir(exist_ok=True)

import pickle
with open(OUTPUT_DIR / "model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

metrics = {
    "model": "TF-IDF + Multinomial Naive Bayes",
    "accuracy": accuracy,
    "f1_macro": f1_macro,
    "fit_time_s": fit_time_s,
    "pred_time_ms_per_claim": pred_time_ms,
    "n_train": len(X_train),
    "n_test": len(X_test),
    "n_classes": int(y.nunique()),
    "classification_report": report,
}
with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Fichiers exportes dans dist/ : model.pkl, metrics.json")