"""Train the SentimentX model on a labelled review CSV, then save the model and its metrics.

Usage:
    python train.py                      # uses data/reviews_large.csv
    python train.py --data my_reviews.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from model import ARTIFACT_DIR, LABELS, METRICS_FILE, MODEL_FILE, normalize_label

BASE = Path(__file__).parent
DEFAULT_DATA = BASE / "data" / "reviews_large.csv"


def build_pipeline():
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=25, max_features=150_000)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
                                  min_df=3, max_features=150_000)),
    ])
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(C=5.0, max_iter=1000, class_weight="balanced")),
    ])


def run(argv=None):
    ap = argparse.ArgumentParser(description="Train SentimentX")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    args = ap.parse_args(argv)

    raw = pd.read_csv(args.data)
    raw.columns = [c.strip().lower() for c in raw.columns]
    text_col = "review" if "review" in raw.columns else "text"
    label_col = "sentiment" if "sentiment" in raw.columns else "label"
    df = pd.DataFrame({
        "text": raw[text_col].fillna("").astype(str).str.strip(),
        "label": raw[label_col].map(normalize_label),
    }).dropna()
    df = df[df["text"] != ""]

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text"], df["label"], test_size=0.15, random_state=42, stratify=df["label"]
    )
    pipe = build_pipeline().fit(X_tr, y_tr)
    pred = pipe.predict(X_te)

    p, r, f, s = precision_recall_fscore_support(y_te, pred, labels=LABELS, zero_division=0)
    metrics = {
        "dataset": Path(args.data).name,
        "total_reviews": int(len(df)),
        "train_size": int(len(X_tr)),
        "test_size": int(len(X_te)),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
        "per_class": {
            lab: {
                "precision": round(float(p[i]), 4),
                "recall": round(float(r[i]), 4),
                "f1": round(float(f[i]), 4),
                "support": int(s[i]),
            }
            for i, lab in enumerate(LABELS)
        },
        "confusion": confusion_matrix(y_te, pred, labels=LABELS).tolist(),
    }

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": pipe, "metrics": metrics}, MODEL_FILE, compress=3)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    print(f"accuracy {metrics['accuracy']}, macro F1 {metrics['macro_f1']}, "
          f"trained on {metrics['train_size']}, tested on {metrics['test_size']}")
    return metrics


if __name__ == "__main__":
    run()
