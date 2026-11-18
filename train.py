"""Train the SentimentX model on the product review dataset and save it with its metrics.

Input : data/reviews_clean.csv (built from Dataset.csv by prepare_dataset.py)
Split : 85% training, 15% held out for testing (stratified, fixed seed)

Usage:
    python train.py
    python train.py --data my_reviews.csv      # any CSV with review + sentiment columns
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from model import ARTIFACT_DIR, LABELS, METRICS_FILE, MODEL_FILE, normalize_label

BASE = Path(__file__).parent
DEFAULT_DATA = BASE / "data" / "reviews_clean.csv"


def build_pipeline():
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=25, max_features=150_000)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
                                  min_df=3, max_features=150_000)),
    ])
    return Pipeline([
        ("features", features),
        ("clf", CalibratedClassifierCV(
            LinearSVC(C=0.1, class_weight="balanced", max_iter=5000), cv=3, method="sigmoid")),
    ])


def load(path):
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    text_col = "review" if "review" in raw.columns else "text"
    label_col = "sentiment" if "sentiment" in raw.columns else "label"
    df = pd.DataFrame({
        "text": raw[text_col].fillna("").astype(str).str.strip(),
        "label": raw[label_col].map(normalize_label),
    }).dropna()
    return df[df["text"] != ""].reset_index(drop=True)


def score(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "test_size": int(len(y_true)),
        "per_class": {lab: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                            "f1": round(float(f[i]), 4), "support": int(s[i])} for i, lab in enumerate(LABELS)},
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def run(argv=None):
    ap = argparse.ArgumentParser(description="Train SentimentX")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    args = ap.parse_args(argv)

    data = load(args.data)
    X_tr, X_te, y_tr, y_te = train_test_split(
        data["text"], data["label"], test_size=0.15, random_state=42, stratify=data["label"]
    )
    pipe = build_pipeline().fit(X_tr, y_tr)
    result = score(y_te, pipe.predict(X_te))

    metrics = {
        "dataset": Path(args.data).name,
        "model": "TF-IDF (words + characters) with calibrated Linear SVM (C=0.1)",
        "total_reviews": int(len(data)),
        "train_size": int(len(X_tr)),
        **result,
        "class_counts": {lab: int(n) for lab, n in data["label"].value_counts().items()},
    }

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": pipe, "metrics": metrics}, MODEL_FILE, compress=3)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    print(f"accuracy {metrics['accuracy']}, macro F1 {metrics['macro_f1']}, "
          f"trained on {metrics['train_size']}, tested on {metrics['test_size']}")
    return metrics


if __name__ == "__main__":
    run()
