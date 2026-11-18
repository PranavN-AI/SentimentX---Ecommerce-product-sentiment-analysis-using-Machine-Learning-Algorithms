"""SentimentX serving layer: read a review CSV, score each review, find complaint themes, build statistics."""
import csv
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from issues import find_issues

BASE = Path(__file__).parent
ARTIFACT_DIR = BASE / "artifacts"
MODEL_FILE = ARTIFACT_DIR / "model.joblib"
METRICS_FILE = ARTIFACT_DIR / "metrics.json"
LABELS = ["negative", "neutral", "positive"]
MAX_ROWS = 5000
TEXT_NAMES = ["review", "text", "comment", "review_text", "content", "body"]
LABEL_NAMES = ["sentiment", "label", "class"]
RATING_NAMES = ["rating", "stars", "score"]

# Filler words that tell us nothing about sentiment; hidden from the "drivers" list.
STOPWORDS = {
    "the", "and", "was", "this", "that", "with", "for", "are", "its", "all", "very", "really",
    "quite", "just", "have", "had", "you", "they", "there", "more", "some", "also", "one", "it",
    "is", "at", "of", "to", "my", "a", "i", "in", "on", "be", "as", "so", "do", "does", "would", "will",
}

_cache = {}


def normalize_label(value):
    s = str(value).strip().lower()
    if s in ("positive", "pos", "1", "good"):
        return "positive"
    if s in ("negative", "neg", "0", "bad"):
        return "negative"
    if s in ("neutral", "neu", "2", "mixed"):
        return "neutral"
    return None


def stars_to_label(value):
    try:
        stars = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(stars):
        return None
    if stars <= 2:
        return "negative"
    if stars == 3:
        return "neutral"
    return "positive"


def read_reviews(fileobj):
    raw = fileobj.read()
    text = raw.decode("utf-8-sig", errors="replace") if isinstance(raw, bytes) else raw
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise ValueError("The file is empty.")
    width = len(rows[0])
    for line_no, row in enumerate(rows[1:], start=2):
        if row and len(row) != width:
            raise ValueError(
                f"Line {line_no} has {len(row)} fields but the header has {width}. "
                "Put review text that contains commas inside double quotes."
            )
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df.columns = [str(c).strip().lower() for c in df.columns]

    text_col = next((c for c in TEXT_NAMES if c in df.columns), None)
    if text_col is None:
        text_col = next((c for c in df.columns if df[c].dtype == object and c not in LABEL_NAMES), None)
    if text_col is None:
        raise ValueError("No text column found. Name the column 'review' or 'text'.")

    data = pd.DataFrame({"text": df[text_col].fillna("").astype(str).str.strip()})
    label_col = next((c for c in LABEL_NAMES if c in df.columns), None)
    rating_col = next((c for c in RATING_NAMES if c in df.columns), None)
    data["label"] = df[label_col].map(normalize_label) if label_col else None
    data["stars"] = pd.to_numeric(df[rating_col], errors="coerce") if rating_col else np.nan
    data["expected"] = data["stars"].map(stars_to_label) if rating_col else None

    data = data[data["text"] != ""].head(MAX_ROWS).reset_index(drop=True)
    if data.empty:
        raise ValueError("The file has no reviews with text in them.")
    return data


def ensure_model():
    """Load the trained model; train it first if the artifact is missing."""
    if not MODEL_FILE.exists():
        from train import run
        run([])
    if "bundle" not in _cache:
        _cache["bundle"] = joblib.load(MODEL_FILE)
    return _cache["bundle"]


def _top_words(model, features_names, classes):
    """Average the per-class weights of the calibrated SVM's cross-validation models."""
    clf = model.named_steps["clf"]
    coef = np.mean([c.estimator.coef_ for c in clf.calibrated_classifiers_], axis=0)
    word_idx = np.array([
        i for i, n in enumerate(features_names)
        if n.startswith("words__") and " " not in n and n[7:] not in STOPWORDS and len(n) - 7 >= 3
    ])
    top = {}
    for lab in LABELS:
        row = coef[list(classes).index(lab)][word_idx]
        best = word_idx[np.argsort(-row)[:10]]
        top[lab] = [str(features_names[j]).removeprefix("words__") for j in best]
    return top


def analyze_csv(fileobj):
    bundle = ensure_model()
    model, metrics = bundle["model"], bundle["metrics"]

    data = read_reviews(fileobj)
    classes = list(model.classes_)
    proba = model.predict_proba(data["text"])[:, [classes.index(l) for l in LABELS]]

    data["sentiment"] = [LABELS[i] for i in proba.argmax(axis=1)]
    data["confidence"] = proba.max(axis=1)
    data["score"] = proba[:, 2] - proba[:, 0]  # -1 very negative ... +1 very positive

    total = len(data)
    counts = data["sentiment"].value_counts().reindex(LABELS, fill_value=0)
    distribution = [
        {"label": lab, "count": int(counts[lab]), "percent": round(float(counts[lab]) / total * 100, 1)}
        for lab in LABELS
    ]

    features_names = model.named_steps["features"].get_feature_names_out()
    top_words = _top_words(model, features_names, classes)

    labelled = data[data["label"].notna()]
    user_accuracy = (
        round(float((labelled["sentiment"] == labelled["label"]).mean()) * 100, 1) if len(labelled) else None
    )
    checked = data[data["expected"].notna()]
    star_agreement = (
        round(float((checked["sentiment"] == checked["expected"]).mean()) * 100, 1) if len(checked) else None
    )

    issues = find_issues(data["text"].tolist(), data["sentiment"].tolist(), target="negative")

    reviews = [
        {
            "text": t,
            "sentiment": s,
            "confidence": round(float(c), 3),
            "score": round(float(sc), 3),
            "stars": None if pd.isna(st) else float(st),
            "expected": None if e is None or pd.isna(e) else e,
        }
        for t, s, c, sc, st, e in zip(
            data["text"], data["sentiment"], data["confidence"], data["score"], data["stars"], data["expected"]
        )
    ]

    return {
        "total": total,
        "distribution": distribution,
        "mean_score": round(float(data["score"].mean()), 3),
        "top_words": top_words,
        "user_accuracy": user_accuracy,
        "star_agreement": star_agreement,
        "issues": issues,
        "metrics": metrics,
        "reviews": reviews,
    }
