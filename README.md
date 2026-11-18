# SentimentX - The Sentiment Analysis using Machine Learning

Upload a CSV of product reviews. SentimentX labels each review as positive, neutral or negative, shows where it sits on a sentiment scale, finds the customer issues behind negative reviews, and lets you download the analysis as Excel (.xlsx), CSV or JSON.

Features:
- Overview with sentiment split, a spectrum of every review, and the top customer issue
- Customer issues: ranked complaint themes (delivery, build quality, defects, price, and more) with example reviews
- Reviews: filter by sentiment, search, and see confidence for each prediction
- Model quality: accuracy, per-class precision and recall, and a confusion matrix
- Export: Excel workbook (Summary, Reviews, Top issues, Top words, Model quality), CSV, JSON

## Run it

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000. The model is already trained (`artifacts/`). If it's missing, the app trains it on first start.

## Training data

`data/reviews_clean.csv` (196,299 real product reviews) is built from the original `Dataset.csv` by `prepare_dataset.py`:

- The short headline and the full review text are combined into one review.
- Ratings map to labels: 1–2 stars negative, 3 neutral, 4–5 positive.
- Malformed rows, empty reviews and exact duplicates are removed.

Class counts: 142,998 positive, 36,041 negative, 17,260 neutral.

Accuracy on the 15% held-out split: 98.72% (macro F1 0.9741). The model is a Linear SVM (C=0.1) with balanced class weights, wrapped in sigmoid calibration so each prediction has a confidence value.

## Retrain

```bash
python prepare_dataset.py        # only if you replace Dataset.csv
python train.py                  # uses data/reviews_clean.csv
python train.py --data your.csv  # any CSV with review and sentiment (or label) columns
```

Training uses 85% of the data and keeps 15% for testing. The metrics are written to `artifacts/metrics.json`.

## How it works

1. Each review becomes word features (1 and 2 word phrases) and character features (2 to 5 letter pieces, which handle misspellings).
2. A calibrated Linear SVM assigns probabilities to the three classes. Confidence is the highest probability, and the score is P(positive) minus P(negative).
4. Customer issues: keyword rules group the complaints in negative reviews into themes. A review can match more than one theme. Edit `issues.py` to change the themes.
3. Metrics are computed on the 15% test split, which the model never trained on.

## Requirements

`pip install -r requirements.txt` installs FastAPI, scikit-learn, pandas, and openpyxl (for Excel export).
