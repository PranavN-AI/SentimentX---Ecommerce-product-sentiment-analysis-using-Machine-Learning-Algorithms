"""Clean the uploaded product-review dataset for SentimentX.

Input : Dataset.csv (Latin-1; columns Product_name, Price, Rate, Review, Summary)
Output: data/reviews_clean.csv with columns review, sentiment, stars

In this file the short headline sits in 'Review' and the full review text sits in
'Summary', so the two are combined. Ratings map to labels: 1-2 negative, 3 neutral,
4-5 positive. Malformed rows and duplicate reviews are removed.
"""
import csv
from collections import Counter
from pathlib import Path

import pandas as pd

SRC = Path("Dataset.csv")
OUT = Path("data") / "reviews_clean.csv"

df = pd.read_csv(SRC, encoding="latin-1", dtype=str, keep_default_na=False)
print("raw rows:", len(df))

stars = pd.to_numeric(df["Rate"].str.strip(), errors="coerce")
df = df[stars.between(1, 5)].copy()
df["stars"] = stars[df.index].astype(int)
print("after removing malformed ratings:", len(df))

parts = [df["Review"].str.strip(), df["Summary"].str.strip()]
text = parts[0].where(parts[1] == "", parts[0] + ". " + parts[1])
text = text.where(parts[0] != "", parts[1]).str.replace(r"\s+", " ", regex=True).str.strip()
df["review"] = text
df = df[df["review"] != ""]
print("after removing empty reviews:", len(df))

df = df.drop_duplicates(subset=["review", "stars"]).reset_index(drop=True)
print("after removing duplicates:", len(df))

df["sentiment"] = df["stars"].map(lambda s: "negative" if s <= 2 else ("neutral" if s == 3 else "positive"))
OUT.parent.mkdir(exist_ok=True)
df[["review", "sentiment", "stars"]].to_csv(OUT, index=False, quoting=csv.QUOTE_MINIMAL)
print("class counts:", Counter(df["sentiment"]))
print("median review length (chars):", int(df["review"].str.len().median()))
print("wrote", OUT, len(df))
