"""
SafeSignal - first prototype of the triage step.
Trains the TF-IDF + Logistic Regression baseline on all reports and scores a new free-text report.

Usage:
  python src/07_ihm_triage_demo.py "While lifting the steel bar with the hoist, the chain broke and the load fell next to the operator."
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import DATA
df = pd.read_csv(DATA / "ihm_clean.csv")
y = (df["Potential_Level"].map({"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}) >= 4).astype(int)

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
X = vec.fit_transform(df["Description"])
clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=3).fit(X, y)

THRESHOLD = 0.40  # deliberately below 0.5: missing a dangerous near-miss costs more than a false alarm


def triage(text: str) -> dict:
    x = vec.transform([text])
    p = float(clf.predict_proba(x)[0, 1])
    contrib = x.toarray()[0] * clf.coef_[0]
    idx = np.argsort(contrib)[::-1][:5]
    terms = [vec.get_feature_names_out()[i] for i in idx if contrib[i] > 0]
    return {"p_high_potential": round(p, 2),
            "flag": "HIGH POTENTIAL - review within 24h" if p >= THRESHOLD else "Standard review",
            "evidence_terms": terms}


if __name__ == "__main__":
    examples = sys.argv[1:] or [
        "While lifting the steel bar with the hoist, the chain broke and the load fell 30 cm from the operator.",
        "The assistant was cleaning the office desk and cut his finger slightly with a piece of paper.",
    ]
    for t in examples:
        print("\nREPORT:", t)
        for k, v in triage(t).items():
            print(f"  {k}: {v}")
