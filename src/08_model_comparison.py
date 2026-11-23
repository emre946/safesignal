"""
08 - Assignment 2 model comparison.

Two tasks, same held-out test sets for every approach (stratified 80/20, seed 42):
  A. MECHANISM  - OSHA narrative -> 8 OIICS event groups (falls, caught-in, struck-by, heat ...)
  B. PRECURSOR  - MSHA narrative -> serious outcome (fatal / permanent disability) vs. not,
                  outcome words removed so the model learns the mechanism, not the result.

Approaches tried
  1. Majority / base-rate baseline
  2. TF-IDF word 1-2 grams + logistic regression      (Assignment 1 baseline)
  3. TF-IDF character 3-5 grams + linear model          (robust to typos / spelling variants)
  4. Pre-trained word embeddings (spaCy en_core_web_md, 300-d GloVe-style vectors, averaged) + logistic regression
  5. TF-IDF + small neural network (MLP, 1 hidden layer of 128 units, early stopping)

Run: python src/08_model_comparison.py      (~5-8 minutes)
Needs: pip install spacy && python -m spacy download en_core_web_md
"""
import json
import re
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from common import DATA, RES, BLUE, ORANGE, GREY, INK2, save
from safesignal_core import EVENT_GROUP, TIER, OUTCOME_WORDS, clean_text

SEED = 42


def spacy_vectors(texts):
    import spacy
    nlp = spacy.load("en_core_web_md", disable=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler", "tok2vec", "senter"])
    out = np.zeros((len(texts), 300), dtype=np.float32)
    for i, doc in enumerate(nlp.pipe((str(t).lower() for t in texts), batch_size=512)):
        toks = [t for t in doc if t.has_vector and not t.is_stop and t.is_alpha]
        if toks:
            out[i] = np.mean([t.vector for t in toks], axis=0)
    return out


def timed_fit_predict(name, make, Xtr, ytr, Xte, proba=False):
    t0 = time.time(); m = make().fit(Xtr, ytr); t1 = time.time()
    p = m.predict_proba(Xte) if proba else m.predict(Xte); t2 = time.time()
    return m, p, round(t1 - t0, 1), round((t2 - t1) / Xte.shape[0] * 1000, 3)


# ================= Task A: mechanism (OSHA) =================
o = pd.read_csv(DATA / "osha_severe_injury_2015_2017.csv.gz", low_memory=False)
o = o.dropna(subset=["Final Narrative"]).drop_duplicates(subset=["ID"])
o["y"] = o["Event"].astype(str).str[0].map(EVENT_GROUP).fillna("Other")
o = o[o.y.map(o.y.value_counts()) >= 100]
tr, te = train_test_split(o.index, test_size=0.2, stratify=o.y, random_state=SEED)
ytr, yte = o.loc[tr, "y"].values, o.loc[te, "y"].values
txt_tr, txt_te = o.loc[tr, "Final Narrative"], o.loc[te, "Final Narrative"]

rowsA = []
word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, stop_words="english")
Wtr, Wte = word.fit_transform(txt_tr), word.transform(txt_te)
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, sublinear_tf=True, max_features=150000)
Ctr, Cte = char.fit_transform(txt_tr), char.transform(txt_te)
print("embedding OSHA narratives with spaCy vectors ...")
Etr, Ete = spacy_vectors(txt_tr), spacy_vectors(txt_te)
word_small = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, stop_words="english", max_features=20000)
Str, Ste = word_small.fit_transform(txt_tr), word_small.transform(txt_te)

candA = [
    ("1 Majority baseline", lambda: DummyClassifier(strategy="most_frequent"), Wtr, Wte),
    ("2 TF-IDF words + LogReg", lambda: LogisticRegression(max_iter=3000, C=4, class_weight="balanced"), Wtr, Wte),
    ("3 TF-IDF characters + linear SVM", lambda: LinearSVC(C=0.5, class_weight="balanced"), Ctr, Cte),
    ("4 Word embeddings (spaCy) + LogReg", lambda: LogisticRegression(max_iter=4000, C=2, class_weight="balanced"), Etr, Ete),
    ("5 TF-IDF + neural net (MLP)", lambda: MLPClassifier(hidden_layer_sizes=(128,), early_stopping=True, max_iter=40, random_state=SEED), Str, Ste),
]
for name, mk, A, B in candA:
    print("A:", name)
    _, pred, fit_s, ms = timed_fit_predict(name, mk, A, ytr, B)
    rowsA.append({"Task": "A Mechanism (OSHA, 8 classes)", "Approach": name, "Accuracy": round(accuracy_score(yte, pred), 3),
                  "Macro-F1": round(f1_score(yte, pred, average="macro"), 3), "Train s": fit_s, "ms / report": ms})

# ================= Task B: serious-outcome precursor (MSHA) =================
m = pd.read_csv(DATA / "msha_accidents_2016_2026.csv.gz", low_memory=False)
m["Tier"] = m["DEGREE_INJURY"].map(TIER)
m = m.dropna(subset=["Tier", "NARRATIVE"])
m = m[m.Tier != "0 Near-miss (no injury)"]
m["y"] = (m.Tier == "3 Serious").astype(int)
trb, teb = train_test_split(m.index, test_size=0.2, stratify=m.y, random_state=SEED)
ybtr, ybte = m.loc[trb, "y"].values, m.loc[teb, "y"].values
cl_tr = m.loc[trb, "NARRATIVE"].map(clean_text); cl_te = m.loc[teb, "NARRATIVE"].map(clean_text)
stop = list(ENGLISH_STOP_WORDS | OUTCOME_WORDS)
wB = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True, stop_words=stop)
WBtr, WBte = wB.fit_transform(cl_tr), wB.transform(cl_te)
cB = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, sublinear_tf=True, max_features=150000)
CBtr, CBte = cB.fit_transform(cl_tr), cB.transform(cl_te)
print("embedding MSHA narratives with spaCy vectors ...")
EBtr, EBte = spacy_vectors(cl_tr), spacy_vectors(cl_te)
sB = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True, stop_words=stop, max_features=20000)
SBtr, SBte = sB.fit_transform(cl_tr), sB.transform(cl_te)


def topk(y, p, share=0.10):
    k = int(len(p) * share); idx = np.argsort(p)[::-1][:k]
    return round(y[idx].sum() / y.sum(), 3), round(y[idx].mean(), 3)


candB = [
    ("1 Base rate (random)", lambda: DummyClassifier(strategy="prior"), WBtr, WBte),
    ("2 TF-IDF words + LogReg", lambda: LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"), WBtr, WBte),
    ("3 TF-IDF characters + linear SVM", lambda: CalibratedClassifierCV(LinearSVC(C=0.3, class_weight="balanced"), cv=3), CBtr, CBte),
    ("4 Word embeddings (spaCy) + LogReg", lambda: LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced"), EBtr, EBte),
    ("5 TF-IDF + neural net (MLP)", lambda: MLPClassifier(hidden_layer_sizes=(128,), early_stopping=True, max_iter=40, random_state=SEED), SBtr, SBte),
]
rowsB = []
for name, mk, A, B in candB:
    print("B:", name)
    _, pr, fit_s, ms = timed_fit_predict(name, mk, A, ybtr, B, proba=True)
    p = pr[:, 1]
    rec10, prec10 = topk(ybte, p)
    rowsB.append({"Task": "B Precursor (MSHA, serious vs not)", "Approach": name, "ROC-AUC": round(roc_auc_score(ybte, p), 3),
                  "PR-AUC": round(average_precision_score(ybte, p), 3), "Recall @ top 10%": rec10, "Precision @ top 10%": prec10,
                  "Train s": fit_s, "ms / report": ms})

A = pd.DataFrame(rowsA); B = pd.DataFrame(rowsB)
A.to_csv(RES / "model_comparison_mechanism.csv", index=False)
B.to_csv(RES / "model_comparison_precursor.csv", index=False)
json.dump({"mechanism": rowsA, "precursor": rowsB, "test_sizes": {"mechanism": int(len(yte)), "precursor": int(len(ybte)),
           "precursor_serious_rate": round(ybte.mean(), 4)}}, open(RES / "model_comparison.json", "w"), indent=2)
pd.set_option("display.width", 200)
print(A.to_string(index=False)); print(B.to_string(index=False))

# chart - orange = model selected for the app (see README for why)
SELECTED = {"A": "3 TF-IDF characters + linear SVM", "B": "2 TF-IDF words + LogReg"}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
lab = lambda s: s.split(" ", 1)[1]
for ax, df, col, ttl, key in [(axes[0], A, "Macro-F1", "A. Mechanism (OSHA): macro-F1", "A"),
                              (axes[1], B, "PR-AUC", "B. Precursor (MSHA): PR-AUC", "B")]:
    cols = [ORANGE if a == SELECTED[key] else (GREY if i == 0 else BLUE) for i, a in zip(df.index, df.Approach)]
    ax.barh([lab(s) for s in df.Approach][::-1], df[col][::-1], color=cols[::-1], height=0.6)
    for i, v in enumerate(df[col][::-1]):
        ax.text(v + max(df[col]) * 0.015, i, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, max(df[col]) * 1.25); ax.set_title(ttl); ax.grid(axis="y", visible=False)
fig.text(0.5, -0.02, "Orange = selected for the prototype. Same held-out test sets (20%) for every approach.", ha="center", fontsize=8.5, color=INK2)
fig.tight_layout()
save(fig, "model_comparison.png")
