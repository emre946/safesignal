"""
03 - Precursor analysis on MSHA mine accident reports (60,000 recent records, 2016-2026).

Idea (teacher feedback: "focus on near-misses - which small events are the harbingers of a big accident?"):
  1. Build a severity pyramid per accident mechanism: no-injury events (near-misses) -> minor -> lost time -> serious.
  2. Train a text model that learns what SERIOUS events (fatal / permanent disability) look like,
     with outcome words (died, amputated ...) removed so it learns the mechanism, not the result.
  3. Score every NO-INJURY event: a near-miss that "reads like" a fatal accident is a precursor to act on.
Run: python src/03_msha_precursors.py
"""
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

from common import DATA, RES, BLUE, ORANGE, GREY, INK2, save

TIER = {
    "ACCIDENT ONLY": "0 Near-miss (no injury)",
    "NO DYS AWY FRM WRK,NO RSTR ACT": "1 Minor", "ALL OTHER CASES (INCL 1ST AID)": "1 Minor",
    "DAYS AWAY FROM WORK ONLY": "2 Lost time", "DAYS RESTRICTED ACTIVITY ONLY": "2 Lost time",
    "DYS AWY FRM WRK & RESTRCTD ACT": "2 Lost time",
    "PERM TOT OR PERM PRTL DISABLTY": "3 Serious", "FATALITY": "3 Serious",
}
# words that describe the OUTCOME rather than the mechanism -> removed to avoid label leakage
OUTCOME_WORDS = {"died", "dies", "death", "dead", "deceased", "fatal", "fatally", "fatality", "killed", "pronounced",
                 "coroner", "autopsy", "amputated", "amputation", "amputating", "permanent", "permanently",
                 "disability", "disabled", "surgery", "hospital", "hospitalized", "transported", "ambulance",
                 "airlifted", "lifeflight", "treated", "injured", "injury", "injuries", "ee", "employee", "victim",
                 "severed", "severing", "sever", "passed", "away", "unresponsive", "investigation", "partial", "tip",
                 "fingertip", "resulting", "removed", "appears", "cpr", "medical", "emt", "stitches", "fracture", "fractured",
                 "hernia", "loss", "ongoing", "unknown", "lost"}

m = pd.read_csv(DATA / "msha_accidents_2016_2026.csv.gz", low_memory=False)
m["Tier"] = m["DEGREE_INJURY"].map(TIER)
m = m.dropna(subset=["Tier", "NARRATIVE"]).copy()
m["Mechanism"] = m["CLASSIFICATION"].str.title().str.replace("Or", "or").str.strip()
stats = {"rows_used": len(m), "tier_counts": m.Tier.value_counts().sort_index().to_dict()}

# ---------------- 1. pyramid per mechanism
pv = pd.crosstab(m.Mechanism, m.Tier)
pv["Total"] = pv.sum(axis=1)
pv = pv[pv.Total >= 300]
pv["Serious per 1,000 events"] = (pv["3 Serious"] / pv.Total * 1000).round(1)
pv["Near-misses per serious event"] = (pv["0 Near-miss (no injury)"] / pv["3 Serious"].replace(0, np.nan)).round(1)
pv = pv.sort_values("Serious per 1,000 events", ascending=False)
pv.to_csv(RES / "msha_precursor_pyramid.csv")
stats["pyramid"] = pv.astype(object).where(pv.notna(), None).to_dict(orient="index")

top = pv.head(10)[::-1]
fig, ax = plt.subplots(figsize=(7.4, 3.9))
cols = [ORANGE if v >= 25 else BLUE for v in top["Serious per 1,000 events"]]
ax.barh(top.index, top["Serious per 1,000 events"], color=cols, height=0.62)
for i, (s, nm) in enumerate(zip(top["Serious per 1,000 events"], top["0 Near-miss (no injury)"])):
    ax.text(s + 0.8, i, f"{s:.0f}   |  {nm:,} near-misses reported", va="center", fontsize=8, color=INK2)
ax.set_xlabel("Serious outcomes (fatal / permanent disability) per 1,000 events")
ax.set_xlim(0, top["Serious per 1,000 events"].max() * 1.45); ax.grid(axis="y", visible=False)
ax.set_title("Where outcomes are worst, near-misses are barely reported")
save(fig, "msha_precursor_pyramid.png")

# ---------------- 2. serious-outcome model on injury events (near-misses held out)
inj = m[m.Tier != "0 Near-miss (no injury)"]
y = (inj.Tier == "3 Serious").astype(int).values
stop = list(ENGLISH_STOP_WORDS | OUTCOME_WORDS)


def clean(t):
    return re.sub(r"\d+", " ", str(t).lower())


vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True, stop_words=stop, preprocessor=clean)
X = vec.fit_transform(inj.NARRATIVE)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0).fit(Xtr, ytr)
p = clf.predict_proba(Xte)[:, 1]
thr = 0.5
stats["serious_model"] = {
    "train": int(Xtr.shape[0]), "test": int(Xte.shape[0]), "serious_rate_test": round(yte.mean(), 4),
    "roc_auc": round(roc_auc_score(yte, p), 3), "pr_auc": round(average_precision_score(yte, p), 3),
    "pr_auc_random_baseline": round(yte.mean(), 3),
    "recall_at_0.5": round(recall_score(yte, p >= thr), 3), "precision_at_0.5": round(precision_score(yte, p >= thr), 3),
}
print(stats["serious_model"])
fn = np.array(vec.get_feature_names_out()); co = clf.coef_[0]
stats["top_serious_terms"] = fn[np.argsort(co)[::-1][:25]].tolist()
stats["top_minor_terms"] = fn[np.argsort(co)[:15]].tolist()

# refit on all injury events, then score the held-out near-misses
clf_all = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0).fit(X, y)
nm = m[m.Tier == "0 Near-miss (no injury)"].copy()
nm["p_serious_like"] = clf_all.predict_proba(vec.transform(nm.NARRATIVE))[:, 1]
nm["words"] = nm.NARRATIVE.str.split().str.len()
nm.sort_values("p_serious_like", ascending=False)[["DOCUMENT_NO", "ACCIDENT_DT", "Mechanism", "p_serious_like", "NARRATIVE"]] \
    .to_csv(RES / "msha_near_miss_scores.csv", index=False)
stats["near_miss_scored"] = len(nm)
FLAG = 0.9
stats["flag_threshold"] = FLAG
stats["near_miss_flagged"] = int((nm.p_serious_like >= FLAG).sum())
stats["near_miss_flagged_share"] = round((nm.p_serious_like >= FLAG).mean() * 100, 1)
stats["flagged_by_mechanism"] = nm[nm.p_serious_like >= FLAG].Mechanism.value_counts().head(8).to_dict()
# diverse examples: best two short reports per mechanism
ex = (nm[nm.words.between(12, 45)].drop_duplicates("NARRATIVE").sort_values("p_serious_like", ascending=False)
      .groupby("Mechanism").head(2).head(10))
stats["example_flagged_near_misses"] = [{"p": round(r.p_serious_like, 2), "mechanism": r.Mechanism, "text": r.NARRATIVE}
                                        for r in ex.itertuples()]

# distribution chart
fig, ax = plt.subplots(figsize=(6.6, 2.9))
bins = np.linspace(0, 1, 21)
ax.hist(nm.p_serious_like, bins=bins, color=BLUE, alpha=0.9, edgecolor="white")
ax.axvline(FLAG, color=ORANGE, lw=2)
ax.text(0.02, 0.95, f"{stats['near_miss_flagged']} near-misses score \u2265 {FLAG}:\nthey read like serious accidents", transform=ax.transAxes,
        ha="left", va="top", color=ORANGE, fontsize=8.5, fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.9))
ax.set_xlabel("Model score: how much a no-injury event resembles a fatal / disabling one")
ax.set_ylabel("Near-miss reports"); ax.set_title(f"Scoring {len(nm):,} no-injury events for hidden danger")
save(fig, "msha_near_miss_scores.png")

json.dump(stats, open(RES / "msha_stats.json", "w"), indent=2, default=str)
print(pv[["0 Near-miss (no injury)", "3 Serious", "Total", "Serious per 1,000 events"]].head(12))
print(stats["near_miss_flagged"], stats["flagged_by_mechanism"])
for e in stats["example_flagged_near_misses"]:
    print(e)
print(stats["top_serious_terms"])
