"""
02 - OSHA Severe Injury Reports (Jan 2015 - Feb 2017, 21,578 narratives)
EDA by sector and mechanism, environmental-heat seasonality, and a mechanism classifier from free text.
Run: python src/02_osha_eda.py
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from common import DATA, RES, NAICS2, BLUE, ORANGE, AQUA, GREY, INK2, SEQ, save

EVENT_GROUP = {  # first digit of the OIICS event code
    "1": "Violence / animals", "2": "Transportation", "3": "Fire / explosion",
    "4": "Falls, slips, trips", "5": "Harmful exposure (heat, electricity, chemicals)",
    "6": "Contact with objects / equipment", "7": "Overexertion",
}

df = pd.read_csv(DATA / "osha_severe_injury_2015_2017.csv.gz", low_memory=False)
stats = {"rows_raw": len(df)}
df["EventDate"] = pd.to_datetime(df["EventDate"], errors="coerce")
df = df.dropna(subset=["EventDate", "Final Narrative"]).drop_duplicates(subset=["ID"])
df["naics2"] = df["Primary NAICS"].astype(str).str[:2]
df["Sector"] = df["naics2"].map(NAICS2).fillna("Other / unknown")
df["EventGroup"] = df["Event"].astype(str).str[0].map(EVENT_GROUP).fillna("Other")
df["Heat"] = df["EventTitle"].str.contains("environmental heat", case=False, na=False)
df["Amputation"] = df["Amputation"].fillna(0).clip(upper=1)
df["Month"] = df["EventDate"].dt.month
df["Words"] = df["Final Narrative"].str.split().str.len()
stats.update({
    "rows_clean": len(df), "date_min": str(df.EventDate.min().date()), "date_max": str(df.EventDate.max().date()),
    "states": int(df.State.nunique()), "amputation_share": round(df.Amputation.mean() * 100, 1),
    "hospitalized_share": round((df.Hospitalized >= 1).mean() * 100, 1),
    "words_mean": round(df.Words.mean(), 1),
    "sector_counts": df.Sector.value_counts().head(10).to_dict(),
    "event_group_counts": df.EventGroup.value_counts().to_dict(),
    "heat_cases": int(df.Heat.sum()),
})
df.drop(columns=["Final Narrative"]).to_csv(RES / "osha_clean_no_text.csv.gz", index=False, compression="gzip")

# ---- F1: sectors x mechanism (stacked share)
top = df.Sector.value_counts().head(8).index
ct = pd.crosstab(df[df.Sector.isin(top)].Sector, df.EventGroup)
order = ["Contact with objects / equipment", "Falls, slips, trips", "Harmful exposure (heat, electricity, chemicals)",
         "Transportation", "Overexertion"]
ct = ct.reindex(columns=order, fill_value=0)
ct["Other"] = pd.crosstab(df[df.Sector.isin(top)].Sector, df.EventGroup).drop(columns=order, errors="ignore").sum(axis=1)
ct = ct.loc[df.Sector.value_counts().head(8).index[::-1]]
share = ct.div(ct.sum(axis=1), axis=0) * 100
cols = [BLUE, ORANGE, AQUA, "#eda100", "#6250d6", GREY]
fig, ax = plt.subplots(figsize=(7.6, 3.8))
left = np.zeros(len(share))
for c, col in zip(share.columns, cols):
    ax.barh(share.index, share[c], left=left, color=col, height=0.62, label=c, edgecolor="white", linewidth=1.5)
    left += share[c].values
for i, n in enumerate(ct.sum(axis=1)):
    ax.text(101, i, f"n={n:,}", va="center", fontsize=8, color=INK2)
ax.set_xlim(0, 112); ax.set_xlabel("% of severe injury reports"); ax.grid(axis="y", visible=False)
ax.set_title("What hurts workers differs by sector (OSHA, 21.5k reports)")
ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.16))
save(fig, "osha_sector_mechanism.png")
stats["sector_mechanism_share"] = share.round(1).to_dict(orient="index")

# ---- F2: heat seasonality by sector group
g = np.where(df.Sector.isin(["Construction", "Agriculture"]), "Construction + agriculture", "All other sectors")
heat = df[df.Heat].assign(G=g[df.Heat.values])
m = pd.crosstab(heat.Month, heat.G).reindex(range(1, 13), fill_value=0)
# normalise to per-1,000 reports of that group in the same month
base = pd.crosstab(df.Month, g).reindex(range(1, 13), fill_value=0)
rate = (m / base * 1000).round(1)
fig, ax = plt.subplots(figsize=(6.8, 3.0))
months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
ax.plot(range(1, 13), rate["Construction + agriculture"], color=ORANGE, lw=2, marker="o", ms=4, label="Construction + agriculture")
ax.plot(range(1, 13), rate["All other sectors"], color=BLUE, lw=2, marker="o", ms=4, label="All other sectors")
ax.set_xticks(range(1, 13), months); ax.set_ylabel("Heat cases per 1,000 reports")
ax.set_title(f"{heat.Month.isin([6,7,8]).mean()*100:.0f}% of environmental-heat injuries happen in June-August")
ax.legend(frameon=False, loc="upper left")
save(fig, "osha_heat_seasonality.png")
stats["heat_rate_per_1000_by_month"] = rate.to_dict()
stats["heat_share_summer_JJA"] = round(heat.Month.isin([6, 7, 8]).mean() * 100, 1)
stats["heat_share_constr_agri"] = round((heat.G == "Construction + agriculture").mean() * 100, 1)
stats["constr_agri_share_of_all_reports"] = round((g == "Construction + agriculture").mean() * 100, 1)

# ---- Mechanism classifier from free text (auto-tagging new reports)
X, y = df["Final Narrative"], df["EventGroup"]
keep = y.map(y.value_counts()) >= 100
X, y = X[keep], y[keep]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, stop_words="english"),
                    LogisticRegression(max_iter=3000, C=4, class_weight="balanced"))
clf.fit(Xtr, ytr); pred = clf.predict(Xte)
rep = classification_report(yte, pred, output_dict=True, zero_division=0)
stats["mechanism_clf"] = {"test_size": int(len(yte)), "accuracy": round(accuracy_score(yte, pred), 3),
                          "macro_f1": round(f1_score(yte, pred, average="macro"), 3),
                          "per_class_f1": {k: round(v["f1-score"], 3) for k, v in rep.items() if k in set(y)}}
print(json.dumps(stats["mechanism_clf"], indent=1))
json.dump(stats, open(RES / "osha_stats.json", "w"), indent=2, default=str)
print("heat cases", stats["heat_cases"], "summer share", stats["heat_share_summer_JJA"], "constr/agri share", stats["heat_share_constr_agri"])
