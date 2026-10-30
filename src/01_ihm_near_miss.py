"""
01 - Near-miss signal in the IHM Stefanini dataset (actual vs potential severity) + first triage baseline.
Run: python src/01_ihm_near_miss.py
"""
import json, re
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from common import DATA, FIG, RES, BLUE, ORANGE, SEQ, INK, INK2, save

LEVELS = ["I", "II", "III", "IV", "V", "VI"]

# ======================= 1. LOAD & CLEAN =======================
raw = pd.read_csv(DATA / "ihm_stefanini_accidents.csv")
stats = {"raw_rows": len(raw), "raw_cols": raw.shape[1]}

df = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")])
df = df.rename(columns={"Data": "Date", "Countries": "Country", "Genre": "Gender",
                        "Employee or Third Party": "Employee_Type",
                        "Industry Sector": "Sector", "Accident Level": "Accident_Level",
                        "Potential Accident Level": "Potential_Level",
                        "Critical Risk": "Critical_Risk"})
stats["missing_values_total"] = int(df.isna().sum().sum())
stats["duplicates_found"] = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
df["Date"] = pd.to_datetime(df["Date"])
df["Critical_Risk"] = df["Critical_Risk"].str.strip()
df["Description"] = df["Description"].str.strip()
df["Month"] = df["Date"].dt.to_period("M")
df["Weekday"] = df["Date"].dt.day_name()
df["Desc_Words"] = df["Description"].str.split().str.len()
roman = {k: i + 1 for i, k in enumerate(LEVELS)}
df["AL_num"] = df["Accident_Level"].map(roman)
df["PL_num"] = df["Potential_Level"].map(roman)
df["Severity_Gap"] = df["PL_num"] - df["AL_num"]
df["High_Potential"] = (df["PL_num"] >= 4).astype(int)  # IV-VI => high potential severity

stats.update({
    "clean_rows": len(df),
    "date_min": str(df["Date"].min().date()), "date_max": str(df["Date"].max().date()),
    "countries": int(df["Country"].nunique()), "plants": int(df["Local"].nunique()),
    "critical_risk_categories": int(df["Critical_Risk"].nunique()),
    "others_share_critical_risk": round((df["Critical_Risk"] == "Others").mean() * 100, 1),
    "accident_level_counts": df["Accident_Level"].value_counts().reindex(LEVELS, fill_value=0).to_dict(),
    "potential_level_counts": df["Potential_Level"].value_counts().reindex(LEVELS, fill_value=0).to_dict(),
    "sector_counts": df["Sector"].value_counts().to_dict(),
    "country_counts": df["Country"].value_counts().to_dict(),
    "gender_counts": df["Gender"].value_counts().to_dict(),
    "employee_type_counts": df["Employee_Type"].value_counts().to_dict(),
    "desc_words": df["Desc_Words"].describe().round(1).to_dict(),
    "share_level_I": round((df["Accident_Level"] == "I").mean() * 100, 1),
    "share_high_potential": round(df["High_Potential"].mean() * 100, 1),
    "share_levelI_but_high_potential": round(
        ((df["Accident_Level"] == "I") & (df["High_Potential"] == 1)).sum() / (df["Accident_Level"] == "I").sum() * 100, 1),
    "mean_severity_gap": round(df["Severity_Gap"].mean(), 2),
    "share_gap_ge2": round((df["Severity_Gap"] >= 2).mean() * 100, 1),
    "high_potential_by_sector_pct": (df.groupby("Sector")["High_Potential"].mean() * 100).round(1).to_dict(),
    "high_potential_by_emp_pct": (df.groupby("Employee_Type")["High_Potential"].mean() * 100).round(1).to_dict(),
    "desc_words_by_potential": df.groupby("High_Potential")["Desc_Words"].mean().round(1).to_dict(),
})
df.drop(columns=["Month"]).to_csv(DATA / "ihm_clean.csv", index=False)

# ======================= 2. FIGURES =======================

# F1: actual vs potential level side by side
fig, ax = plt.subplots(figsize=(6.4, 3.4))
x = np.arange(len(LEVELS)); w = 0.38
a = df["Accident_Level"].value_counts().reindex(LEVELS, fill_value=0)
p = df["Potential_Level"].value_counts().reindex(LEVELS, fill_value=0)
ax.bar(x - w/2 - 0.01, a, w, color=BLUE, label="Actual accident level")
ax.bar(x + w/2 + 0.01, p, w, color=ORANGE, label="Potential accident level")
for xi, v in zip(x - w/2, a): ax.text(xi, v + 4, str(v), ha="center", fontsize=8, color=INK2)
for xi, v in zip(x + w/2, p): ax.text(xi, v + 4, str(v), ha="center", fontsize=8, color=INK2)
ax.set_xticks(x, LEVELS); ax.set_xlabel("Severity level (I = least, VI = most severe)"); ax.set_ylabel("Reports")
ax.set_title("Most accidents end mild, but their potential is much higher")
ax.legend(frameon=False); ax.grid(axis="x", visible=False)
save(fig, "ihm_actual_vs_potential.png")

# F2: heatmap actual x potential
ct = pd.crosstab(df["Accident_Level"], df["Potential_Level"]).reindex(index=LEVELS[:5], columns=LEVELS, fill_value=0)
fig, ax = plt.subplots(figsize=(5.2, 3.6))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fd"] + SEQ)
im = ax.imshow(ct.values, cmap=cmap, aspect="auto")
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        v = ct.values[i, j]
        if v: ax.text(j, i, v, ha="center", va="center", fontsize=8, color="white" if v > 60 else INK)
ax.set_xticks(range(6), LEVELS); ax.set_yticks(range(5), LEVELS[:5])
ax.set_xlabel("Potential accident level"); ax.set_ylabel("Actual accident level")
ax.set_title("Near-miss signal: Level I accidents with Level IV potential")
ax.grid(False)
save(fig, "ihm_heatmap_actual_potential.png")

# F3: critical risk top 10 (excluding 'Others')
cr = df.loc[df["Critical_Risk"] != "Others", "Critical_Risk"].value_counts().head(10)[::-1]
fig, ax = plt.subplots(figsize=(6.0, 3.6))
ax.barh(cr.index, cr.values, color=BLUE, height=0.65)
for i, v in enumerate(cr.values): ax.text(v + 0.3, i, str(v), va="center", fontsize=8, color=INK2)
ax.set_xlabel("Reports"); ax.grid(axis="y", visible=False)
ax.set_title(f"Top 10 critical risks ('Others' = {stats['others_share_critical_risk']}% excluded)")
save(fig, "ihm_critical_risks.png")

# F4: monthly trend
m = df.groupby("Month").size(); mh = df[df["High_Potential"] == 1].groupby("Month").size().reindex(m.index, fill_value=0)
fig, ax = plt.subplots(figsize=(6.8, 3.0))
xs = m.index.to_timestamp()
ax.plot(xs, m.values, color=BLUE, lw=2, marker="o", ms=4, label="All reports")
ax.plot(xs, mh.values, color=ORANGE, lw=2, marker="o", ms=4, label="High potential (IV-VI)")
ax.set_ylabel("Reports per month"); ax.set_title("Monthly reports, Jan 2016 - Jul 2017 (Jul 2017 partial)")
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y")); ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
save(fig, "ihm_monthly_trend.png")

# F5: high-potential share by sector and employee type
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
for ax, col in zip(axes, ["Sector", "Employee_Type"]):
    s = (df.groupby(col)["High_Potential"].mean() * 100).sort_values()
    ax.bar(s.index, s.values, color=SEQ[3], width=0.6)
    for i, v in enumerate(s.values): ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8, color=INK2)
    ax.set_title(f"By {col.replace('_', ' ').lower()}"); ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=8)
axes[0].set_ylabel("% high potential (IV-VI)"); axes[0].set_ylim(0, 75)
fig.suptitle("Share of reports with high potential severity (IV-VI)", fontweight="bold", fontsize=11, y=1.04)
save(fig, "ihm_high_potential_share.png")

# F6: description length by class + top words
STOP = set("""the a an of to in and on at with by for from his he was is that this which it as be were her she
when who are while had has been its into their they them out up then there so not or but do did than also
being one two three four five six left right causing caused employee worker operator mr time moment""".split())
def tokens(t):
    return [w for w in re.findall(r"[a-z]+", t.lower()) if len(w) > 2 and w not in STOP]
hi = pd.Series([w for t in df.loc[df.High_Potential == 1, "Description"] for w in tokens(t)]).value_counts()
lo = pd.Series([w for t in df.loc[df.High_Potential == 0, "Description"] for w in tokens(t)]).value_counts()
rel = ((hi / hi.sum()) / ((lo.reindex(hi.index).fillna(0) + 1) / lo.sum()))[hi >= 12].sort_values(ascending=False).head(10)[::-1]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1, 1.3]})
bins = np.arange(0, 200, 10)
axes[0].hist(df.loc[df.High_Potential == 0, "Desc_Words"], bins=bins, color=BLUE, alpha=0.75, label="Low (I-III)")
axes[0].hist(df.loc[df.High_Potential == 1, "Desc_Words"], bins=bins, color=ORANGE, alpha=0.75, label="High (IV-VI)")
axes[0].set_xlabel("Words per description"); axes[0].set_ylabel("Reports"); axes[0].legend(frameon=False, fontsize=8)
axes[0].set_title("Report length")
axes[1].barh(rel.index, rel.values, color=ORANGE, height=0.6)
axes[1].set_xlabel("Over-representation in high-potential reports (x)")
axes[1].set_title("Words that signal high potential"); axes[1].grid(axis="y", visible=False)
fig.tight_layout()
save(fig, "ihm_text_signals.png")
stats["top_high_signal_words"] = [str(w) for w in rel.index[::-1]]

# ======================= 3. FIRST MODEL EXPERIMENTS =======================
X = df["Description"].values
y_bin = df["High_Potential"].values
y_multi = df["Potential_Level"].replace({"VI": "V"}).to_numpy(dtype=object).astype(str)  # VI has 1 sample -> merged into V

def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")

models = {
    "Majority baseline": lambda: DummyClassifier(strategy="most_frequent"),
    "TF-IDF + Complement NB": lambda: make_pipeline(tfidf(), ComplementNB()),
    "TF-IDF + Logistic Regression": lambda: make_pipeline(tfidf(), LogisticRegression(max_iter=2000, class_weight="balanced", C=3)),
    "TF-IDF + Linear SVM": lambda: make_pipeline(tfidf(), LinearSVC(class_weight="balanced", C=0.5)),
    "TF-IDF + Random Forest": lambda: make_pipeline(tfidf(), RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42)),
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
cm_best = None
for task, y in [("Binary: high vs low potential", y_bin), ("Multi-class: potential level I-V", y_multi)]:
    for name, mk in models.items():
        preds = np.empty(len(y), dtype=object)
        for tr, te in skf.split(X, y):
            m = mk().fit(X[tr], y[tr]); preds[te] = m.predict(X[te])
        preds = preds.astype(y.dtype)
        r = {"Task": task, "Model": name,
             "Accuracy": round(accuracy_score(y, preds), 3),
             "Macro-F1": round(f1_score(y, preds, average="macro"), 3)}
        if task.startswith("Binary"):
            r["Recall (high)"] = round(recall_score(y, preds, pos_label=1), 3)
            if name == "TF-IDF + Logistic Regression":
                cm_best = confusion_matrix(y, preds, labels=[0, 1]).tolist()
        rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(RES / "ihm_baseline_results.csv", index=False)
stats["binary_confusion_logreg"] = cm_best
print(res.to_string(index=False))

# F7: results chart (binary task)
b = res[res.Task.str.startswith("Binary")].set_index("Model")
fig, ax = plt.subplots(figsize=(6.6, 3.0))
yy = np.arange(len(b)); h = 0.38
ax.barh(yy + h/2 + 0.01, b["Macro-F1"], h, color=BLUE, label="Macro-F1")
ax.barh(yy - h/2 - 0.01, b["Recall (high)"], h, color=ORANGE, label="Recall on high-potential class")
for i, (f, rc) in enumerate(zip(b["Macro-F1"], b["Recall (high)"])):
    ax.text(f + 0.01, i + h/2, f"{f:.2f}", va="center", fontsize=8, color=INK2)
    ax.text(rc + 0.01, i - h/2, f"{rc:.2f}", va="center", fontsize=8, color=INK2)
ax.set_yticks(yy, b.index); ax.set_xlim(0, 1.05); ax.invert_yaxis()
ax.set_title("First experiments - 5-fold CV, binary severity triage"); ax.grid(axis="y", visible=False)
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
save(fig, "ihm_baseline_results.png")

json.dump(stats, open(RES / "ihm_stats.json", "w"), indent=2, default=str)
print(json.dumps(stats, indent=1, default=str))
