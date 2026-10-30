"""
06 - Türkiye context: why this matters here (SGK + İSİG Meclisi), with GB (HSE) and US (BLS) for comparison.
Numbers are hand-collected from the cited sources into data/turkey_international_context.csv.
Run: python src/06_turkey_context.py
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
from common import DATA, RES, BLUE, ORANGE, INK2, save

c = pd.read_csv(DATA / "turkey_international_context.csv")
isig = c[(c.source.str.startswith("İSİG")) & c.category.str.startswith("cause:")].copy()
isig["label"] = isig.category.str.replace("cause: ", "")
isig = isig[isig.label != "Suicide"].sort_values("value")
fig, ax = plt.subplots(figsize=(6.4, 3.0))
cols = [ORANGE if l in ("Crushing / collapse", "Fall from height", "Electrocution") else BLUE for l in isig.label]
ax.barh(isig.label, isig.value, color=cols, height=0.62)
for i, v in enumerate(isig.value):
    ax.text(v + 6, i, f"{v:.0f}", va="center", fontsize=8, color=INK2)
ax.set_xlabel("Worker deaths, 2025 (İSİG Meclisi, at least)"); ax.grid(axis="y", visible=False)
ax.set_title("Türkiye 2025: what kills workers")
save(fig, "turkey_isig_2025_causes.png")
json.dump(c.to_dict(orient="records"), open(RES / "turkey_context.json", "w"), indent=2, ensure_ascii=False)
print(isig[["label", "value"]])
