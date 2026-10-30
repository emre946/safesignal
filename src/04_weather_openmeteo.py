"""
04 - Do severe injuries rise on hot days? OSHA reports x Open-Meteo historical weather.

Teacher feedback: "Open-Meteo'dan kaza gününün sıcaklığını alıp aşırı sıcakta inşaat ve tarım kazalarının
artıp artmadığına bakabilirsiniz."

Method
  1. Snap every OSHA report to a 1°x1° grid cell (lat/lon rounded) and keep the cells with >= 30 reports
     (~170 cells, ~82% of all reports).
  2. Download the daily maximum temperature (ERA5 reanalysis) for each cell from the free Open-Meteo
     archive API for the whole study period (one request per cell, cached in data/weather_cache.csv).
  3. Build a cell x day panel and compute the injury RATE per 1,000 cell-days in each temperature band,
     separately for outdoor sectors (construction + agriculture) and all other sectors. Using rates
     (not raw counts) controls for how many days fall into each band.

Run (needs internet, ~3 minutes the first time, then cached):
    python src/04_weather_openmeteo.py
Requires 02_osha_eda.py to have been run first (it writes results/osha_clean_no_text.csv.gz).
"""
import json
import time
import urllib.parse
import urllib.request
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import DATA, RES, BLUE, ORANGE, INK2, save

API = "https://archive-api.open-meteo.com/v1/archive"
CACHE = DATA / "weather_cache.csv"
MIN_REPORTS_PER_CELL = 30
BANDS = [-50, 10, 20, 25, 30, 35, 60]
LABELS = ["<10°C", "10-20", "20-25", "25-30", "30-35", "≥35°C"]


def fetch_cell(lat, lon, start, end):
    q = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                                "daily": "temperature_2m_max", "timezone": "auto", "format": "csv"})
    with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:
        txt = r.read().decode("utf-8")
    # Open-Meteo CSV: 2 metadata lines, a blank line, then the daily table
    table = txt[txt.index("time,"):]
    w = pd.read_csv(StringIO(table))
    w.columns = ["date", "tmax"]
    w["lat"], w["lon"] = lat, lon
    return w


def load_weather(cells, start, end):
    cache = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(columns=["date", "tmax", "lat", "lon"])
    have = set(zip(cache.lat.round(1), cache.lon.round(1)))
    todo = [c for c in cells if (round(c[0], 1), round(c[1], 1)) not in have]
    print(f"{len(cells)} grid cells, {len(todo)} to download")
    failures_in_a_row = 0
    for i, (lat, lon) in enumerate(todo, 1):
        for attempt in range(3):
            try:
                cache = pd.concat([cache, fetch_cell(lat, lon, start, end)], ignore_index=True)
                failures_in_a_row = 0
                break
            except Exception as e:  # rate limit / network hiccup -> wait and retry
                print(f"  retry {lat},{lon}: {e}"); time.sleep(5 * (attempt + 1))
        else:
            failures_in_a_row += 1
            if failures_in_a_row >= 2:
                cache.to_csv(CACHE, index=False)
                raise SystemExit("Open-Meteo not reachable - check your internet connection and run again.")
        if i % 20 == 0:
            cache.to_csv(CACHE, index=False); print(f"  {i}/{len(todo)} cells")
        time.sleep(0.3)
    cache.to_csv(CACHE, index=False)
    return cache


def main():
    df = pd.read_csv(RES / "osha_clean_no_text.csv.gz", parse_dates=["EventDate"])
    df = df.dropna(subset=["Latitude", "Longitude"])
    df["lat"], df["lon"] = df.Latitude.round(0), df.Longitude.round(0)
    df["date"] = df.EventDate.dt.strftime("%Y-%m-%d")
    df["Outdoor"] = df.Sector.isin(["Construction", "Agriculture"])
    counts = df.groupby(["lat", "lon"]).size()
    cells = [tuple(c) for c in counts[counts >= MIN_REPORTS_PER_CELL].index]
    start, end = df.EventDate.min().strftime("%Y-%m-%d"), df.EventDate.max().strftime("%Y-%m-%d")

    w = load_weather(cells, start, end)
    w = w[w.set_index(["lat", "lon"]).index.isin(cells)].dropna(subset=["tmax"])
    w["band"] = pd.cut(w.tmax, BANDS, labels=LABELS, right=False)

    d = df[df.set_index(["lat", "lon"]).index.isin(cells)]
    per_day = d.groupby(["lat", "lon", "date", "Outdoor"]).size().unstack("Outdoor", fill_value=0)
    per_day.columns = ["other" if not c else "outdoor" for c in per_day.columns]
    heat = d[d.Heat].groupby(["lat", "lon", "date"]).size().rename("heat")
    panel = w.set_index(["lat", "lon", "date"]).join(per_day).join(heat).fillna(0)

    g = panel.groupby("band", observed=False)
    res = pd.DataFrame({
        "cell_days": g.size(),
        "outdoor_per_1000_days": g.outdoor.mean() * 1000,
        "other_per_1000_days": g.other.mean() * 1000,
        "heat_cases_per_1000_days": g.heat.mean() * 1000,
    }).round(2)
    ref = res.loc["20-25"]
    res["outdoor_vs_20_25"] = (res.outdoor_per_1000_days / ref.outdoor_per_1000_days).round(2)
    res["other_vs_20_25"] = (res.other_per_1000_days / ref.other_per_1000_days).round(2)
    res.to_csv(RES / "weather_heat_rates.csv")
    print(res)

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    x = np.arange(len(LABELS)); wd = 0.38
    ax.bar(x - wd / 2 - 0.01, res.outdoor_vs_20_25, wd, color=ORANGE, label="Construction + agriculture")
    ax.bar(x + wd / 2 + 0.01, res.other_vs_20_25, wd, color=BLUE, label="All other sectors")
    ax.axhline(1, color=INK2, lw=1, ls="--")
    for xi, v in zip(x - wd / 2, res.outdoor_vs_20_25):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, LABELS); ax.set_xlabel("Daily maximum temperature (Open-Meteo / ERA5)")
    ax.set_ylabel("Injury rate vs 20-25°C days")
    ax.set_title("Severe injuries per cell-day by temperature band")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "weather_heat_rates.png")

    out = {"cells": len(cells), "reports_covered": int(len(d)), "share_covered": round(len(d) / len(df) * 100, 1),
           "period": [start, end], "rates": res.to_dict(orient="index")}
    json.dump(out, open(RES / "weather_stats.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
