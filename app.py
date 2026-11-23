"""
SafeSignal - Streamlit prototype (Assignment 2)

    pip install -r requirements.txt
    streamlit run app.py

Paste an incident or near-miss report (English or Turkish) -> mechanism, precursor priority with evidence words,
similar real accident reports, the day's temperature from Open-Meteo, and a 5-minute toolbox talk to download.
The first start trains the models (~20-30 s) and caches them in models/.
"""
import datetime as dt
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import safesignal_core as core  # noqa: E402

_spec = importlib.util.spec_from_file_location("toolbox", ROOT / "src" / "05_toolbox_talk.py")
toolbox = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(toolbox)

st.set_page_config(page_title="SafeSignal", page_icon="🦺", layout="wide")

CITIES = {  # a few Turkish industrial cities; any lat/lon works
    "İstanbul": (41.01, 28.98), "Kocaeli": (40.77, 29.94), "Bursa": (40.19, 29.06), "Ankara": (39.93, 32.86),
    "İzmir": (38.42, 27.14), "Konya": (37.87, 32.48), "Gaziantep": (37.07, 37.38), "Adana": (37.00, 35.32),
    "Antalya": (36.90, 30.70), "Diyarbakır": (37.91, 40.23), "Şanlıurfa": (37.16, 38.79), "Zonguldak": (41.45, 31.79),
}
SECTORS = ["Construction", "Manufacturing", "Agriculture", "Mining & oil/gas", "Transport & warehousing",
           "Wholesale trade", "Retail trade", "Utilities", "Admin & waste services", "Health care"]
OUTDOOR = {"Construction", "Agriculture", "Mining & oil/gas"}
EXAMPLES = {
    "Real MSHA near-miss (EN)": "No injury - near miss accident - Employee was dumping load out of 785B truck - pulled out with the box up & made contact with overhead power lines.",
    "Conveyor jam (EN)": "Operator reached into the conveyor to clear a jam while it was still running; his glove was caught but he pulled his hand free. No injury.",
    "Forklift, Turkish (TR)": "Depoda forklift geri manevra yaparken arkadan geçen yayaya kıl payı çarpmadı. Geri vites ikazı çalışmıyordu. Kimse yaralanmadı.",
    "Scaffold, Turkish (TR)": "İskelede çalışan işçinin elinden çekiç düştü, aşağıda yürüyen operatörün yanına düştü. Kimse yaralanmadı.",
}


@st.cache_resource(show_spinner="Training / loading models (first start only, ~30 s)…")
def models():
    return core.load_or_train()


@st.cache_resource(show_spinner=False)
def osha_for_talks():
    return toolbox.load()


@st.cache_data(show_spinner=False, ttl=3600)
def temperature(lat, lon, day):
    try:
        return core.max_temp(lat, lon, day)
    except Exception:
        return None


# ------------------------------------------------------------------ layout
st.title("🦺 SafeSignal")
st.caption("Catching the near-misses that come before serious workplace accidents · SDG 8.8 · Group 2, AI for the SDGs")

with st.sidebar:
    st.header("Context")
    sector = st.selectbox("Sector", SECTORS, index=0)
    city = st.selectbox("Location", list(CITIES) + ["Other (lat/lon)"])
    if city == "Other (lat/lon)":
        lat = st.number_input("Latitude", value=41.0); lon = st.number_input("Longitude", value=29.0)
    else:
        lat, lon = CITIES[city]
    day = st.date_input("Date of the event / shift", dt.date.today())
    talk_lang = st.radio("Toolbox talk language", ["tr", "en"], format_func=lambda x: "Türkçe" if x == "tr" else "English", horizontal=True)
    st.divider()
    st.caption("Models: character TF-IDF + linear SVM (mechanism, macro-F1 0.80) and word TF-IDF + logistic regression "
               "(serious-outcome precursor, ROC-AUC 0.89). Trained on 21.5k OSHA and 52k MSHA reports. "
               "SafeSignal ranks reports - a safety officer decides.")

if "report" not in st.session_state:
    st.session_state.report = ""
st.write("**Try an example:**")
cols = st.columns(len(EXAMPLES))
for c, (k, v) in zip(cols, EXAMPLES.items()):
    if c.button(k, use_container_width=True):
        st.session_state.report = v
report = st.text_area("Incident / near-miss report (English or Turkish)", key="report", height=110,
                      placeholder="e.g. While lifting the steel beam with the crane, the sling slipped and the load swung past two workers…")
go = st.button("Analyse report", type="primary")

if go and report.strip():
    M = models()
    r = core.analyse(M, report)
    if "error" in r:
        st.error("No safety-related words were recognised. Try describing what happened in more detail.")
        st.stop()
    if r["language"] == "tr":
        st.info(f"Turkish report detected - experimental glossary translation used by the models: *{r['model_text']}*")
    if r["coverage"] < 4:
        st.warning("Very few words matched the model's vocabulary - treat the result as uncertain.")

    t = temperature(lat, lon, day.isoformat())
    band = core.heat_band(t) if t is not None else None
    ratios = core.HEAT_RATIO_OUTDOOR if sector in OUTDOOR else core.HEAT_RATIO_OTHER
    hot = t is not None and t >= 30 and sector in OUTDOOR

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("1 · Mechanism")
        mech, pm = r["mechanism"][0]
        st.markdown(f"#### {mech}")
        st.caption(f"{pm:.0%} confidence · from 8 OSHA mechanism groups")
        st.bar_chart(pd.DataFrame({"probability": [p for _, p in r["mechanism"]]}, index=[m for m, _ in r["mechanism"]]), horizontal=True, height=150)
    with c2:
        st.subheader("2 · Precursor priority")
        icon = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}[r["level"]]
        pct = min(r["percentile"], 99.0)
        st.markdown(f"#### {icon} {r['level']}")
        st.caption(f"Reads riskier than {pct:.0f}% of 52k past mine incident reports · HIGH = top 10%")
        st.write("**Why:** " + (", ".join(f"`{e}`" for e in r["evidence"]) if r["evidence"] else "no strong signal words"))
        st.caption("Ranks how much the report reads like accidents that ended in death or permanent disability.")
    with c3:
        st.subheader("3 · Heat context")
        if t is None:
            st.markdown("#### offline")
            st.caption("Open-Meteo not reachable - check the internet connection.")
        else:
            st.markdown(f"#### {t:.1f} °C · {city if city != 'Other (lat/lon)' else 'location'}")
            st.caption(f"Daily max (Open-Meteo) · injury rate ×{ratios[band]:.2f} vs 20-25 °C days in this sector type")
            st.caption(("Outdoor sector: " if sector in OUTDOOR else "Indoor sector: ") +
                       "rate ratio from our OSHA × Open-Meteo analysis (correlation, not causation).")
        if hot:
            st.error("Hot day for outdoor work - heat block added to today's toolbox talk.")

    st.subheader("4 · Similar real reports")
    for cse in r["similar"]:
        st.markdown(f"- **{cse['source']} · {cse['mechanism']} · {cse['outcome']}** ({cse['date']}, similarity {cse['similarity']:.2f})  \n  {cse['text']}")

    st.subheader("5 · Toolbox talk for the next shift")
    talk = toolbox.build(osha_for_talks(), sector, talk_lang, month=day.month, focus=r["hazard"], hot_day=hot)
    shown = "### " + talk.lstrip("# ").replace("\n## ", "\n#### ")  # smaller headings inside the app
    st.markdown(shown)
    st.download_button("Download toolbox talk (.md)", talk, file_name=f"toolbox_talk_{sector.split()[0].lower()}_{talk_lang}.md")
    st.caption("⚠️ Decision support only. Never use a LOW score to close a report without review. "
               "Models were trained on US data; Turkish input uses an experimental glossary.")
elif go:
    st.warning("Please paste a report first.")
