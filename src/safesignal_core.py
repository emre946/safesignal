"""
SafeSignal core: shared label maps, text cleaning, model training/loading and the end-to-end `analyse()` call
used by both the Streamlit app (app.py) and the command line.
"""
import json
import re
import urllib.parse
import urllib.request
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.pairwise import linear_kernel

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

EVENT_GROUP = {  # first digit of the OIICS event code
    "1": "Violence / animals", "2": "Transportation", "3": "Fire / explosion",
    "4": "Falls, slips, trips", "5": "Harmful exposure (heat, electricity, chemicals)",
    "6": "Contact with objects / equipment", "7": "Overexertion",
}
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
                 "fingertip", "resulting", "removed", "appears", "cpr", "medical", "emt", "stitches", "fracture",
                 "fractured", "hernia", "loss", "ongoing", "unknown", "lost",
                 "near", "miss", "hurt", "harmed", "unharmed", "uninjured", "accident", "incident", "only"}

# Experimental Turkish -> English safety glossary so Turkish reports can be scored by the English-trained models.
# Longest phrases first. Unknown Turkish words are simply ignored by the English vocabulary.
TR_GLOSSARY = [
    ("elektrik çarptı", "electric shock electrocuted"), ("elektrik çarpması", "electric shock"), ("yüksekten düştü", "fell from height"),
    ("yüksekten düşme", "fall from height"), ("ramak kala", "near miss"), ("iş makinesi", "heavy equipment machine"),
    ("geri manevra", "backing up reversing"), ("havai hat", "overhead power line"), ("enerji hattı", "power line"),
    ("elinden düştü", "dropped falling object struck"), ("elinden", "dropped object struck"), ("elinden kaydı", "dropped falling object"), ("çekiç", "hammer tool"),
    ("anahtar", "wrench tool"), ("malzeme düştü", "falling material struck"), ("yukarıdan", "from above falling object"),
    ("aşağıda", "below"), ("aşağıya", "below"), ("kıl payı", "near miss"), ("az kalsın", "near miss"), ("yaya", "pedestrian"),
    ("yayanın", "pedestrian"), ("koruyucu", "guard"), ("iskele", "scaffold"), ("iskeleden", "scaffold"), ("merdiven", "ladder"), ("merdivenden", "ladder"),
    ("çatı", "roof"), ("çatıdan", "roof"), ("düştü", "fell"), ("düşen", "falling object"), ("düşme", "fall"), ("kaydı", "slipped"),
    ("kaydı ve", "slipped"), ("takıldı", "tripped"), ("sıkıştı", "caught pinched"), ("sıkışması", "caught pinched"), ("sıkışma", "caught"),
    ("kaptırdı", "caught in machine"), ("ezildi", "crushed"), ("ezilme", "crushed"), ("göçük", "collapse cave in"), ("göçtü", "collapsed"),
    ("çarptı", "struck"), ("çarpma", "struck"), ("çarpması", "struck"), ("vinç", "crane"), ("vinçle", "crane"), ("yük", "load"),
    ("yükü", "load"), ("halat", "rope sling"), ("sapan", "sling"), ("zincir", "chain"), ("koptu", "broke snapped"), ("kırıldı", "broke"),
    ("forklift", "forklift"), ("kamyon", "truck"), ("dozer", "dozer"), ("ekskavatör", "excavator"), ("kepçe", "loader bucket"),
    ("pres", "press"), ("presin", "press"), ("bant", "belt conveyor"), ("konveyör", "conveyor"), ("taşıyıcı", "conveyor"),
    ("testere", "saw"), ("taşlama", "grinder"), ("matkap", "drill"), ("delme", "drilling"), ("kalıp", "mold die"),
    ("makine", "machine"), ("makineye", "machine"), ("makinenin", "machine"), ("dönen", "rotating"), ("mil", "shaft"),
    ("eldiven", "glove"), ("el", "hand"), ("eli", "hand"), ("parmak", "finger"), ("parmağı", "finger"), ("kol", "arm"), ("kolu", "arm"),
    ("ayak", "foot"), ("kafa", "head"), ("başına", "head"), ("göz", "eye"), ("gözüne", "eye"),
    ("sıcak", "hot heat"), ("sıcaklık", "heat temperature"), ("güneş", "sun heat"), ("bayıldı", "fainted collapsed heat"),
    ("yanık", "burn"), ("yandı", "burned"), ("yangın", "fire"), ("patlama", "explosion"), ("patladı", "exploded"), ("kaynak", "welding"),
    ("kimyasal", "chemical"), ("asit", "acid"), ("gaz", "gas"), ("zehirlenme", "poisoning toxic"), ("toz", "dust"),
    ("elektrik", "electric electrical"), ("kablo", "cable cord"), ("pano", "electrical panel"),
    ("kaldırırken", "lifting"), ("kaldırma", "lifting"), ("taşırken", "carrying"), ("bel", "back strain"), ("zorlanma", "strain"),
    ("araç", "vehicle"), ("aracı", "vehicle"), ("trafik", "traffic vehicle"), ("servis", "vehicle"), ("devrildi", "overturned rolled over"),
    ("boşluk", "opening hole"), ("boşluğa", "opening hole"), ("döşeme", "floor"), ("kenar", "edge"), ("korkuluk", "guardrail"),
    ("emniyet kemeri", "harness"), ("kemer", "harness"), ("baret", "hard hat"), ("hendek", "trench"), ("kazı", "excavation"),
    ("kaya", "rock"), ("tavan", "roof ceiling"), ("ocak", "mine"), ("maden", "mine"), ("işçi", "worker"), ("operatör", "operator"),
    ("çalışan", "worker"), ("kimse yaralanmadı", "no one hurt"), ("yaralanmadı", "not hurt"), ("metre", "meter"), ("santim", "inches"),
]


_STEMS = ("amputat", "fatal", "sever", "hospital", "deceas", "disab", "fractur", "surger", "injur")
_OUTCOME_RX = re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, OUTCOME_WORDS), key=len, reverse=True)) + r")\b"
                         r"|\b(?:" + "|".join(_STEMS) + r")\w*", re.I)


def clean_text(t):
    """Lower-case, drop numbers and REMOVE outcome words/stems from the text itself, so no model
    (word, character or embedding based) can see how the accident ended."""
    t = re.sub(r"\d+", " ", str(t).lower())
    return re.sub(r"\s+", " ", _OUTCOME_RX.sub(" ", t)).strip()


def looks_turkish(text):
    return bool(re.search(r"[çğışöüÇĞİŞÖÜ]", text)) or bool(re.search(r"\b(ve|bir|ile|sırasında|olan|için)\b", text.lower()))


def tr_to_en(text):
    t = text.lower().replace("i̇", "i")
    hits = []
    for tr, en in sorted(TR_GLOSSARY, key=lambda x: -len(x[0])):
        if re.search(r"(?<!\w)" + re.escape(tr), t):
            hits.append(en)
            t = re.sub(r"(?<!\w)" + re.escape(tr) + r"\w*", " ", t)
    return " ".join(hits)


# ---------------------------------------------------------------- training
def _load_osha():
    o = pd.read_csv(DATA / "osha_severe_injury_2015_2017.csv.gz", low_memory=False)
    o = o.dropna(subset=["Final Narrative"]).drop_duplicates(subset=["ID"])
    o["Mechanism"] = o["Event"].astype(str).str[0].map(EVENT_GROUP).fillna("Other")
    return o


def _load_msha():
    m = pd.read_csv(DATA / "msha_accidents_2016_2026.csv.gz", low_memory=False)
    m["Tier"] = m["DEGREE_INJURY"].map(TIER)
    return m.dropna(subset=["Tier", "NARRATIVE"])


def train_models():
    """Train the two selected models (see 08_model_comparison.py) on all data and build the retrieval index."""
    o = _load_osha()
    keep = o.Mechanism.map(o.Mechanism.value_counts()) >= 100
    # selected in 08_model_comparison.py: character 3-5 grams + linear SVM (best macro-F1), calibrated for probabilities
    mech_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, sublinear_tf=True, max_features=150000)
    Xo = mech_vec.fit_transform(o.loc[keep, "Final Narrative"])
    mech_clf = CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3).fit(Xo, o.loc[keep, "Mechanism"])

    m = _load_msha()
    inj = m[m.Tier != "0 Near-miss (no injury)"]
    prec_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True,
                               stop_words=list(ENGLISH_STOP_WORDS | OUTCOME_WORDS), preprocessor=clean_text)
    Xm = prec_vec.fit_transform(inj.NARRATIVE)
    prec_clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0).fit(Xm, (inj.Tier == "3 Serious").astype(int))

    # retrieval index of real cases (OSHA + MSHA) for "similar real reports"
    cases = pd.concat([
        pd.DataFrame({"source": "OSHA", "date": o.EventDate.astype(str).str[:10], "text": o["Final Narrative"],
                      "outcome": np.where(o.Amputation.fillna(0) > 0, "amputation", "hospitalised"), "mechanism": o.Mechanism}),
        pd.DataFrame({"source": "MSHA", "date": m.ACCIDENT_DT.astype(str).str[:10], "text": m.NARRATIVE,
                      "outcome": m.Tier.str[2:], "mechanism": m.CLASSIFICATION.str.title()}),
    ], ignore_index=True)
    cases = cases[cases.text.str.split().str.len().between(12, 90)].reset_index(drop=True)
    ret_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
    R = ret_vec.fit_transform(cases.text)
    # reference distribution: scores of all historical no-injury events (near-misses), for percentile ranks
    # reference distribution for percentile ranks: scores of all mine incident reports with an injury outcome
    # (the population on which "top 10% catches ~65% of serious outcomes" was measured)
    nm_scores = np.sort(prec_clf.predict_proba(Xm)[:, 1])
    return {"mech_vec": mech_vec, "mech_clf": mech_clf, "prec_vec": prec_vec, "prec_clf": prec_clf,
            "ret_vec": ret_vec, "R": R, "cases": cases, "nm_scores": nm_scores}


def load_or_train():
    import joblib
    MODELS.mkdir(exist_ok=True)
    f = MODELS / "safesignal_models.joblib"
    if f.exists():
        try:
            return joblib.load(f)
        except Exception:
            pass
    M = train_models()
    joblib.dump(M, f, compress=3)
    return M


# ---------------------------------------------------------------- weather (Open-Meteo)
def max_temp(lat, lon, date):
    """Daily maximum temperature (°C) for a place and date: archive API for past days, forecast API for today/future."""
    d = pd.Timestamp(date)
    past = d < pd.Timestamp.today().normalize() - pd.Timedelta(days=5)
    api = "https://archive-api.open-meteo.com/v1/archive" if past else "https://api.open-meteo.com/v1/forecast"
    q = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "start_date": d.strftime("%Y-%m-%d"),
                                "end_date": d.strftime("%Y-%m-%d"), "daily": "temperature_2m_max", "timezone": "auto"})
    with urllib.request.urlopen(f"{api}?{q}", timeout=20) as r:
        js = json.loads(r.read().decode())
    return float(js["daily"]["temperature_2m_max"][0])


# from results/weather_heat_rates.csv (src/04_weather_openmeteo.py, run 2026-11-23)
HEAT_RATIO_OUTDOOR = {"<10": 0.78, "10-20": 0.86, "20-25": 1.00, "25-30": 1.20, "30-35": 1.48, "≥35": 1.34}
HEAT_RATIO_OTHER = {"<10": 1.27, "10-20": 1.05, "20-25": 1.00, "25-30": 1.16, "30-35": 1.23, "≥35": 1.22}


def heat_band(t):
    return "<10" if t < 10 else "10-20" if t < 20 else "20-25" if t < 25 else "25-30" if t < 30 else "30-35" if t < 35 else "≥35"


def mechanism_to_hazard(mech, text):
    """Map a predicted OSHA mechanism group to the hazard keys used by the toolbox talk generator."""
    t = text.lower()
    if re.search(r"power line|overhead line|electr|volt|shock", t):
        return "electrical"
    if mech.startswith("Falls"):
        return "fall_height" if re.search(r"ladder|roof|scaffold|height|feet|lift|platform|opening|edge", t) else "fall_same"
    if mech.startswith("Contact"):
        return "caught" if re.search(r"caught|pinch|machine|press|conveyor|roller|belt|shaft|rotating|gear|saw", t) else "struck"
    if mech.startswith("Transportation"):
        return "vehicle"
    if mech.startswith("Harmful"):
        return "electrical" if re.search(r"electr|power|volt|shock|line", t) else "heat" if re.search(r"heat|hot|sun|temperature", t) else "chemical"
    if mech.startswith("Fire"):
        return "hot"
    if mech.startswith("Overexertion"):
        return "overexertion"
    return None


# ---------------------------------------------------------------- end to end
def analyse(M, text, top_n_cases=3):
    """Report text (English or Turkish) -> mechanism, precursor score, evidence words, similar real cases."""
    lang = "tr" if looks_turkish(text) else "en"
    model_text = tr_to_en(text) if lang == "tr" else text
    out = {"language": lang, "model_text": model_text}
    if not model_text.strip():
        out["error"] = "No safety-related words recognised."
        return out
    # mechanism
    xm = M["mech_vec"].transform([model_text])
    pm = M["mech_clf"].predict_proba(xm)[0]
    order = np.argsort(pm)[::-1]
    out["mechanism"] = [(M["mech_clf"].classes_[i], float(pm[i])) for i in order[:3]]
    # precursor (selected: word TF-IDF + logistic regression - ties on PR-AUC, best ROC-AUC, explainable)
    xp = M["prec_vec"].transform([model_text])
    p = float(M["prec_clf"].predict_proba(xp)[0, 1])
    contrib = xp.toarray()[0] * M["prec_clf"].coef_[0]
    names = M["prec_vec"].get_feature_names_out()
    idx = np.argsort(contrib)[::-1]
    out["p_serious_like"] = p
    out["evidence"] = [names[i] for i in idx[:6] if contrib[i] > 0]
    out["coverage"] = int(xp.nnz)
    pct = float(np.searchsorted(M["nm_scores"], p) / len(M["nm_scores"]) * 100)
    out["percentile"] = pct  # riskier-looking than pct% of ~52k historical mine incident reports
    out["level"] = "HIGH" if pct >= 90 else "MEDIUM" if pct >= 70 else "LOW"
    out["hazard"] = mechanism_to_hazard(out["mechanism"][0][0], model_text)
    # similar real cases
    sims = linear_kernel(M["ret_vec"].transform([model_text]), M["R"]).ravel()
    top = np.argsort(sims)[::-1][:top_n_cases]
    out["similar"] = [dict(M["cases"].iloc[i], similarity=float(sims[i])) for i in top if sims[i] > 0]
    return out


if __name__ == "__main__":
    import sys
    M = load_or_train()
    for t in sys.argv[1:] or ["While dumping, the haul truck pulled forward with the bed raised and contacted the overhead power line. No one was hurt.",
                              "Forklift geri manevra yaparken yayanın yanından geçti, kıl payı çarpmadı. Kimse yaralanmadı."]:
        r = analyse(M, t)
        print("\n", t); print({k: v for k, v in r.items() if k != "similar"})
        for c in r.get("similar", []):
            print("   ~", round(c["similarity"], 2), c["source"], c["mechanism"], "|", c["text"][:110])
