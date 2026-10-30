# SafeSignal – catching the near-misses that precede serious workplace accidents

**AI for the SDGs Capstone – Group 2** · Mehmet Emre Uğurlu, Mert Topçakar, Emre Aküzüm, Erdal Emre Ayaydın
**SDG 8 – Target 8.8** (safe and secure working environments), Indicator 8.8.1 · secondary link: SDG 13 (climate / heat)

> **The question we answer:** *Which small incidents are warning signs of a big accident?*
> SafeSignal reads free-text incident and near-miss reports, flags the ones that "read like" a fatal or disabling
> accident, adds weather context, and turns real accident reports into a **sector-specific 5-minute toolbox talk**
> (iş başı eğitimi) for the next shift.

## What changed after Assignment 1 feedback
| Feedback | What we did |
|---|---|
| The idea is too general – sharpen it | Narrowed the focus to **near-miss → serious-accident precursors** and a concrete output (toolbox talk) |
| Use OSHA Severe Injury Reports | Added **21,578 OSHA narratives** (2015-01 → 2017-02) with sector, date, location and injury codes |
| Türkiye: SGK, İSİG; UK: HSE for comparison | Added SGK 2024–25, İSİG Meclisi 2025 (sector + cause), HSE 2025/26 and BLS 2024 figures with sources |
| Add weather from Open-Meteo, test extreme heat in construction/agriculture | Built `04_weather_openmeteo.py`: joins every report to the day's max temperature (ERA5) and compares injury **rates** by temperature band |
| Focus on near-misses for originality | Added **MSHA mine reports (60,000)** that include **5,231 no-injury events**, trained a "serious-outcome" text model and scored every near-miss |
| Output: a tool that writes a sector-specific toolbox talk from real reports | Built `05_toolbox_talk.py` (Turkish / English, retrieval + templates, optional LLM rewrite) |

## Data
| File | Source | Rows | Used for |
|---|---|---|---|
| `data/ihm_stefanini_accidents.csv` | IHM Stefanini, Industrial Safety & Health Analytics DB (Kaggle) | 425 | Actual vs **potential** severity – the near-miss signal |
| `data/osha_severe_injury_2015_2017.csv.gz` | U.S. OSHA Severe Injury Reports (employer names/addresses removed) | 21,578 | Sector × mechanism EDA, heat analysis, mechanism classifier, toolbox talks |
| `data/msha_accidents_2016_2026.csv.gz` | U.S. MSHA mine accident reports (Form 7000-1), most recent 60,000 records | 60,000 | Severity pyramid and near-miss precursor model |
| `data/turkey_international_context.csv` | SGK, İSİG Meclisi, HSE (GB), BLS (US) – URLs in the file | 25 | Türkiye relevance and comparison |
| `data/weather_cache.csv` | Open-Meteo historical weather API (ERA5), downloaded 2026-10-31 | ~136k | Daily max temperature per location-day |

## Pipeline
| Script | What it does | Key result |
|---|---|---|
| `src/01_ihm_near_miss.py` | EDA + triage baseline on IHM | 28% of "Level I" events had Level IV+ potential; TF-IDF + LogReg macro-F1 0.68 (baseline 0.37) |
| `src/02_osha_eda.py` | Sector × mechanism profile, heat seasonality, mechanism classifier | Mechanism from text: **accuracy 0.90, macro-F1 0.79** (4,315 test reports); 82% of heat injuries in Jun–Aug |
| `src/03_msha_precursors.py` | Severity pyramid per mechanism; serious-outcome model (outcome words removed); scores all near-misses | **ROC-AUC 0.89, PR-AUC 0.21 vs 0.017 random**; 497 of 5,231 near-misses (9.5%) flagged as "reads like a serious accident" |
| `src/04_weather_openmeteo.py` | OSHA × Open-Meteo daily max temperature; injury rate per 1,000 location-days by temperature band | 172 grid cells, 135,880 location-days: construction + agriculture injury rate **+48% on 30–35 °C days** vs 20–25 °C (other sectors +24%); correlation, not yet controlled for seasonal workforce |
| `src/05_toolbox_talk.py` | Sector-specific 5-minute toolbox talk from real reports | Examples in `outputs/toolbox_talks/` |
| `src/06_turkey_context.py` | Türkiye / GB / US context figure | İSİG 2025: ≥2,105 worker deaths; SGK 2025: 1,931 |
| `src/07_ihm_triage_demo.py` | Scores a new free-text report | "the chain broke and the load fell…" → P(high) = 0.73 |

## How to run
```bash
pip install -r requirements.txt
python run_all.py                       # everything, incl. Open-Meteo download
python src/05_toolbox_talk.py --sector Construction --lang tr --month 7
python src/05_toolbox_talk.py --list    # available sectors
```

## Why these modelling choices
* **Text → mechanism** and **text → serious-outcome** are both TF-IDF + logistic regression: fast, measurable and
  explainable (each flag comes with the words behind it). Transformer embeddings and an LLM classifier are the
  Assignment 2 comparison.
* **Label leakage control:** words that describe the *outcome* (died, amputated, hospital, surgery …) are removed
  before training the serious-outcome model, so it learns the *mechanism* – which is all a near-miss report contains.
* **Near-misses are held out** from training and only scored, so the model never sees the reports it ranks.
* **Weather uses rates, not counts:** injuries per 1,000 location-days in each temperature band, so hot bands are
  not favoured just because there are more summer working days.

## Ethics (summary)
False negatives (a dangerous near-miss marked low) are the main risk → the tool ranks, a safety officer decides.
The data are U.S. / one Latin-American company, so results must be re-validated on Turkish reports before use.
No personal data is used as a feature (OSHA employer names and addresses were removed). The optional LLM step may
only rephrase retrieved facts; it never adds numbers or medical advice.

## Sources
ILO (2023) *A Call for Safer and Healthier Working Environments* · UN SDG metadata 8.8.1 · OSHA Severe Injury
Reports · MSHA Accident Injuries data · IHM Stefanini (Kaggle) · SGK statistics as reported in Evrensel (2026) ·
İSİG Meclisi 2025 report as reported in Bianet · HSE fatal injury statistics 2025/26 · BLS CFOI 2024 · Open-Meteo
Historical Weather API (ERA5).
