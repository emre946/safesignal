# SafeSignal – 2–3 minute demo recording script (Assignment 2)

**Before recording**
1. `pip install -r requirements.txt`
2. `streamlit run app.py` → the browser opens at http://localhost:8501
3. Click any example and **Analyse** once, so the models are trained and cached (first time ~30 s).
4. Sidebar: Sector **Construction**, Location **İstanbul**, Date: a hot summer day, e.g. **2025-07-24** (if the card shows under 30 °C, try another July day), talk language **Türkçe**.
5. Start the recorder: **Win + Alt + R** (Xbox Game Bar) or Snipping Tool → Record. Stop with the same keys.

| Time | On screen | Say (English) |
|---|---|---|
| 0:00–0:20 | App title | "This is SafeSignal, our Assignment 2 prototype for SDG 8.8. It reads incident and near-miss reports and flags the ones that look like they could have been fatal, so safety officers can act before the next accident." |
| 0:20–0:50 | Click **Real MSHA near-miss (EN)** → **Analyse report** | "This is a real no-injury report from US mine data: a truck dumped with its box up and touched overhead power lines. Nobody was hurt." |
| 0:50–1:20 | Point at cards 1 and 2 | "The mechanism model – a character n-gram SVM, our best model with 0.80 macro-F1 – classifies the mechanism. The precursor model, a logistic regression with 0.89 ROC-AUC, rates it HIGH: it reads riskier than 99% of past mine reports. The evidence words – overhead power, dumping, truck – show why." |
| 1:20–1:40 | Card 3 + section 4 | "The heat card calls Open-Meteo for the day's maximum temperature. In our analysis outdoor injury rates were 48% higher on 30–35 °C days. Below are the most similar real accident reports – here the near-miss could have become an electrocution." |
| 1:40–2:10 | Click **Forklift, Turkish (TR)** → **Analyse** | "Our users are in Türkiye, so the tool also accepts Turkish reports through an experimental glossary. This forklift near-miss is correctly classified as transportation." |
| 2:10–2:40 | Scroll the toolbox talk, click **Download** | "Finally it writes a five-minute toolbox talk for the next shift, in Turkish, built from real cases, with the warning signs to report and today's controls. On a hot day it adds a heat block." |
| 2:40–2:55 | Bottom warning | "SafeSignal only ranks reports – a safety officer always decides. Thank you." |
