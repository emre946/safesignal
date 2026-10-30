"""
05 - Toolbox talk generator: a sector-specific 5-minute pre-shift safety briefing built from REAL accident reports.

Teacher feedback: "Çıktı olarak gerçek kaza raporlarından sektöre özel kısa bir iş başı eğitimi hazırlayan bir araç."

How it works (retrieval + templates, no API key needed):
  1. Pick the sector's most frequent and most severe injury mechanisms from 21.5k OSHA reports.
  2. Retrieve short, real narratives for those mechanisms (amputation / hospitalisation cases first).
  3. Add the warning signs (near-miss precursors) and controls for each mechanism, plus a heat block
     for outdoor sectors in the hot season.
  4. Optional: with --llm and an ANTHROPIC_API_KEY, an LLM rewrites the briefing in plainer language.
     The LLM only rephrases the retrieved facts; it never adds statistics.

Usage:
  python src/05_toolbox_talk.py --sector Construction --lang tr --month 7
  python src/05_toolbox_talk.py --list
"""
import argparse
import os
import re

import pandas as pd

from common import DATA, OUT, NAICS2

HAZARDS = {  # key: (regex on OSHA EventTitle, EN name, TR name)
    "caught": (r"caught in|compressed or pinched|crushed|entangled", "Caught in / between machinery", "Makineye sıkışma / kaptırma"),
    "fall_height": (r"fall.*lower level|fall from|fall through", "Fall from height", "Yüksekten düşme"),
    "fall_same": (r"fall on same level|slip|trip", "Slips, trips and same-level falls", "Kayma, takılma, aynı seviyede düşme"),
    "struck": (r"struck by|struck against|slipping or swinging", "Struck by objects / equipment", "Malzeme / ekipman çarpması"),
    "vehicle": (r"vehicle|pedestrian|forklift|roadway|transport", "Vehicles and mobile equipment", "Araç ve iş makineleri"),
    "heat": (r"environmental heat", "Heat stress", "Sıcak çarpması / ısı stresi"),
    "hot": (r"hot objects|fire|explosion|burn|ignition", "Burns, fire and hot surfaces", "Yanık, yangın ve sıcak yüzeyler"),
    "electrical": (r"electric", "Electrical contact", "Elektrik çarpması"),
    "chemical": (r"chemical|caustic|substance|inhalation|toxic|intact skin", "Chemicals and harmful substances", "Kimyasallar ve zararlı maddeler"),
    "overexertion": (r"overexertion|lifting|bodily", "Overexertion and manual handling", "Aşırı zorlanma ve elle taşıma"),
}

SIGNS = {  # warning signs = near-miss precursors to report BEFORE someone is hurt
    "caught": {"en": ["Guards removed, bypassed or loose", "Clearing jams or cleaning while the machine is energised",
                      "Gloves, sleeves or rags near rotating parts"],
               "tr": ["Koruyucusu sökülmüş, devre dışı bırakılmış veya gevşek makineler",
                      "Makine enerjiliyken sıkışma giderme ya da temizlik yapılması",
                      "Dönen parçaların yakınında eldiven, bol kol veya bez"]},
    "fall_height": {"en": ["Missing or loose guardrails, open edges or floor openings", "Ladders not tied off or on uneven ground",
                           "Harness worn but not clipped on"],
                    "tr": ["Eksik/gevşek korkuluk, açık kenar veya döşeme boşluğu", "Bağlanmamış ya da düz olmayan zemine kurulmuş merdiven",
                           "Emniyet kemeri takılı ama bir noktaya bağlanmamış"]},
    "fall_same": {"en": ["Oil, water or debris on walkways", "Poor lighting or cables across the floor", "Rushing while carrying loads"],
                  "tr": ["Yürüme yolunda yağ, su veya malzeme", "Yetersiz aydınlatma, yerde kablolar", "Yük taşırken acele etmek"]},
    "struck": {"en": ["Loads lifted over people or without tag lines", "Unsecured material stacked at height",
                      "Tools or parts dropped from above (even with no one hit)"],
               "tr": ["İnsanların üzerinden ya da kılavuz ip olmadan yük kaldırma", "Yüksekte sabitlenmemiş istif",
                      "Yukarıdan düşen alet/parça (kimseye çarpmasa bile)"]},
    "vehicle": {"en": ["Pedestrians and forklifts sharing the same aisle", "Reversing without a spotter or alarm",
                       "Near-collisions at blind corners"],
                "tr": ["Yayalarla forkliftin aynı koridoru kullanması", "Gözcü ya da geri vites ikazı olmadan geri manevra",
                       "Kör köşelerde kıl payı çarpışmalar"]},
    "heat": {"en": ["Workers new to the heat or back from leave", "Headache, dizziness, cramps or confusion",
                    "No shade or drinking water near the work area"],
             "tr": ["Sıcağa yeni başlayan ya da izinden dönen çalışanlar", "Baş ağrısı, baş dönmesi, kramp veya bilinç bulanıklığı",
                    "Çalışma alanının yakınında gölge veya içme suyu olmaması"]},
    "hot": {"en": ["Hot work without a fire watch", "Leaking hoses or fittings near heat sources", "Small fires that were 'put out quickly'"],
            "tr": ["Yangın gözcüsü olmadan sıcak iş", "Isı kaynağı yakınında sızdıran hortum/bağlantı", "'Hemen söndürülen' küçük yangınlar"]},
    "electrical": {"en": ["Damaged cords or exposed conductors", "Equipment working close to overhead power lines",
                          "Small shocks or tingles from tools"],
                   "tr": ["Hasarlı kablo veya açıkta iletken", "Havai hatların yakınında çalışan ekipman", "Aletlerden gelen küçük çarpmalar/karıncalanma"]},
    "chemical": {"en": ["Unlabelled containers", "Strong smells or irritation reported by workers", "Missing or worn gloves and goggles"],
                 "tr": ["Etiketsiz kaplar", "Çalışanların bildirdiği keskin koku veya tahriş", "Eksik ya da yıpranmış eldiven ve gözlük"]},
    "overexertion": {"en": ["Repeated lifting above shoulder height", "Loads moved by one person that need two", "Workers reporting back pain"],
                     "tr": ["Omuz hizasının üstüne sık kaldırma", "İki kişilik yükün tek kişiyle taşınması", "Bel ağrısı bildiren çalışanlar"]},
}

CONTROLS = {
    "caught": {"en": ["Lock out / tag out before clearing, cleaning or maintenance - every time",
                      "Never reach past a guard; report missing guards immediately"],
               "tr": ["Sıkışma giderme, temizlik ve bakım öncesi her seferinde Kilitle / Etiketle (LOTO)",
                      "Koruyucunun ötesine asla uzanma; eksik koruyucuyu hemen bildir"]},
    "fall_height": {"en": ["100% tie-off above 1.8 m; inspect harness and anchor before use", "Cover and mark every opening"],
                    "tr": ["1,8 m üzerinde sürekli bağlı çalış; kullanmadan önce kemeri ve ankrajı kontrol et",
                           "Her döşeme boşluğunu kapat ve işaretle"]},
    "fall_same": {"en": ["Clean spills at once and keep walkways clear", "Wear slip-resistant footwear"],
                  "tr": ["Dökülen sıvıyı hemen temizle, yürüme yollarını açık tut", "Kaymaz tabanlı ayakkabı giy"]},
    "struck": {"en": ["Stay out of the load path; use tag lines", "Secure tools and material at height"],
               "tr": ["Yükün altına ve güzergâhına girme; kılavuz ip kullan", "Yüksekteki alet ve malzemeyi sabitle"]},
    "vehicle": {"en": ["Separate pedestrian and vehicle routes", "Make eye contact with the driver before crossing"],
                "tr": ["Yaya ve araç yollarını ayır", "Geçmeden önce operatörle göz teması kur"]},
    "heat": {"en": ["Water - rest - shade: drink every 15-20 minutes", "Buddy system; stop work at first symptoms"],
             "tr": ["Su - dinlenme - gölge: 15-20 dakikada bir su iç", "Eşli çalış; ilk belirtide işi durdur"]},
    "hot": {"en": ["Hot-work permit and fire watch", "Check hoses and fittings before starting"],
            "tr": ["Sıcak iş izni ve yangın gözcüsü", "Başlamadan önce hortum ve bağlantıları kontrol et"]},
    "electrical": {"en": ["De-energise and verify before work", "Keep the safe distance from overhead lines"],
                   "tr": ["Çalışmadan önce enerjiyi kes ve kontrol et", "Havai hatlarla güvenli mesafeyi koru"]},
    "chemical": {"en": ["Read the label / SDS before use", "Wear the right gloves and eye protection"],
                 "tr": ["Kullanmadan önce etiketi / SDS'i oku", "Doğru eldiven ve göz koruyucu kullan"]},
    "overexertion": {"en": ["Use lifting aids or a second person", "Lift with legs, keep the load close"],
                     "tr": ["Kaldırma ekipmanı ya da ikinci kişi kullan", "Bacaklarla kaldır, yükü vücuda yakın tut"]},
}

T = {
    "en": {"title": "5-minute toolbox talk", "why": "Why this matters today", "cases": "What really happened (real OSHA reports)",
           "signs": "Warning signs - report these as near-misses", "do": "What we do today", "ask": "Questions for the crew",
           "q": ["Where on our site could this happen this week?", "Has anyone seen one of these warning signs recently?",
                 "What will we do differently on this shift?"],
           "src": "Source: OSHA Severe Injury Reports {a}-{b}, {n:,} reports in this sector. Generated by SafeSignal.",
           "stat": "In {n:,} severe injury reports from {sector}, the most common causes were:",
           "amp": "{p}% of these reports involved an amputation."},
    "tr": {"title": "5 dakikalık iş başı eğitimi", "why": "Bugün neden önemli", "cases": "Gerçekte ne oldu (gerçek OSHA raporları)",
           "signs": "Uyarı işaretleri - bunları ramak kala olarak bildir", "do": "Bugün ne yapıyoruz", "ask": "Ekibe sorular",
           "q": ["Bu hafta sahamızda bu nerede olabilir?", "Son zamanlarda bu uyarı işaretlerinden birini gören oldu mu?",
                 "Bu vardiyada neyi farklı yapacağız?"],
           "src": "Kaynak: OSHA Ağır Yaralanma Raporları {a} - {b}, bu sektörde {n:,} rapor. SafeSignal tarafından üretildi.",
           "stat": "{sector} sektöründeki {n:,} ağır yaralanma raporunda en sık nedenler:",
           "amp": "Uzuv kaybı (amputasyon) olan rapor oranı: %{p}."},
}
SECTOR_TR = {"Construction": "İnşaat", "Manufacturing": "İmalat", "Agriculture": "Tarım", "Wholesale trade": "Toptan ticaret",
             "Transport & warehousing": "Taşımacılık ve depolama", "Retail trade": "Perakende", "Mining & oil/gas": "Madencilik",
             "Admin & waste services": "İdari ve atık hizmetleri", "Health care": "Sağlık", "Accommodation & food": "Konaklama ve yiyecek",
             "Utilities": "Elektrik, su, gaz"}


def hazard_of(title):
    for k, (rx, *_) in HAZARDS.items():
        if re.search(rx, str(title), re.I):
            return k
    return None


def load():
    df = pd.read_csv(DATA / "osha_severe_injury_2015_2017.csv.gz", low_memory=False)
    df["EventDate"] = pd.to_datetime(df.EventDate, errors="coerce")
    df["Sector"] = df["Primary NAICS"].astype(str).str[:2].map(NAICS2).fillna("Other")
    df["Hazard"] = df.EventTitle.map(hazard_of)
    df["Words"] = df["Final Narrative"].str.split().str.len()
    return df


def build(df, sector, lang="en", month=None, n_hazards=3):
    L = T[lang]
    s = df[df.Sector == sector]
    if s.empty:
        raise SystemExit(f"Unknown sector '{sector}'. Use --list.")
    top = s.Hazard.value_counts().head(n_hazards).index.tolist()
    outdoor = sector in ("Construction", "Agriculture")
    if month in (6, 7, 8, 9) and outdoor and "heat" not in top:
        top.append("heat")
    name = SECTOR_TR.get(sector, sector) if lang == "tr" else sector
    hz = lambda k: HAZARDS[k][2] if lang == "tr" else HAZARDS[k][1]

    md = [f"# {L['title']}: {name}", "", f"## {L['why']}", L["stat"].format(n=len(s), sector=name), ""]
    for k in top:
        share = (s.Hazard == k).mean() * 100
        md.append(f"- **{hz(k)}** - " + (f"%{share:.0f}" if lang == "tr" else f"{share:.0f}%"))
    md += ["", L["amp"].format(p=round((s.Amputation.fillna(0) > 0).mean() * 100)), "", f"## {L['cases']}"]
    for k in top[:2]:
        c = s[(s.Hazard == k) & s.Words.between(20, 60)].copy()
        c["sev"] = c.Amputation.fillna(0) * 2 + c.Hospitalized.fillna(0)
        if not c.empty:
            r = c.sort_values(["sev", "Words"], ascending=[False, True]).iloc[0]
            md.append(f"> *{hz(k)}* ({r.EventDate:%Y-%m}): {r['Final Narrative'].strip()}")
            md.append("")
    md.append(f"## {L['signs']}")
    for k in top:
        md += [f"- **{hz(k)}:** " + "; ".join(SIGNS[k][lang])]
    md += ["", f"## {L['do']}"]
    for k in top:
        md += [f"- [ ] {c}" for c in CONTROLS[k][lang]]
    md += ["", f"## {L['ask']}"] + [f"{i}. {q}" for i, q in enumerate(L["q"], 1)]
    md += ["", "---", "*" + L["src"].format(a=f"{df.EventDate.min():%Y-%m}", b=f"{df.EventDate.max():%Y-%m}", n=len(s)) + "*"]
    out = "\n".join(md)
    if lang == "tr":  # Turkish thousands separator
        out = re.sub(r"(?<=\d),(?=\d{3}\b)", ".", out)
    return out


def llm_rewrite(text, lang):
    """Optional: plain-language rewrite with Claude. Facts and numbers must stay unchanged."""
    import anthropic  # pip install anthropic
    client = anthropic.Anthropic()
    lang_name = "Turkish" if lang == "tr" else "English"
    msg = client.messages.create(
        model="claude-sonnet-4-5", max_tokens=1500,
        messages=[{"role": "user", "content": (
            f"Rewrite this toolbox talk in short, plain {lang_name} that a site supervisor can read aloud in 5 minutes. "
            "Keep every number, percentage and quoted case exactly as given. Do not add new facts, statistics or "
            "medical advice. Keep the markdown headings.\n\n" + text)}])
    return msg.content[0].text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", default="Construction")
    ap.add_argument("--lang", choices=["en", "tr"], default="tr")
    ap.add_argument("--month", type=int, default=None, help="1-12; adds a heat block for outdoor sectors in summer")
    ap.add_argument("--llm", action="store_true", help="rewrite with Claude (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    data = load()
    if a.list:
        print(data.Sector.value_counts().to_string()); raise SystemExit
    talk = build(data, a.sector, a.lang, a.month)
    if a.llm and os.getenv("ANTHROPIC_API_KEY"):
        talk = llm_rewrite(talk, a.lang)
    fn = OUT / "toolbox_talks" / f"{a.sector.lower().replace(' & ', '_').replace(' ', '_').replace('/', '_')}_{a.lang}.md"
    fn.parent.mkdir(parents=True, exist_ok=True)
    fn.write_text(talk, encoding="utf-8")
    print(talk); print(f"\nSaved -> {fn}")
