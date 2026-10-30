"""Shared paths, plotting style and helpers for all SafeSignal scripts."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
RES = ROOT / "results"
OUT = ROOT / "outputs"
for p in (FIG, RES, OUT):
    p.mkdir(exist_ok=True)

# validated reference palette (light mode)
BLUE, ORANGE, AQUA, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#9a9893"
SEQ = ["#cfe0f6", "#9cc0ec", "#5f9be0", "#2a78d6", "#1d58a3", "#123a6e"]
INK, INK2, GRIDC = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": GRIDC, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlecolor": INK,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

# 2-digit NAICS sector names (US Census)
NAICS2 = {
    "11": "Agriculture", "21": "Mining & oil/gas", "22": "Utilities", "23": "Construction",
    "31": "Manufacturing", "32": "Manufacturing", "33": "Manufacturing",
    "42": "Wholesale trade", "44": "Retail trade", "45": "Retail trade",
    "48": "Transport & warehousing", "49": "Transport & warehousing",
    "51": "Information", "52": "Finance", "53": "Real estate", "54": "Professional services",
    "55": "Management", "56": "Admin & waste services", "61": "Education", "62": "Health care",
    "71": "Arts & recreation", "72": "Accommodation & food", "81": "Other services", "92": "Public administration",
}


def save(fig, name):
    fig.savefig(FIG / name, facecolor="white")
    plt.close(fig)
