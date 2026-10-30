"""Run the whole SafeSignal pipeline in order. Weather step needs internet (Open-Meteo)."""
import subprocess, sys
steps = ["01_ihm_near_miss.py", "02_osha_eda.py", "03_msha_precursors.py", "06_turkey_context.py", "04_weather_openmeteo.py"]
for s in steps:
    print(f"\n=== {s} ===")
    r = subprocess.run([sys.executable, f"src/{s}"])
    if r.returncode != 0:
        print(f"[!] {s} failed (weather needs internet) - continuing")
for sec in ["Construction", "Manufacturing", "Agriculture", "Transport & warehousing"]:
    subprocess.run([sys.executable, "src/05_toolbox_talk.py", "--sector", sec, "--lang", "tr", "--month", "7"], stdout=subprocess.DEVNULL)
print("\nDone. See figures/, results/ and outputs/toolbox_talks/")
