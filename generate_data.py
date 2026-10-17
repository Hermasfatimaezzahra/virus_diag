"""Génère data/patient_data.csv (données synthétiques pour l'exercice)."""
import csv
import numpy as np

rng = np.random.default_rng(42)
rows = []
for _ in range(300):
    infecte = int(rng.random() < 0.45)
    temp = round(rng.normal(38.8 if infecte else 36.9, 0.6), 1)
    fievre = int(rng.random() < (0.85 if infecte else 0.1))
    toux = int(rng.random() < (0.75 if infecte else 0.2))
    fatigue = int(rng.random() < (0.7 if infecte else 0.25))
    pcr = int(rng.random() < (0.92 if infecte else 0.05))
    rows.append([temp, fievre, toux, fatigue, pcr, infecte])

with open("data/patient_data.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["temperature", "fievre", "toux", "fatigue", "pcr", "infecte"])
    w.writerows(rows)
print("data/patient_data.csv généré :", len(rows), "patients")
