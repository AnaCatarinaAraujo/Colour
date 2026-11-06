"""
Sensitivity check (does NOT change the pipeline): how much do the dark pixels
added to the grain mask by the morphological closing affect the colour?

For every grain photo in src/treatment_mapping.csv:
  * mask = segment_chickpeas(img)  (unchanged, as in the colour analysis)
  * T    = Otsu threshold of the HSV V channel (same call as in
           segment_chickpeas); pixels with V <= T are "below threshold"
  * dark = mask pixels with V <= T  (only the closing can add them; the
           opening only removes pixels)
  * L*a*b* with all mask pixels vs without the dark pixels, both measured
    with extract_color_values and calibrated with model_selected.json
  * dE*ab (CIE76) between the two calibrated values

Outputs: outputs/calibration/dark_pixels/
Usage (from the repository root): python src/analysis_dark_pixels.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colour.io import read_image_unicode  # noqa: E402
from colour.vision import segment_chickpeas  # noqa: E402
from colour.core import extract_color_values  # noqa: E402
from colour.calibration import apply_calibration, load_model, delta_e76  # noqa: E402
from run_calibration import TREATMENT_ORDER  # noqa: E402

IMAGES = Path("images")
MAPPING = Path("src/treatment_mapping.csv")
MODEL = Path("outputs/calibration/model_selected.json")
OUT = Path("outputs/calibration/dark_pixels")
LAB = ["L_star", "a_star", "b_star"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_model(MODEL)
    mapping = pd.read_csv(MAPPING)
    rows = []
    for img_name, treatment in zip(mapping.image, mapping.treatment):
        img = read_image_unicode(IMAGES / img_name)
        v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
        T, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = segment_chickpeas(img)
        inside = mask > 0
        dark = inside & (v <= T)
        bright_mask = np.where(inside & ~dark, 255, 0).astype(np.uint8)

        full = extract_color_values(img, mask)
        clean = extract_color_values(img, bright_mask)
        lab_full = np.array([full[c] for c in LAB])
        lab_clean = np.array([clean[c] for c in LAB])
        cal_full = apply_calibration(lab_full, model)
        cal_clean = apply_calibration(lab_clean, model)
        n_mask, n_dark = int(inside.sum()), int(dark.sum())
        rows.append({
            "image": img_name, "treatment": treatment, "otsu_V": T,
            "pixels_mask": n_mask, "pixels_dark": n_dark,
            "pct_dark": 100 * n_dark / n_mask,
            "L_cal_all": cal_full[0], "a_cal_all": cal_full[1], "b_cal_all": cal_full[2],
            "L_cal_without_dark": cal_clean[0], "a_cal_without_dark": cal_clean[1],
            "b_cal_without_dark": cal_clean[2],
            "dL": cal_clean[0] - cal_full[0], "da": cal_clean[1] - cal_full[1],
            "db": cal_clean[2] - cal_full[2],
            "dE_cal": float(delta_e76(cal_full, cal_clean)),
            "dE_uncal": float(delta_e76(lab_full, lab_clean)),
        })
    df = pd.DataFrame(rows)
    df["treatment"] = pd.Categorical(df.treatment, TREATMENT_ORDER, ordered=True)
    df = df.sort_values(["treatment", "image"])
    df.to_csv(OUT / "dark_pixels_per_photo.csv", index=False, encoding="utf-8-sig")

    g = df.groupby("treatment", observed=True)
    per_t = pd.DataFrame({
        "n_fotografias": g.size(),
        "pct_dark_medio": g.pct_dark.mean(),
        "pct_dark_max": g.pct_dark.max(),
        "dE_medio": g.dE_cal.mean(),
        "dE_max": g.dE_cal.max(),
        "foto_dE_max": g.apply(lambda d: d.loc[d.dE_cal.idxmax(), "image"], include_groups=False),
    })
    per_t.to_csv(OUT / "dark_pixels_max_by_treatment.csv", encoding="utf-8-sig")

    nice = df[["treatment", "image", "pct_dark", "dE_cal", "dL", "da", "db"]].rename(columns={
        "treatment": "Tratamento", "image": "Fotografia", "pct_dark": "Píxeis escuros (%)",
        "dE_cal": "ΔE* (calibrado)", "dL": "ΔL*", "da": "Δa*", "db": "Δb*"})
    with pd.ExcelWriter(OUT / "dark_pixels.xlsx") as xw:
        nice.round(2).to_excel(xw, sheet_name="Por fotografia", index=False)
        per_t.round(2).to_excel(xw, sheet_name="Máximo por tratamento")
        df.round(3).to_excel(xw, sheet_name="Dados completos", index=False)

    pd.set_option("display.width", 200)
    print(nice.round(2).to_string(index=False))
    print(per_t.round(2).to_string())
    print("Global: pct_dark max %.2f%%, dE max %.2f, dE médio %.2f"
          % (df.pct_dark.max(), df.dE_cal.max(), df.dE_cal.mean()))


if __name__ == "__main__":
    main()
