"""
Pantone calibration of the chickpea colour measurements.

Stages (all outputs in outputs/calibration/, original data untouched):
  1. measure every Pantone card (same Lab conversion as the grain script)
  2. match each card to the colorimeter reference by code
  3. fit candidate calibration models, validate by leave-one-out
  4. measure every grain photo (segment_chickpeas + extract_color_values,
     identical to src/main.py, each file counted once)
  5. apply the selected calibration to each photo, then summarise by treatment

Usage, from the repository root:
    python src/run_calibration.py
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colour.io import read_image_unicode  # noqa: E402
from colour.vision import segment_chickpeas, create_overlay_image  # noqa: E402
from colour.calibration import (  # noqa: E402
    MODELS, measure_card, measure_region, save_mask_check, fit_model,
    apply_calibration, loo_predictions, delta_e76, model_input, save_model,
)

HERE = Path(__file__).resolve().parent
COR = Path.home() / "OneDrive" / "Área de Trabalho" / "Trabalho" / "PhD" / "Broccoli" / "Cor"
DEFAULT_PANTONE = COR / "fotos_calibração_originais"
DEFAULT_REF = COR / "Calibration Curve Sophia.xlsx"
LAB = ["L_star", "a_star", "b_star"]
REF = ["L*", "a*", "b*"]
TREATMENT_ORDER = ["Soaked", "Im.1", "Im.2", "Im.3", "Boiled", "ImT.1", "ImT.2", "ImT.3",
                   "Sol.1", "Sol.2", "Sol.3", "Sol.4", "Sol.5", "Sol.6", "Sol.7"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
OVERFIT_RATIO = 1.5      # LOO dE / fit dE above this -> overfitting
MAX_GRAIN_SHIFT = 3.0    # dE; max change of a grain prediction when one card is dropped


def list_images(folder: Path):
    # one entry per file (on Windows *.jpg and *.JPG match the same files)
    return sorted({p.resolve() for p in folder.iterdir() if p.suffix.lower() in IMG_EXT})


# ---------------------------------------------------------------------------
def measure_pantone(folder, erode_frac, out):
    (out / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    for p in list_images(folder):
        code = p.stem.split("_", 1)[1]
        img = read_image_unicode(p)
        vals, inner, card = measure_card(img, erode_frac)
        if vals is None:
            print(f"!! segmentação falhou: {p.name}")
            rows.append({"file": p.name, "Code": code, "segmentation_ok": False})
            continue
        save_mask_check(img, inner, card, out / "masks" / f"{p.stem}_mask_check.jpg", code)
        vals.update(file=p.name, Code=code, segmentation_ok=True)
        rows.append(vals)
    meas = pd.DataFrame(rows)
    meas.to_csv(out / "pantone_measured.csv", index=False)
    return meas


def load_reference(path):
    ref = pd.read_excel(path)
    ref = ref[["Number", "Code"] + REF].dropna(subset=["Code"])  # '(Matlab)' columns ignored
    ref["Code"] = ref["Code"].astype(str).str.strip()
    dup = ref[ref.Code.duplicated(keep=False)]
    g = ref.groupby("Code")
    ref_mean = g[REF].mean().reset_index()
    ref_mean["n_reference_readings"] = g.size().values
    return ref_mean, dup


def measure_grains(img_dir, mapping, out):
    (out / "grain_masks").mkdir(parents=True, exist_ok=True)
    rows = []
    for p in list_images(img_dir):
        if p.name not in mapping.index:
            continue
        img = read_image_unicode(p)
        mask = segment_chickpeas(img)
        vals = measure_region(img, mask)
        vals.update(image=p.name, treatment=mapping.loc[p.name, "treatment"],
                    area_pixels=int((mask > 0).sum()))
        rows.append(vals)
        ov = cv2.cvtColor(create_overlay_image(img, mask), cv2.COLOR_RGB2BGR)
        ov = cv2.resize(ov, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
        cv2.imencode(".jpg", ov)[1].tofile(str(out / "grain_masks" / f"{p.stem}_overlay.jpg"))
    return pd.DataFrame(rows)


def lch(df):
    df = df.copy()
    df["C_star"] = np.hypot(df.a_star, df.b_star)
    df["h_deg"] = np.degrees(np.arctan2(df.b_star, df.a_star)) % 360
    return df


def summarise(df):
    """Mean, SD (n-1) and n per treatment; the unit is one photograph."""
    vars_ = ["L_star", "a_star", "b_star", "C_star", "h_deg"]
    g = df.groupby("treatment")
    tab = pd.DataFrame(index=[t for t in TREATMENT_ORDER if t in g.groups])
    tab["n"] = g.size()
    for v in vars_:
        tab[f"{v}_mean"] = g[v].mean()
        tab[f"{v}_sd"] = g[v].std(ddof=1)
    # hue: all hues lie far from 0/360, so the arithmetic mean is valid;
    # we still check it against the circular mean
    circ = g["h_deg"].apply(lambda h: np.degrees(np.arctan2(np.sin(np.radians(h)).mean(),
                                                            np.cos(np.radians(h)).mean())) % 360)
    assert np.allclose(circ.reindex(tab.index), tab.h_deg_mean, atol=0.5), "hue wraps 0/360"
    s = tab.loc["Soaked"]
    tab["dE_vs_Soaked"] = np.sqrt((tab.L_star_mean - s.L_star_mean) ** 2 +
                                  (tab.a_star_mean - s.a_star_mean) ** 2 +
                                  (tab.b_star_mean - s.b_star_mean) ** 2)
    tab["replicates"] = [replicate_type(t) for t in tab.index]
    tab.index.name = "treatment"
    return tab


def pretty(tab):
    out = pd.DataFrame(index=tab.index)
    out["n (fotografias)"] = tab["n"].astype(int)
    for v, name in [("L_star", "L*"), ("a_star", "a*"), ("b_star", "b*"),
                    ("C_star", "C*"), ("h_deg", "h° (0–360)")]:
        out[name] = [f"{m:.1f} ± {sd:.1f}" if pd.notna(sd) else f"{m:.1f}"
                     for m, sd in zip(tab[f"{v}_mean"], tab[f"{v}_sd"])]
    out["ΔE* vs Soaked"] = tab["dE_vs_Soaked"].round(1)
    out["DP refere-se a"] = [SD_MEANING[replicate_type(t)] for t in out.index]
    return out


# Solar-cooking treatments have no independent replicates: their photos are
# of the same batch, so their SD is within-batch only (no inferential tests).
SD_MEANING = {
    "replicas": "réplicas independentes (lotes)",
    "sem_replicas": "fotografias do mesmo lote (sem réplicas; DP só dentro do lote)",
}


def replicate_type(treatment: str) -> str:
    return "sem_replicas" if treatment.startswith("Sol.") else "replicas"


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pantone-dir", type=Path, default=DEFAULT_PANTONE)
    ap.add_argument("--reference", type=Path, default=DEFAULT_REF)
    ap.add_argument("--images", type=Path, default=Path("images"))
    ap.add_argument("--mapping", type=Path, default=HERE / "treatment_mapping.csv")
    ap.add_argument("--erode-frac", type=float, default=0.04)
    ap.add_argument("--out", type=Path, default=Path("outputs/calibration"))
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # 1-2. Pantone cards
    meas = measure_pantone(args.pantone_dir, args.erode_frac, out)
    ref, dup = load_reference(args.reference)
    ok = meas[meas.segmentation_ok]
    merged = ok.merge(ref, on="Code", how="inner")
    merged.to_csv(out / "pantone_matched.csv", index=False)
    unmatched = {"foto_sem_referencia": sorted(set(ok.Code) - set(ref.Code)),
                 "referencia_sem_foto": sorted(set(ref.Code) - set(meas.Code)),
                 "segmentacao_falhou": meas.loc[~meas.segmentation_ok, "file"].tolist()}

    # 4. grains (needed now for the stability check of the models)
    mapping = pd.read_csv(args.mapping).set_index("image")
    grains = measure_grains(args.images, mapping, out)
    missing = sorted(set(mapping.index) - set(grains.image))

    # 3. models
    Y = merged[REF].to_numpy(float)
    res, per_card, stab, models = [], merged[["Code", "file"] + LAB + REF].copy(), {}, {}
    per_card["dE_before"] = delta_e76(merged[LAB].to_numpy(float), Y)
    for kind, (space, desc) in MODELS.items():
        X = model_input(merged, space)
        Xg = model_input(grains, space)
        m = fit_model(X, Y, kind)
        models[kind] = m
        fit_err = delta_e76(apply_calibration(X, m), Y)
        cv_pred, g_folds = loo_predictions(X, Y, kind, X_extra=Xg)
        cv_err = delta_e76(cv_pred, Y)
        g_full = apply_calibration(Xg, m)
        # how much do grain predictions move when one card is dropped?
        g_shift = delta_e76(g_folds, g_full[None]).max(axis=0)
        per_card[f"dE_fit_{kind}"] = fit_err
        per_card[f"dE_loo_{kind}"] = cv_err
        res.append({"model": kind, "description": desc,
                    "n_params": int(np.asarray(m["coef"]).size if kind != "lab_linear_per_channel" else 6),
                    "dE_fit_mean": fit_err.mean(), "dE_fit_max": fit_err.max(),
                    "dE_loo_mean": cv_err.mean(), "dE_loo_median": np.median(cv_err),
                    "dE_loo_max": cv_err.max(), "card_loo_max": merged.Code.iloc[cv_err.argmax()],
                    "loo_over_fit": cv_err.mean() / fit_err.mean(),
                    "grain_max_shift_dE_mean": g_shift.mean(),
                    "grain_max_shift_dE_max": g_shift.max()})
        save_model(m, out / f"model_{kind}.json")
    res = pd.DataFrame(res)
    before = per_card["dE_before"]
    res.loc[-1] = {"model": "sem_calibracao", "description": "valores medidos (sem calibração)",
                   "n_params": 0, "dE_fit_mean": before.mean(), "dE_fit_max": before.max(),
                   "dE_loo_mean": before.mean(), "dE_loo_median": before.median(),
                   "dE_loo_max": before.max(), "card_loo_max": merged.Code.iloc[before.argmax()]}
    res = res.sort_index().reset_index(drop=True)
    res.to_csv(out / "calibration_model_comparison.csv", index=False)
    per_card.to_csv(out / "pantone_errors_per_card.csv", index=False)

    # Selection: lowest LOO mean dE among models WITHOUT signs of overfitting.
    #   - LOO error more than 1.5x the fitting error, or
    #   - predicted grain colour moving > 3 dE when a single card is removed.
    cand = res[res.model != "sem_calibracao"].copy()
    cand["overfit_flag"] = (cand.loo_over_fit > OVERFIT_RATIO) | \
                           (cand.grain_max_shift_dE_max > MAX_GRAIN_SHIFT)
    res = res.merge(cand[["model", "overfit_flag"]], on="model", how="left")
    res.to_csv(out / "calibration_model_comparison.csv", index=False)
    best_unconstrained = cand.loc[cand.dE_loo_mean.idxmin(), "model"]
    ok_models = cand[~cand.overfit_flag]
    best = ok_models.loc[ok_models.dE_loo_mean.idxmin(), "model"]
    selected = models[best]
    save_model(selected, out / "model_selected.json")

    # 5. apply to each photo, then summarise
    grains = lch(grains)
    grains.to_csv(out / "grain_measurements_uncalibrated.csv", index=False)
    cal = grains.copy()
    cal[LAB] = apply_calibration(grains, selected)
    cal = lch(cal)
    cal["model"] = best
    cal.to_csv(out / "grain_measurements_calibrated.csv", index=False)

    tab_cal, tab_unc = summarise(cal), summarise(grains)

    # sensitivity: treatment means under every candidate model
    sens = []
    for kind, m in models.items():
        c = grains.copy()
        c[LAB] = apply_calibration(grains, m)
        t = summarise(lch(c))
        t["model"] = kind
        sens.append(t.reset_index())
    sens = pd.concat(sens)
    sens.round(2).to_csv(out / "sensitivity_treatment_means_by_model.csv", index=False)
    dsel = sens[sens.model == best].set_index("treatment")
    sens["dE_vs_selected_model"] = [
        delta_e76(r[["L_star_mean", "a_star_mean", "b_star_mean"]].to_numpy(float),
                  dsel.loc[r.treatment, ["L_star_mean", "a_star_mean", "b_star_mean"]].to_numpy(float))
        for _, r in sens.iterrows()]
    sens.groupby("model").dE_vs_selected_model.agg(["mean", "max"]).round(2).to_csv(
        out / "sensitivity_vs_selected_model.csv")
    tab_cal.round(1).to_csv(out / "treatment_summary_calibrated_numeric.csv")
    tab_unc.round(1).to_csv(out / "treatment_summary_uncalibrated_numeric.csv")
    p_cal, p_unc = pretty(tab_cal), pretty(tab_unc)
    p_cal.to_csv(out / "treatment_summary_calibrated.csv", encoding="utf-8-sig")
    p_unc.to_csv(out / "treatment_summary_uncalibrated.csv", encoding="utf-8-sig")
    with pd.ExcelWriter(out / "treatment_summary.xlsx") as xw:
        p_cal.to_excel(xw, sheet_name="Calibrado")
        p_unc.to_excel(xw, sheet_name="Nao calibrado")
        tab_cal.round(1).to_excel(xw, sheet_name="Calibrado (numerico)")
        tab_unc.round(1).to_excel(xw, sheet_name="Nao calibrado (numerico)")
        res.round(2).to_excel(xw, sheet_name="Modelos", index=False)

    pd.Series({"selected_model": best, "lowest_loo_without_overfit_check": best_unconstrained, **{k: "; ".join(v) for k, v in unmatched.items()},
               "mapping_images_not_found": "; ".join(missing),
               "duplicated_reference_codes": "; ".join(sorted(set(dup.Code)))}
              ).to_csv(out / "run_info.csv", header=False)

    pd.set_option("display.width", 250)
    print(f"Cartões: {len(meas)} fotografados, {len(merged)} usados. Sem correspondência: {unmatched}")
    print("Referências repetidas (média usada):\n", dup.to_string(index=False))
    print(res.drop(columns="description").round(2).to_string(index=False))
    print("Escolhido:", best, "| menor LOO sem verificação de sobreajuste:", best_unconstrained)
    print(sens.groupby("model").dE_vs_selected_model.agg(["mean", "max"]).round(2))
    print("Fotografias do mapeamento não encontradas:", missing)
    print(p_cal.to_string()); print(p_unc.to_string())


if __name__ == "__main__":
    main()
