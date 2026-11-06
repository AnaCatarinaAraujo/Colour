"""
Statistics between treatments on the per-photograph colour values.

For each variable (L*, a*, b*, C*, h°):
  * Levene test (homogeneity of variances, centre = mean)
  * Welch's ANOVA (does not assume equal variances)
  * Games-Howell post-hoc (unequal variances and unequal n)
  * compact letter display from the Games-Howell p-values

Only treatments WITH independent replicates are tested (Soaked, Im.1-3,
Boiled, ImT.1-3). Solar-cooking treatments (Sol.1-Sol.7) have no independent
replicates (photos of the same batch) and are marked with † instead of letters.

Usage (from the repository root, after run_calibration.py):
    python src/stats_treatments.py
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_calibration import TREATMENT_ORDER, replicate_type  # noqa: E402

VARS = [("L_star", "L*"), ("a_star", "a*"), ("b_star", "b*"),
        ("C_star", "C*"), ("h_deg", "h° (0–360)")]
ALPHA = 0.05
DAGGER = "†"
FOOTNOTE = ("Médias na mesma coluna com letras diferentes são significativamente "
            "diferentes (ANOVA de Welch + Games-Howell, α = 0,05; unidade = fotografia). "
            f"{DAGGER} Cozedura solar: sem réplicas independentes (fotografias do mesmo "
            "lote); DP apenas dentro do lote; não incluído nos testes estatísticos.")


def welch_anova(groups):
    n = np.array([len(g) for g in groups], float)
    m = np.array([np.mean(g) for g in groups])
    v = np.array([np.var(g, ddof=1) for g in groups])
    k = len(groups)
    w = n / v
    mw = np.sum(w * m) / np.sum(w)
    a = np.sum(w * (m - mw) ** 2) / (k - 1)
    tmp = np.sum((1 - w / np.sum(w)) ** 2 / (n - 1))
    b = 1 + 2 * (k - 2) / (k ** 2 - 1) * tmp
    f = a / b
    df1, df2 = k - 1, (k ** 2 - 1) / (3 * tmp)
    return f, df1, df2, stats.f.sf(f, df1, df2)


def games_howell(data: dict):
    names = list(data)
    k = len(names)
    rows = []
    for a, b in itertools.combinations(names, 2):
        xa, xb = np.asarray(data[a], float), np.asarray(data[b], float)
        na, nb = len(xa), len(xb)
        va, vb = xa.var(ddof=1) / na, xb.var(ddof=1) / nb
        diff = xa.mean() - xb.mean()
        se = np.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
        q = abs(diff) / se * np.sqrt(2)
        p = stats.studentized_range.sf(q, k, df)
        rows.append({"A": a, "B": b, "mean_A": xa.mean(), "mean_B": xb.mean(),
                     "diff": diff, "se": se, "df": df, "q": q, "p": min(p, 1.0)})
    return pd.DataFrame(rows)


def compact_letters(means: pd.Series, pairs: pd.DataFrame, alpha=ALPHA) -> dict:
    """Insert-absorb algorithm (Piepho 2004). Letters ordered by decreasing mean."""
    order = list(means.sort_values(ascending=False).index)
    cols = [set(order)]
    for _, r in pairs[pairs.p < alpha].iterrows():
        new = []
        for c in cols:
            if r.A in c and r.B in c:
                new += [c - {r.A}, c - {r.B}]
            else:
                new.append(c)
        # absorb: drop columns contained in another column
        new = [c for i, c in enumerate(new)
               if not any((c < d) or (c == d and j < i) for j, d in enumerate(new) if j != i)]
        cols = new
    # name the columns in order of the first (highest-mean) group they contain
    cols.sort(key=lambda c: min(order.index(g) for g in c))
    letters = {g: "" for g in order}
    for i, c in enumerate(cols):
        for g in c:
            letters[g] += "abcdefghijklmnopqrstuvwxyz"[i]
    return {g: "".join(sorted(s)) for g, s in letters.items()}


def analyse(df: pd.DataFrame, label: str):
    tested = [t for t in TREATMENT_ORDER
              if t in set(df.treatment) and replicate_type(t) == "replicas"]
    summary, posthoc, letters_all = [], [], {}
    for var, name in VARS:
        data = {t: df.loc[df.treatment == t, var].to_numpy(float) for t in tested}
        lev = stats.levene(*data.values(), center="mean")
        f, df1, df2, p_w = welch_anova(list(data.values()))
        gh = games_howell(data)
        gh.insert(0, "variable", name)
        means = pd.Series({t: d.mean() for t, d in data.items()})
        if p_w < ALPHA:
            letters = compact_letters(means, gh)
        else:  # global test not significant -> no differences reported
            letters = {t: "a" for t in tested}
        letters_all[var] = letters
        summary.append({"dados": label, "variável": name, "k": len(tested),
                        "N": sum(len(d) for d in data.values()),
                        "Levene_W": lev.statistic, "Levene_p": lev.pvalue,
                        "variâncias_homogéneas": lev.pvalue >= ALPHA,
                        "Welch_F": f, "Welch_df1": df1, "Welch_df2": df2, "Welch_p": p_w,
                        "pares_signif_GH": int((gh.p < ALPHA).sum()) if p_w < ALPHA else 0})
        posthoc.append(gh)
    return pd.DataFrame(summary), pd.concat(posthoc), letters_all


def table_with_letters(tab_num: pd.DataFrame, letters_all: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=tab_num.index)
    out["n (fotografias)"] = tab_num["n"].astype(int)
    for var, name in VARS:
        col = []
        for t in tab_num.index:
            m, sd = tab_num.loc[t, f"{var}_mean"], tab_num.loc[t, f"{var}_sd"]
            mark = letters_all[var].get(t, DAGGER) if replicate_type(t) == "replicas" else DAGGER
            col.append(f"{m:.1f} ± {sd:.1f} {mark}")
        out[name] = col
    out["ΔE* vs Soaked"] = tab_num["dE_vs_Soaked"].round(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("outputs/calibration"))
    args = ap.parse_args()
    d = args.dir
    out = d / "stats"
    out.mkdir(exist_ok=True)

    sheets = {}
    all_summary, all_posthoc = [], []
    for label, meas, num in [("calibrado", "grain_measurements_calibrated.csv",
                              "treatment_summary_calibrated_numeric.csv"),
                             ("nao_calibrado", "grain_measurements_uncalibrated.csv",
                              "treatment_summary_uncalibrated_numeric.csv")]:
        df = pd.read_csv(d / meas)
        tab_num = pd.read_csv(d / num, index_col=0)
        s, ph, letters = analyse(df, label)
        ph.insert(0, "dados", label)
        all_summary.append(s)
        all_posthoc.append(ph)
        t = table_with_letters(tab_num, letters)
        t.to_csv(out / f"treatment_summary_{label}_letters.csv", encoding="utf-8-sig")
        sheets[f"Tabela {label}"] = t

    summary = pd.concat(all_summary)
    posthoc = pd.concat(all_posthoc)
    summary.to_csv(out / "levene_welch.csv", index=False, encoding="utf-8-sig")
    posthoc.to_csv(out / "games_howell.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out / "treatment_stats.xlsx") as xw:
        for name, t in sheets.items():
            t.to_excel(xw, sheet_name=name)
            pd.DataFrame({"Nota": [FOOTNOTE]}).to_excel(
                xw, sheet_name=name, startrow=len(t) + 2, index=False)
        summary.round(4).to_excel(xw, sheet_name="Levene e Welch", index=False)
        posthoc.round(4).to_excel(xw, sheet_name="Games-Howell", index=False)
    (out / "NOTA.txt").write_text(FOOTNOTE + "\n", encoding="utf-8")

    pd.set_option("display.width", 250)
    print(summary.round(4).to_string(index=False))
    print(sheets["Tabela calibrado"].to_string())
    print(FOOTNOTE)


if __name__ == "__main__":
    main()
