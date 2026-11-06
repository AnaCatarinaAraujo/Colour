"""
Figure 5: representative grain photographs and the segmentation used in the
colour analysis.

* One photo per sample: the one closest (CIE76 dE) to its treatment mean in
  calibrated L*a*b* (outputs/calibration/grain_measurements_calibrated.csv).
* Photos are shown WITHOUT colour calibration (illustrative only).
* The zoom (focal length) changed between sessions (24-49 mm), so every photo
  is resampled by f_ref / f to a common magnification (assumes the same
  camera-to-sample distance), then cropped with the same size, centred on
  the grain. Masks: segment_chickpeas() exactly as in the colour analysis.

Layout (chosen): one row per sample (original | mask | contour overlay).
Alternative layout B (originals on top, overlays below) with --layout-b.

Usage (from the repository root):  python src/make_figure5.py [--layout-b]
"""
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colour.io import read_image_unicode  # noqa: E402
from colour.vision import segment_chickpeas  # noqa: E402

SAMPLES = ["Soaked", "Boiled", "Sol.1", "Sol.6", "ImT.2", "Sol.3"]
IMAGES = Path("images")
CAL = Path("outputs/calibration/grain_measurements_calibrated.csv")
OUT = Path("outputs/figures")
CONTOUR_RGB = (0, 230, 255)   # cyan contour, visible on orange grain / black cloth
MARGIN = 1.08                 # crop = largest grain-pile extent x margin
MM = 1 / 25.4

plt.rcParams.update({"font.family": "Arial", "font.size": 10,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def focal_length(path: Path) -> float:
    return float(Image.open(path).getexif().get_ifd(0x8769).get(0x920A))


def pick_representatives():
    c = pd.read_csv(CAL)
    rows = []
    for t in SAMPLES:
        d = c[c.treatment == t].copy()
        mean = d[["L_star", "a_star", "b_star"]].mean()
        d["dE_to_mean"] = np.linalg.norm(d[["L_star", "a_star", "b_star"]] - mean, axis=1)
        best = d.sort_values("dE_to_mean").iloc[0]
        rows.append({"sample": t, "image": best.image, "dE_to_mean": best.dE_to_mean,
                     "n_photos_in_treatment": len(d),
                     "treatment_mean_Lab_calibrated": "L*={:.1f}, a*={:.1f}, b*={:.1f}".format(*mean),
                     "photo_Lab_calibrated": "L*={:.1f}, a*={:.1f}, b*={:.1f}".format(
                         best.L_star, best.a_star, best.b_star)})
    return pd.DataFrame(rows)


def load(sel):
    """Read, segment, rescale to common magnification."""
    f_ref = min(focal_length(IMAGES / i) for i in sel.image)  # downscale only
    items = []
    for _, r in sel.iterrows():
        p = IMAGES / r.image
        bgr = read_image_unicode(p)
        mask = segment_chickpeas(bgr)
        f = focal_length(p)
        s = f_ref / f
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if s != 1:
            rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
            mask = np.where(mask >= 128, 255, 0).astype(np.uint8)
        # centre on the grain pile: bounding box of the largest component(s)
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
        big = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 0.01 * st[1:, 4].max()]
        x0 = min(st[i, 0] for i in big); y0 = min(st[i, 1] for i in big)
        x1 = max(st[i, 0] + st[i, 2] for i in big); y1 = max(st[i, 1] + st[i, 3] for i in big)
        items.append({**r.to_dict(), "focal_mm": f, "scale": s, "rgb": rgb, "mask": mask,
                      "centre": ((x0 + x1) / 2, (y0 + y1) / 2), "extent": max(x1 - x0, y1 - y0)})
    return items, f_ref


def crop(items):
    size = int(np.ceil(max(it["extent"] for it in items) * MARGIN))
    size = min(size, min(min(it["rgb"].shape[:2]) for it in items))
    for it in items:
        h, w = it["mask"].shape
        cx, cy = it["centre"]
        x0 = int(np.clip(round(cx - size / 2), 0, w - size))
        y0 = int(np.clip(round(cy - size / 2), 0, h - size))
        it["rgb_c"] = it["rgb"][y0:y0 + size, x0:x0 + size]
        it["mask_c"] = it["mask"][y0:y0 + size, x0:x0 + size]
        ov = it["rgb_c"].copy()
        cs, _ = cv2.findContours(it["mask_c"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, cs, -1, CONTOUR_RGB, thickness=max(2, size // 350),
                         lineType=cv2.LINE_AA)
        it["overlay_c"] = ov
        it["crop_xywh"] = (x0, y0, size, size)
    return size


def panel(ax, img, letter, cmap=None):
    ax.imshow(img, cmap=cmap, vmin=0, vmax=255, interpolation="antialiased")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0.03, 0.97, f"({letter})", transform=ax.transAxes, ha="left", va="top",
            color="white", fontsize=10)


def letters():
    return iter("abcdefghijklmnopqrstuvwxyz")


def layout_a(items):
    cols = ["Original", "Segmentation mask", "Mask contour"]
    pw, gap, left, top = 38 * MM, 1.2 * MM, 12 * MM, 6 * MM
    W = left + 3 * pw + 2 * gap + 1 * MM
    H = top + len(items) * pw + (len(items) - 1) * gap + 1 * MM
    fig = plt.figure(figsize=(W, H), facecolor="white")
    L = letters()
    for r, it in enumerate(items):
        for c, key in enumerate(["rgb_c", "mask_c", "overlay_c"]):
            x = (left + c * (pw + gap)) / W
            y = 1 - (top + r * (pw + gap) + pw) / H
            ax = fig.add_axes([x, y, pw / W, pw / H])
            panel(ax, it[key], next(L), cmap="gray" if key == "mask_c" else None)
            if r == 0:
                ax.set_title(cols[c], fontsize=10, pad=3)
            if c == 0:
                ax.set_ylabel(it["sample"], fontsize=10, labelpad=4)
    return fig


def layout_b(items):
    pw, gap, left, top = 28 * MM, 1.0 * MM, 13 * MM, 6 * MM
    n = len(items)
    W = left + n * pw + (n - 1) * gap + 1 * MM
    H = top + 2 * pw + gap + 1 * MM
    fig = plt.figure(figsize=(W, H), facecolor="white")
    L = letters()
    for r, (key, rowname) in enumerate([("rgb_c", "Original"), ("overlay_c", "Mask contour")]):
        for c, it in enumerate(items):
            x = (left + c * (pw + gap)) / W
            y = 1 - (top + r * (pw + gap) + pw) / H
            ax = fig.add_axes([x, y, pw / W, pw / H])
            panel(ax, it[key], next(L))
            if r == 0:
                ax.set_title(it["sample"], fontsize=10, pad=3)
            if c == 0:
                ax.set_ylabel(rowname, fontsize=10, labelpad=4)
    return fig


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUT / f"{stem}.pdf", dpi=600, facecolor="white")
    # preview: <= 1500 px wide, < 1 MB
    w_in = fig.get_size_inches()[0]
    fig.savefig(OUT / f"{stem}_preview.png", dpi=min(300, 1500 / w_in), facecolor="white")
    p = OUT / f"{stem}_preview.png"
    q = 300
    while p.stat().st_size >= 1_000_000 and q > 50:
        q -= 25
        fig.savefig(p, dpi=min(q, 1500 / w_in), facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sel = pick_representatives()
    items, f_ref = load(sel)
    size = crop(items)
    save(layout_a(items), "Figure5_images")
    if "--layout-b" in sys.argv:
        save(layout_b(items), "Figure5_images_layoutB")

    info = pd.DataFrame([{k: it[k] for k in ["sample", "image", "dE_to_mean", "photo_Lab_calibrated",
                                             "treatment_mean_Lab_calibrated", "n_photos_in_treatment",
                                             "focal_mm", "scale", "crop_xywh"]} for it in items])
    la = {it["sample"]: "({}) ({}) ({})".format(*"abcdefghijklmnopqr"[3 * i:3 * i + 3])
          for i, it in enumerate(items)}
    lines = [
        "Figure 5 - fotografias usadas em cada painel",
        "",
        "Escolha: em cada tratamento, a fotografia com menor dE*ab (CIE76) em relação à média do",
        "tratamento em L*a*b* calibrado. As imagens mostradas NÃO estão calibradas em cor",
        "(são ilustrativas); a calibração só se aplica aos valores numéricos.",
        f"Escala comum: cada foto reamostrada por f_ref/f (f_ref = {f_ref:.0f} mm, a menor distância",
        "focal entre as fotos escolhidas), assumindo a mesma distância câmara-amostra.",
        f"Recorte: {size} x {size} px (após reamostragem), centrado no grão, igual em todos os painéis.",
        "Máscara: segment_chickpeas() (Otsu no canal V + abertura/fecho 5x5), a mesma da análise de cor.",
        "Sem barra de escala: as fotografias não têm uma referência de tamanho conhecida.",
        "",
    ]
    for _, r in info.iterrows():
        lines += [f"{r['sample']}: {r.image}",
                  f"    painéis: {la[r['sample']]} (original, máscara, contorno)",
                  f"    dE* em relação à média do tratamento: {r.dE_to_mean:.1f} "
                  f"(n = {r.n_photos_in_treatment} fotografias)",
                  f"    foto: {r.photo_Lab_calibrated}; média do tratamento: {r.treatment_mean_Lab_calibrated}",
                  f"    distância focal: {r.focal_mm:.0f} mm; fator de escala: {r.scale:.3f}", ""]
    (OUT / "Figure5_images_panels.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
