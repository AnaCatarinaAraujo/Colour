"""
Colour calibration with Pantone cards.

Measured values use exactly the same conversion as the grain analysis
(``colour.core.extract_color_values``: OpenCV BGR2LAB on the 8-bit image,
then L*100/255, a-128, b-128). In addition the mean *linear* RGB of the
same pixels is stored (sRGB transfer function removed per pixel), which is
the input of the RGB->XYZ models.

Calibration models map a measurement to the colorimeter reference L*a*b*.
Two families are available:

* ``space='lab'``: input = measured L*a*b*; output = L*a*b*
* ``space='rgb'``: input = mean linear RGB; output = XYZ, then converted
  to L*a*b* with the reference white ``WHITE``.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .core import extract_color_values

# D65 / 2 deg reference white (normalised Y = 1). Only used by the RGB->XYZ
# models to move between XYZ and L*a*b*; the same white is used in both
# directions so the fit stays in the colorimeter's L*a*b* space.
WHITE = np.array([0.95047, 1.0, 1.08883])

_SRGB_LUT = np.array([
    (v / 255) / 12.92 if v / 255 <= 0.04045 else ((v / 255 + 0.055) / 1.055) ** 2.4
    for v in range(256)
])


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------
def mean_linear_rgb(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    px = img_bgr[mask > 0]  # BGR uint8
    lin = _SRGB_LUT[px]
    return lin.mean(axis=0)[::-1]  # -> R, G, B


def measure_region(img_bgr: np.ndarray, mask: np.ndarray) -> dict | None:
    """Lab exactly as in the grain script + mean linear RGB."""
    vals = extract_color_values(img_bgr, mask)
    if vals is None:
        return None
    vals["Rlin"], vals["Glin"], vals["Blin"] = mean_linear_rgb(img_bgr, mask)
    return vals


def segment_card(img_bgr: np.ndarray, erode_frac: float = 0.04):
    """Return (inner_mask, full_card_mask) for the Pantone card.

    1. Otsu threshold on the blurred V channel (card is brighter than the
       black cloth background).
    2. Keep the large connected component nearest the image centre
       (excludes other cards peeking in at the image borders).
    3. Fill holes, then erode by ``erode_frac`` x the card's shorter side so
       the edge, shadows and fibres at the border are excluded.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = cv2.GaussianBlur(hsv[:, :, 2], (0, 0), 3)
    _, mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    h, w = mask.shape
    centre = np.array([w / 2, h / 2])
    best, best_d = None, np.inf
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 0.005 * h * w:
            continue
        d = np.linalg.norm(cents[i] - centre)
        if d < best_d:
            best, best_d = i, d
    if best is None:
        return None, None

    card = np.zeros_like(mask)
    card[labels == best] = 255
    contours, _ = cv2.findContours(card, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    card = np.zeros_like(mask)
    cv2.drawContours(card, contours, -1, 255, thickness=cv2.FILLED)

    short_side = min(stats[best, cv2.CC_STAT_WIDTH], stats[best, cv2.CC_STAT_HEIGHT])
    r = max(3, int(round(erode_frac * short_side)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.erode(card, kernel), card


def save_mask_check(img_bgr, inner, card, out_path: Path, title: str = ""):
    """Crop around the card; outer contour in red, measured region in green."""
    ys, xs = np.where(card > 0)
    pad = 80
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, card.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, card.shape[1])
    vis = img_bgr.copy()
    for m, col in ((card, (0, 0, 255)), (inner, (0, 255, 0))):
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cs, -1, col, 6)
    crop = vis[y0:y1, x0:x1]
    scale = 800 / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if title:
        cv2.putText(crop, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2)
    ok, enc = cv2.imencode(out_path.suffix, crop)
    if ok:
        enc.tofile(str(out_path))


def measure_card(img_bgr, erode_frac: float = 0.04):
    inner, card = segment_card(img_bgr, erode_frac)
    if inner is None or inner.sum() == 0:
        return None, None, None
    vals = measure_region(img_bgr, inner)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[inner > 0].astype(float)
    lab[:, 0] *= 100 / 255
    lab[:, 1:] -= 128
    vals["L_sd_px"], vals["a_sd_px"], vals["b_sd_px"] = lab.std(axis=0)
    vals["area_pixels"] = int((inner > 0).sum())
    vals["card_area_pixels"] = int((card > 0).sum())
    return vals, inner, card


# --------------------------------------------------------------------------
# Colour maths
# --------------------------------------------------------------------------
def _f(t):
    d = 6 / 29
    return np.where(t > d ** 3, np.cbrt(t), t / (3 * d ** 2) + 4 / 29)


def _finv(t):
    d = 6 / 29
    return np.where(t > d, t ** 3, 3 * d ** 2 * (t - 4 / 29))


def xyz_to_lab(xyz, white=WHITE):
    f = _f(np.asarray(xyz, float) / white)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def lab_to_xyz(lab, white=WHITE):
    lab = np.asarray(lab, float)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    return np.stack([_finv(fx), _finv(fy), _finv(fz)], axis=-1) * white


def delta_e76(a, b) -> np.ndarray:
    return np.linalg.norm(np.asarray(a, float) - np.asarray(b, float), axis=-1)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
# name -> (input space, description)
MODELS = {
    "lab_linear_per_channel": ("lab", "L*a*b*: ref_c = p1*meas_c + p0, por canal (6 par.)"),
    "lab_affine": ("lab", "L*a*b*: afim 3x3 + termo independente (12 par.)"),
    "lab_affine_L2": ("lab", "L*a*b*: afim + L*^2 (curva de tom em L*) (15 par.)"),
    "lab_affine_L2_L3": ("lab", "L*a*b*: afim + L*^2 + L*^3 (18 par.)"),
    "lab_poly2": ("lab", "L*a*b*: polinómio completo de 2.ª ordem (30 par.)"),
    "rgb_linear": ("rgb", "RGB linear -> XYZ: matriz 3x3 (9 par.)"),
    "rgb_root_poly2": ("rgb", "RGB linear -> XYZ: root-polynomial 2.ª ordem (18 par.)"),
    "rgb_poly2": ("rgb", "RGB linear -> XYZ: polinómio 2.ª ordem + termo indep. (30 par.)"),
}


def _features(X: np.ndarray, kind: str) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, float))
    one = np.ones(len(X))
    c0, c1, c2 = X[:, 0], X[:, 1], X[:, 2]
    if kind in ("lab_affine", "lab_linear_per_channel"):
        return np.column_stack([c0, c1, c2, one])
    if kind == "lab_affine_L2":
        Ls = c0 / 100
        return np.column_stack([c0, c1, c2, 100 * Ls ** 2, one])
    if kind == "lab_affine_L2_L3":
        Ls = c0 / 100
        return np.column_stack([c0, c1, c2, 100 * Ls ** 2, 100 * Ls ** 3, one])
    if kind in ("lab_poly2", "rgb_poly2"):
        s = 100.0 if kind == "lab_poly2" else 1.0
        return np.column_stack([c0, c1, c2, c0 ** 2 / s, c1 ** 2 / s, c2 ** 2 / s,
                                c0 * c1 / s, c0 * c2 / s, c1 * c2 / s, one])
    if kind == "rgb_linear":
        return np.column_stack([c0, c1, c2])
    if kind == "rgb_root_poly2":
        return np.column_stack([c0, c1, c2, np.sqrt(c0 * c1), np.sqrt(c1 * c2),
                                np.sqrt(c0 * c2)])
    raise ValueError(kind)


def model_input(meas, space: str) -> np.ndarray:
    """Extract model input from a DataFrame/dict of measurements."""
    cols = ["L_star", "a_star", "b_star"] if space == "lab" else ["Rlin", "Glin", "Blin"]
    if hasattr(meas, "loc"):
        return meas[cols].to_numpy(float)
    return np.array([meas[c] for c in cols], float)


def fit_model(X: np.ndarray, Y_lab_ref: np.ndarray, kind: str) -> dict:
    space = MODELS[kind][0]
    X, Y = np.asarray(X, float), np.asarray(Y_lab_ref, float)
    target = Y if space == "lab" else lab_to_xyz(Y)
    if kind == "lab_linear_per_channel":
        coef = np.zeros((4, 3))
        for c in range(3):
            A = np.column_stack([X[:, c], np.ones(len(X))])
            (p1, p0), *_ = np.linalg.lstsq(A, target[:, c], rcond=None)
            coef[c, c], coef[3, c] = p1, p0
    else:
        coef, *_ = np.linalg.lstsq(_features(X, kind), target, rcond=None)
    return {"kind": kind, "space": space, "description": MODELS[kind][1],
            "white_XYZ": WHITE.tolist(), "coef": coef.tolist()}


def apply_calibration(x, model: dict) -> np.ndarray:
    """Apply a calibration model to ONE or MORE individual measurements.

    ``x`` can be
      * a pandas DataFrame / dict with the output columns of
        ``measure_region`` (L_star, a_star, b_star, Rlin, Glin, Blin), or
      * an array (3,) or (n, 3) already in the model's input space
        (measured L*a*b* for 'lab' models, mean linear RGB for 'rgb' models).
    Returns calibrated L*a*b* with shape (3,) or (n, 3).
    """
    if hasattr(x, "loc") or isinstance(x, dict):
        x = model_input(x, model["space"])
    x = np.asarray(x, float)
    single = x.ndim == 1
    out = _features(x, model["kind"]) @ np.asarray(model["coef"])
    if model["space"] == "rgb":
        out = xyz_to_lab(out, np.asarray(model.get("white_XYZ", WHITE)))
    return out[0] if single else out


def loo_predictions(X, Y, kind: str, X_extra=None):
    """Leave-one-out predictions. If X_extra is given, also returns the
    predictions of every fold's model on X_extra (shape folds x n x 3) to
    check how stable the model is outside the cards."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    pred = np.zeros_like(Y)
    extra = []
    for i in range(len(X)):
        keep = np.arange(len(X)) != i
        m = fit_model(X[keep], Y[keep], kind)
        pred[i] = apply_calibration(X[i], m)
        if X_extra is not None:
            extra.append(apply_calibration(X_extra, m))
    return (pred, np.array(extra)) if X_extra is not None else pred


def save_model(model: dict, path: Path):
    Path(path).write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")


def load_model(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
