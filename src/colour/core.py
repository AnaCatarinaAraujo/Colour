import numpy as np
import cv2

def lab_to_lch(L, a, b):
    C = np.sqrt(a**2 + b**2)
    h = np.degrees(np.arctan2(b, a))

    if h < 0:
        h += 360

    return L, C, h

def extract_color_values(img_bgr, mask):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    pixels_rgb = img_rgb[mask > 0]
    pixels_hsv = img_hsv[mask > 0]
    pixels_lab = img_lab[mask > 0]

    if len(pixels_rgb) == 0:
        return None

    R = pixels_rgb[:, 0].mean()
    G = pixels_rgb[:, 1].mean()
    B = pixels_rgb[:, 2].mean()

    H = pixels_hsv[:, 0].mean() * 2
    S = pixels_hsv[:, 1].mean()
    V = pixels_hsv[:, 2].mean()

    L = pixels_lab[:, 0].mean() * 100 / 255
    a = pixels_lab[:, 1].mean() - 128
    b = pixels_lab[:, 2].mean() - 128

    L_value, C, h = lab_to_lch(L, a, b)

    return {
        "R_mean": R,
        "G_mean": G,
        "B_mean": B,
        "H_mean_deg": H,
        "S_mean": S,
        "V_mean": V,
        "L_star": L_value,
        "a_star": a,
        "b_star": b,
        "C_star": C,
        "h_deg": h
    }
