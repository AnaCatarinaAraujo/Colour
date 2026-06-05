import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import Tk, filedialog


# ============================================================
# ESCOLHER PASTAS COM JANELA
# ============================================================

Tk().withdraw()

input_folder = Path(
    filedialog.askdirectory(title="Escolhe a pasta onde estão as imagens")
)

output_folder = Path(
    filedialog.askdirectory(title="Escolhe a pasta onde queres guardar os resultados")
)

if not input_folder:
    raise ValueError("Não escolheste a pasta das imagens.")

if not output_folder:
    raise ValueError("Não escolheste a pasta dos resultados.")

output_folder.mkdir(exist_ok=True)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

image_extensions = [
    "*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff",
    "*.JPG", "*.JPEG", "*.PNG", "*.TIF", "*.TIFF"
]

min_object_area = 300

preview_each_image = True
ask_acceptance = False


# ============================================================
# FUNÇÕES PARA LER/GUARDAR IMAGENS COM ACENTOS NO CAMINHO
# ============================================================

def read_image_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def write_image_unicode(path, img):
    ext = Path(path).suffix
    success, encoded_img = cv2.imencode(ext, img)

    if success:
        encoded_img.tofile(str(path))
    else:
        print(f"Não foi possível guardar: {path}")


# ============================================================
# FUNÇÕES DE ANÁLISE
# ============================================================

def segment_chickpeas(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    _, mask = cv2.threshold(
        v,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


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


def create_overlay_image(img_bgr, mask):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    overlay = img_rgb.copy()
    overlay[mask > 0] = [255, 0, 0]

    blended = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)

    return blended


def show_segmentation_preview(img_bgr, mask, title="Preview"):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    blended = create_overlay_image(img_bgr, mask)

    plt.figure(figsize=(16, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Imagem original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Máscara considerada")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(blended)
    plt.title("Área analisada em vermelho")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def save_overlay_image(img_bgr, mask, output_path):
    blended_rgb = create_overlay_image(img_bgr, mask)
    blended_bgr = cv2.cvtColor(blended_rgb, cv2.COLOR_RGB2BGR)

    write_image_unicode(output_path, blended_bgr)


def create_annotated_image(img_bgr, mask, min_object_area=300):
    annotated = img_bgr.copy()

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    object_id = 1

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area < min_object_area:
            continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h_box = stats[label, cv2.CC_STAT_HEIGHT]

        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h_box),
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated,
            str(object_id),
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        object_id += 1

    return annotated


# ============================================================
# PROCURAR IMAGENS
# ============================================================

summary_results = []
object_results = []

image_paths = []

for ext in image_extensions:
    image_paths.extend(input_folder.glob(ext))

print("Pasta das imagens:", input_folder)
print("Pasta dos resultados:", output_folder)
print("Número de imagens encontradas:", len(image_paths))

if len(image_paths) == 0:
    raise ValueError("Não foram encontradas imagens na pasta escolhida.")


# ============================================================
# ANALISAR IMAGENS
# ============================================================

for image_path in image_paths:

    print("\n-----------------------------------------")
    print(f"A analisar: {image_path.name}")

    img_bgr = read_image_unicode(image_path)

    if img_bgr is None:
        print(f"Não foi possível abrir: {image_path.name}")
        continue

    mask = segment_chickpeas(img_bgr)

    if preview_each_image:
        show_segmentation_preview(img_bgr, mask, title=image_path.name)

    if ask_acceptance:
        resposta = input("Aceitar esta segmentação? (s/sim/y/yes/n): ").strip().lower()

        if resposta not in ["s", "sim", "y", "yes"]:
            print(f"Imagem ignorada: {image_path.name}")
            continue

    write_image_unicode(
        output_folder / f"{image_path.stem}_mask.png",
        mask
    )

    save_overlay_image(
        img_bgr,
        mask,
        output_folder / f"{image_path.stem}_overlay_area_considered.png"
    )

    annotated = create_annotated_image(
        img_bgr,
        mask,
        min_object_area=min_object_area
    )

    write_image_unicode(
        output_folder / f"{image_path.stem}_annotated.png",
        annotated
    )

    global_values = extract_color_values(img_bgr, mask)

    if global_values is not None:
        global_values["image"] = image_path.name
        global_values["area_pixels"] = int(np.sum(mask > 0))
        summary_results.append(global_values)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    object_id = 1

    for label in range(1, num_labels):

        area = stats[label, cv2.CC_STAT_AREA]

        if area < min_object_area:
            continue

        object_mask = np.zeros(mask.shape, dtype=np.uint8)
        object_mask[labels == label] = 255

        values = extract_color_values(img_bgr, object_mask)

        if values is None:
            continue

        values["image"] = image_path.name
        values["object_id"] = object_id
        values["area_pixels"] = int(area)
        values["x_centroid"] = centroids[label][0]
        values["y_centroid"] = centroids[label][1]

        object_results.append(values)

        object_id += 1


# ============================================================
# EXPORTAR RESULTADOS
# ============================================================

df_summary = pd.DataFrame(summary_results)
df_objects = pd.DataFrame(object_results)

df_summary.to_csv(
    output_folder / "color_summary_by_image.csv",
    index=False,
    encoding="utf-8-sig"
)

df_objects.to_csv(
    output_folder / "color_by_object.csv",
    index=False,
    encoding="utf-8-sig"
)


print("\n=========================================")
print("Análise concluída.")
print(f"Resultados guardados em: {output_folder}")
print("Número de imagens analisadas:", len(summary_results))
print("Número de objetos/grãos analisados:", len(object_results))