import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .io import write_image_unicode

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

def save_overlay_image(img_bgr, mask, output_path: Path):
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
