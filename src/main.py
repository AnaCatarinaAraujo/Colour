import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

from colour.io import read_image_unicode, write_image_unicode
from colour.vision import (
    segment_chickpeas,
    show_segmentation_preview,
    save_overlay_image,
    create_annotated_image
)
from colour.core import extract_color_values

IMAGE_EXTENSIONS = [
    "*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff",
    "*.JPG", "*.JPEG", "*.PNG", "*.TIF", "*.TIFF"
]

def main():
    parser = argparse.ArgumentParser(description="Analyze colours of objects in images.")

    parser.add_argument(
        "--min-area",
        type=int,
        default=300,
        help="Minimum object area in pixels to be considered."
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show segmentation preview for each image."
    )
    parser.add_argument(
        "--ask-acceptance",
        action="store_true",
        help="Prompt for user acceptance of each segmentation mask."
    )
    
    args = parser.parse_args()
    
    input_folder = Path.cwd() / "images"
    output_folder = Path.cwd() / "resultados_cor"
    min_object_area = args.min_area
    preview_each_image = args.preview
    ask_acceptance = args.ask_acceptance
    
    print("Pasta das imagens:", input_folder)
    print("Pasta dos resultados:", output_folder)
    
    if not input_folder.exists():
        print(f"Erro: A pasta de entrada {input_folder} não existe.")
        return
        
    output_folder.mkdir(parents=True, exist_ok=True)
    
    summary_results = []
    object_results = []

    image_paths = []

    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(input_folder.glob(ext))

    print("Número de imagens encontradas:", len(image_paths))

    if len(image_paths) == 0:
        print("Aviso: Não foram encontradas imagens na pasta escolhida.")
        return

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

    if summary_results:
        df_summary = pd.DataFrame(summary_results)
        df_summary.to_csv(
            output_folder / "color_summary_by_image.csv",
            index=False,
            sep=';',
            decimal=',',
            encoding="utf-8-sig"
        )

    if object_results:
        df_objects = pd.DataFrame(object_results)
        df_objects.to_csv(
            output_folder / "color_by_object.csv",
            index=False,
            sep=';',
            decimal=',',
            encoding="utf-8-sig"
        )

    print("\n=========================================")
    print("Análise concluída.")
    print(f"Resultados guardados em: {output_folder}")
    print("Número de imagens analisadas:", len(summary_results))
    print("Número de objetos/grãos analisados:", len(object_results))

if __name__ == "__main__":
    main()
