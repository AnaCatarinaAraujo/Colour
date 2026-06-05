import cv2
import numpy as np
from pathlib import Path

def read_image_unicode(path: Path) -> np.ndarray | None:
    """Reads an image from a path that might contain unicode characters."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Erro ao ler imagem {path}: {e}")
        return None

def write_image_unicode(path: Path, img: np.ndarray) -> bool:
    """Writes an image to a path that might contain unicode characters."""
    ext = path.suffix
    success, encoded_img = cv2.imencode(ext, img)

    if success:
        encoded_img.tofile(str(path))
        return True
    else:
        print(f"Não foi possível guardar: {path}")
        return False
