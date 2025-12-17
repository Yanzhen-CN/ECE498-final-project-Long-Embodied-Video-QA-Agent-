from PIL import Image
from typing import List
import os

def load_images(paths: List[str]) -> List[Image.Image]:
    imgs = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"image not found: {p}")
        img = Image.open(p).convert("RGB")
        imgs.append(img)
    return imgs
