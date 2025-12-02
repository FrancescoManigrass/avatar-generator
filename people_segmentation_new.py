# people_segmentation.py
import numpy as np
from rembg import remove
from PIL import Image

def segment(x):
    """
    x può essere:
      - un path a file immagine (str)
      - un array RGB numpy (H,W,3)
    Ritorna una mask uint8 {0,255}.
    """
    if isinstance(x, str):
        im = Image.open(x).convert("RGBA")
    else:
        im = Image.fromarray(x).convert("RGBA")

    out = remove(im)  # sfondo rimosso, alpha nell'ultimo canale
    alpha = np.array(out.split()[-1])  # [H, W]
    mask = (alpha > 0).astype("uint8") * 255
    return mask
