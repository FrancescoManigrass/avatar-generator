import cv2
import torch
import albumentations as albu
from iglovikov_helper_functions.utils.image_utils import pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from people_segmentation.pre_trained_models import create_model

model = create_model("Unet_2020-07-20")
model.eval()
transform = albu.Compose([albu.Normalize(p=1)], p=1)

def segment(rgb_or_path):
    if isinstance(rgb_or_path, str):
        bgr = cv2.imread(rgb_or_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = rgb_or_path  # numpy RGB

    padded_image, pads = pad(rgb, factor=32)
    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

    with torch.no_grad():
        prediction = model(x)[0][0]

    mask = (prediction > 0).cpu().numpy().astype("uint8")
    mask = unpad(mask, pads)
    return (mask * 255).astype("uint8")
