import numpy as np
import base64
import io
from PIL import Image
import cv2
import json
from pathlib import Path
import yaml

def str2img(bstr:str, convert_pil:bool=True, is_rgb:bool=True)-> np.ndarray:
    image_data = base64.b64decode(bstr)
    databytes = io.BytesIO(image_data)
    out = Image.open(databytes)
    if not convert_pil:
        out = np.array(out).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR) if not is_rgb else out
    return out

def img2str(img):
    '''
    this function only np.ndarray(bgr) and PIL.Image
    not consider bgra...
    '''
    img = img.copy()
    if isinstance(img, np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
    elif isinstance(img, Image.Image):
        ...
    else:
        raise ValueError(f"img must be np.ndarray or Image but {type(img)}")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    encoded = base64.b64encode(img_byte_arr.getvalue())
    decoded = encoded.decode('ascii')
    return decoded

def load_yaml_to_json(path):
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)