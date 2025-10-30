import os
from io import BytesIO

import fal_client
import requests
from PIL import Image


def get_image_from_nano_banana(prompt: str, aspect_ratio: str) -> Image.Image:
    """fal-ai nano-banana 모델을 동기적으로 호출해 이미지를 반환한다."""
    result = fal_client.subscribe(
        "fal-ai/nano-banana",
        arguments={
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
        },
    )
    image_url = result["images"][0]["url"]
    response = requests.get(image_url)
    return Image.open(BytesIO(response.content)).convert("RGB")
