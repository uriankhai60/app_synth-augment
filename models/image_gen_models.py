import os
from io import BytesIO
from diffusers import DiffusionPipeline
import fal_client
import requests
from PIL import Image
import torch
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
import math

model_name = "Qwen/Qwen-Image"
torch_dtype = torch.bfloat16
device = "cuda"

## prepare
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),  # We use shift=3 in distillation
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),  # We use shift=3 in distillation
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,  # set shift_terminal to None
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image", scheduler=scheduler, torch_dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights(
    "lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V1.0.safetensors"
)

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


def get_image_from_qwen(prompt:str, ratio:str) -> Image.Image:
    aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
    }
    width, height = aspect_ratios[ratio]
    outs = pipe(
        prompt = prompt,
        negative_prompt = " ",
        width = width,
        height=height,
        true_cfg_scale=1.0,
        num_inference_steps=8,
    )
    return outs.images[0]
