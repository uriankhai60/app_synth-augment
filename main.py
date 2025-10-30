from dotenv import load_dotenv
from typing import List, Dict
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from utils import *
from models import *
from misc import *

import os
import random

load_dotenv()
logging.langsmith("deepfake")

image_caption_template = load_yaml_to_json("templates/image_caption_template.yaml")


def generate_candidate_prompts(model: str, prompt: str, num_captions: int) -> List[str]:
    """
    이미지 생성용 캡션을 만드는 함수
    """
    # prepare prompts
    system_prompt = (
        image_caption_template.get("system")
        .get("template")
        .format(N=num_captions, USER_PROMPT=prompt)
    )
    user_prompt = (
        image_caption_template.get("user")
        .get("template")
        .format(N=num_captions, USER_PROMPT=prompt)
    )

    # prepare format instructions
    schemas = [
        ResponseSchema(
            name="captions",
            description="List[str] containing image-generation captions (each item is a full sentence).",
        )
    ]
    parser = StructuredOutputParser.from_response_schemas(schemas)
    format_instructions = parser.get_format_instructions()

    # assign messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt + "\n" + f"{format_instructions}"},
    ]

    # gpt model ready
    chat_llm = ChatOpenAI(model=model, temperature=0.2)

    # api call
    gpt_response = chat_llm.invoke(messages)

    # parse response
    outs = parser.parse(gpt_response.content)
    return outs["captions"]


def generation_image(
    model_name: str, caption: str, width: int, height: int, ratio: str
) -> Image.Image:
    out: Image.Image = None
    if model_name == "fal-ai/nano-banana":
        out = get_image_from_nano_banana(caption, ratio)
    elif model_name == "...":
        ...
    elif model_name == "...":
        ...
    else:
        raise ValueError(f"model_name: {model_name} is not implement")
    return out


def main():
    input_prompt = "사실적인 2024년의 한국인을 만들어달라, 이미지는 얼굴이 잘 나와있어야 하며, 이미지 속에는 단 한명의 인물만 있어야 한다"
    n_image = 5
    caption_gen_model = "gpt-4"
    image_gen_model = "fal-ai/nano-banana"
    output_dir = "generated_images"

    # generate image gen prompts(=captions)
    captions = generate_candidate_prompts(caption_gen_model, input_prompt, n_image)

    meta = {}
    gen_images = []
    for idx, caption in enumerate(captions):

        # select ratio and resolution
        selected = random.choice(
            [
                {"1:1": (1024, 1024)},
                {"4:5": (912, 1140)},
                {"3:4": (885, 1180)},
                {"3:2": (1254, 836)},
                {"16:9": (1360, 765)},
            ]
        )
        ratio, (width, height) = next(iter(selected.items()))

        # image_generation
        gen_image = generation_image(
            model_name=image_gen_model,
            caption=caption,
            width=width,
            height=height,
            ratio=ratio,
        )
        # append image
        gen_images.append(gen_image)
        # save meta info
        meta[idx] = {
            "filename": current_timestamp() + ".png",
            "caption": caption,
            "resolution": gen_image.size,
            "ratio": ratio,
            "model_name": image_gen_model,
        }

    # prepare output dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # save meta
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # save image
    for idx in range(len(meta)):
        pil_image = gen_images[idx]
        filename = meta.get(idx).get("filename")
        pil_image.save(output_dir / filename)


if __name__ == "__main__":
    main()
