from dotenv import load_dotenv
from typing import List, Dict
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from utils import *
from models import *
from misc import *
from tqdm import tqdm
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
    chat_llm = ChatOpenAI(model=model, temperature=0.4)

    # api call
    gpt_response = chat_llm.invoke(messages)

    # parse response
    outs = parser.parse(gpt_response.content)
    return outs["captions"]


def generation_image(model_name: str, caption: str, ratio: str) -> Image.Image:
    out: Image.Image = None
    if model_name == "fal-ai/nano-banana":
        out = get_image_from_nano_banana(caption, ratio)
    elif model_name == "qwen-image":
        out = get_image_from_qwen(caption, ratio)
    else:
        raise ValueError(f"model_name: {model_name} is not implement")
    return out


def main(n_image=5):
    input_prompt = "사실적인 현대 아시아인을 만들어달라, 이미지는 얼굴이 잘 나와있어야 하며, 이미지 속에는 단 한명의 인물만 있어야 한다"
    n_image = n_image
    caption_gen_model = "gpt-4o-mini"
    image_gen_model = "qwen-image"
    output_dir = "generated_images__qwen-image_n6000"

    # generate image gen prompts(=captions)
    captions = generate_candidate_prompts(caption_gen_model, input_prompt, n_image)

    gen_images = []
    for idx, caption in enumerate(captions):
        # select ratio
        ratio = random.choice(
            [
                "1:1",
                "16:9",
                "9:16",
                "4:3",
                "3:4",
                "3:2",
                "2:3",
            ]
        )

        # image_generation
        gen_image = generation_image(
            model_name=image_gen_model,
            caption=caption,
            ratio=ratio,
        )
        # append image
        gen_images.append(gen_image)

    # prepare output dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # save image
    for idx, gen_image in enumerate(gen_images):
        filename = current_timestamp() + f"_{idx}" + ".jpg"
        gen_image.save(output_dir / filename)


if __name__ == "__main__":
    for _ in tqdm(range(500)):
        main(n_image=12)
