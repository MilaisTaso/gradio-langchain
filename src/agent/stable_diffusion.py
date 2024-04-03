import base64
import os
from typing import Any
import io
import uuid

import numpy as np
import requests
from langchain.callbacks.manager import get_openai_callback
from langchain.globals import set_debug, set_verbose
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.pydantic_v1 import SecretStr
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from PIL import Image

from src.config import get_config

config = get_config()

if config.LANGCHAIN_DEBUG_MODE == "ALL":
    set_debug(True)
elif config.LANGCHAIN_DEBUG_MODE == "VERBOSE":
    set_verbose(True)

SD_DESIGN_TEMPLATE = """
    You are an excellent designer. With the information given by the user, you can describe an illustration that would impress any illustrator or novelist.

    All you have to do is to use your imagination to describe the details of the illustration scene from the information given by the user.
    Specifically, you should describe the person's clothing, hairstyle, facial expression, age, gender, and other external characteristics; the person's facial expression, state of mind, and emotional landscape; the illustration's composition and object placement (what objects are where and their characteristics); the surrounding landscape and geography, weather and sky conditions, light levels, and the atmosphere conveyed to the person viewing the illustration.
    You will describe the scenery and the placement of the objects (what objects are located where and their characteristics), the surrounding landscape and geography, the weather and sky, the light and the atmosphere conveyed to the viewer. You are very good at describing a scene in a way that appeals to the user. Users are looking for illustrations with people in them. Another person will do the actual illustration, so you should concentrate only on describing the details.

    Use your imagination.
"""

SD_PROMPT_TEMPLATE = """
    You are a talented illustrator. From a description of a scene given by a designer, you can use Stable Diffusion (an image generation model) to generate an illustration that will amaze any designer or artist.

    To generate an illustration, a list of words called "prompt" is required. The prompt determine the quality of the illustration. The more variegated words you include, the more information you include, the better the illustration.
    Please output a brief, carefully selected output of about 20 words for the prompt. You do not have to present the words as they are given by the user, and you may supplement them with other words from your imagination if necessary.

    Prompt output must be in English, and output must be comma-separated word strings.
"""


def generate_sd_prompt(
    prompt: str, models: str, temperature: float, aspect_ratio: str, art_style: str
) -> tuple[str, OpenAICallbackHandler, Any]:
    api_key = None
    if config.OPENAI_API_KEY:
        api_key = SecretStr(config.OPENAI_API_KEY)

    sd_design_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SD_DESIGN_TEMPLATE),
            ("human", "{human_input}"),
        ]
    )

    sd_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SD_PROMPT_TEMPLATE),
            ("human", "{ai_input}"),
        ]
    )

    lim = ChatOpenAI(
        model=models, api_key=api_key, temperature=temperature, verbose=True
    )

    output_parser = StrOutputParser()

    chain = (
        {
            "ai_input": sd_design_template | lim | output_parser,
            "human_input": RunnablePassthrough(),
        }
        | sd_prompt_template
        | lim
        | output_parser
    )

    with get_openai_callback() as callback:
        response = chain.invoke({"human_input": prompt})
    image = generate_image(response, aspect_ratio, art_style)

    return (response, callback, image)


def generate_image(prompt: str, aspect_ratio: str = "1:1", art_style: str = ""):
    quality_prompt = "best quality, masterpiece, extremely detailed, "
    negative_prompt = "low quality, worst quality, out of focus, ugly, error, jpeg artifacts, lowers, blurry, bokeh, \
        bad anatomy, long_neck, long_body, longbody, deformed mutated disfigured, missing arms, extra_arms, mutated hands, \
        extra_legs, bad hands, poorly_drawn_hands, malformed_hands, missing_limb, floating_limbs, disconnected_limbs, extra_fingers, \
        bad fingers, liquid fingers, poorly drawn fingers, missing fingers, extra digit, fewer digits, ugly face, deformed eyes, \
        partial face, partial head, bad face, inaccurate limb, cropped text, signature, watermark, username, artist name, stamp, title, \
        subtitle, date, footer, header"

    # ? local hugging face
    # url = "http://127.0.0.1:7860"
    # payload = {
    #     "prompt": quality_prompt + prompt,
    #     "negative_prompt": negative_prompt,
    #     "steps": 35,
    #     "width": width,
    #     "height": height,
    # }
    # response = requests.post(url=f"{url}/sdapi/v1/txt2img", json=payload)

    # decoded_data = base64.b64decode(response.json()["images"][0])

    # image = Image.open(io.BytesIO(decoded_data))
    # image_np = np.array(image)

    # return image_np

    #! version1
    # engine_id = "stable-diffusion-v1-6"
    # api_host = config.STABILITY_API_HOST
    # api_key = config.STABILITY_API_KEY

    # if api_key is None:
    #     raise Exception("Missing Stability API key.")

    # response = requests.post(
    #     f"{api_host}/v1/generation/{engine_id}/text-to-image",
    #     headers={
    #         "Content-Type": "application/json",
    #         "Accept": "application/json",
    #         "Authorization": f"Bearer {api_key}"
    #     },
    #     json={
    #         "text_prompts": [
    #             {
    #                 "text": quality_prompt + prompt,
    #                 "weight": 0.7
    #             }
    #         ],
    #         "cfg_scale": 7,
    #         "height": height,
    #         "width": width,
    #         "samples": 1,
    #         "steps": 30,
    #         "sampler": "DDIM",
    #         "style_preset": "anime"
    #     },
    # )

    # data = response.json()

    # if response.status_code != 200:
    #     raise Exception("Non-200 response: " + str(response.text))

    # # 保存先のディレクトリパス
    # output_dir = "./output"

    # # ディレクトリが存在しない場合は作成
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # output_images = []

    # for i, image in enumerate(data["artifacts"]):
    #     decoded_image = base64.b64decode(image["base64"])
    #     io_image = Image.open(io.BytesIO(decoded_image))
    #     image_np = np.array(io_image)
    #     output_images.append(image_np)

    #     with open(f"./{output_dir}/v1_txt2img_{i}.png", "wb") as f:
    #         f.write(decoded_image)

    # return output_images

    # * stability core
    api_key = config.STABILITY_API_KEY

    if api_key is None:
        raise Exception("Missing Stability API key.")

    response = requests.post(
        f"https://api.stability.ai/v2beta/stable-image/generate/core",
        headers={"authorization": api_key, "accept": "application/json"},
        files={"none": ""},
        data={
            "prompt": f"({quality_prompt}:1.3) + ({art_style}:1.4) + {prompt}",
            "negative_prompt": negative_prompt,
            "output_format": "webp",
            "aspect_ratio": aspect_ratio,
        },
    )

    data = response.json()

    if response.status_code == 200 and data.get("finish_reason") == "SUCCESS":
        # # 保存先のディレクトリパス
        output_dir = "./output"

        # ディレクトリが存在しない場合は作成
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        decoded_image = base64.b64decode(data.get("image"))
        io_image = Image.open(io.BytesIO(decoded_image))
        image_np = np.array(io_image)

        with open(f"./{output_dir}/v1_txt2img_{uuid.uuid4()}.webp", "wb") as file:
            file.write(decoded_image)
    else:
        raise Exception(str(response.json()))

    return image_np
