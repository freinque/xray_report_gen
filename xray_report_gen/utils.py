import os
import json
from PIL import Image
from .config import IMAGES_DIR, PATH_DOCKER, PATH_HOST, IMAGE_FACTOR, MIN_PIXELS, MAX_PIXELS, MAX_RATIO, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS, FRAME_FACTOR, FPS_MIN_FRAMES, FPS_MAX_FRAMES

def set_api_keys(path_docker=PATH_DOCKER, path_host=PATH_HOST):
    """
    set api keys environment variables
    
    :param path_docker: 
    :param path_host: 
    :return: 
    """
    try:
        with open(os.path.join(path_docker, "hf_token.txt"), 'r') as file:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = file.read().strip()
        print('loaded api key for huggingface')
    except:
        with open(os.path.join(path_host, "hf_token.txt"), 'r') as file:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = file.read().strip()
        print('loaded api key for huggingface')
        pass
    else:
        print('unable to load api key for huggingface')

    try:
        with open(os.path.join(path_docker, "oa_token.txt"), 'r') as file:
            os.environ["OPENAI_API_KEY"] = file.read().strip()
        print('loaded api key for openai')
    except:
        with open(os.path.join(path_host, "oa_token.txt"), 'r') as file:
            os.environ["OPENAI_API_KEY"] = file.read().strip()
        print('loaded api key for openai')
        pass
    else:
        print('unable to load keys from home testProject')


def parse_annotations(df, col, regions):
    for region in regions:
        df[col+'_'+region] = df[col].apply(lambda x: json.loads(x.replace('\'', '\"').replace("```json\n","").replace("```",""))[region])
    return df

def load_images(image_filenames):
    """Load images from a directory."""
    images = []
    for filename in image_filenames:
        images.append(Image.open(filename))
    return images


import logging
import datetime
import os

# Create a logger object
_logger = None

def init_logger(log_dir="./"):
    os.makedirs(log_dir, exist_ok=True)
    global _logger
    _logger = logging.getLogger('MyLogger')
    _logger.setLevel(logging.INFO)  # Set the default logging level to INFO

    # Create a formatter with detailed format including filename and line number
    _formatter = logging.Formatter('%(asctime)s-%(filename)s:%(lineno)d-%(levelname)s >> %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Create a file handler and set the level to INFO
    _file_handler = logging.FileHandler(f'{log_dir}output.{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.log.txt', mode='w')
    _file_handler.setLevel(logging.INFO)
    _file_handler.setFormatter(_formatter)

    # Create a console handler and set the level to INFO
    _console_handler = logging.StreamHandler()
    _console_handler.setLevel(logging.INFO)
    _console_handler.setFormatter(_formatter)

    # Add the handlers to the logger
    _logger.addHandler(_file_handler)
    _logger.addHandler(_console_handler)

def get_logger():
    assert _logger is not None, "Logger is not initialized. Please call init_logger() first."
    return _logger


# original file: https://github.com/kq-chen/qwen-vl-utils/blob/main/src/qwen_vl_utils/vision_process.py
# I made some modifications to the original code.
# 1. Use torchvision.io.VideoReader to read video frames instead of torchvision.io.read_video. The former is much much faster.
# 2. Remove FPS parameter. It is not that necessary.
#from __future__ import annotations

import base64
import math
from io import BytesIO

import requests
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        data = image.split(";", 1)[1]
        if data.startswith("base64,"):
            data = base64.b64decode(data[7:])
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image


def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        # TODO: support http url

        video = ele["video"]
        if video.startswith("file://"):
            video = video[7:]

        frames_data = [f for f in torchvision.io.VideoReader(video, "video")]
        assert(len(frames_data) > 0)

        duration = frames_data[-1]['pts'] - frames_data[0]['pts']
        fps = len(frames_data) / duration

        video = torch.stack([f["data"] for f in frames_data])

        if "nframes" in ele:
            nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
        else:
            min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
            max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, video.size(0))), FRAME_FACTOR)
            nframes = video.size(0) / fps
            nframes = min(max(nframes, min_frames), max_frames)
            nframes = round_by_factor(nframes, FRAME_FACTOR)
        if not (FRAME_FACTOR <= nframes and nframes <= video.size(0)):
            raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {video.size(0)}], but got {nframes}.")

        idx = torch.linspace(0, video.size(0) - 1, nframes).round().long()
        height, width = video.shape[2:]
        video = video[idx]

        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels = ele.get("max_pixels", max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        return images


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None]:
    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_inputs.append(fetch_video(vision_info))
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    return image_inputs, video_inputs
    
