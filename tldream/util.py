import imghdr
import socket
import io
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from diffusers import UniPCMultistepScheduler, DDIMScheduler
from loguru import logger
from diffusers.utils import is_xformers_available

current_dir = Path(__file__).parent.absolute().resolve()


def init_pipe(model_id, device, torch_dtype, cpu_offload, nsfw_filter):
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

    logger.info(f"Loading model: {model_id}")

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble", torch_dtype=torch_dtype
    )

    kwargs = {}
    if not nsfw_filter:
        kwargs.update({"safety_checker": None})

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id, controlnet=controlnet, torch_dtype=torch_dtype, **kwargs
    )
    pipe.enable_attention_slicing()

    if cpu_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(device)

    if is_xformers_available():
        pipe.enable_xformers_memory_efficient_attention()
    return pipe


def get_image_ext(img_bytes):
    w = imghdr.what("", img_bytes)
    if w is None:
        w = "jpeg"
    return w


def preprocess_image(image, dst_width, dst_height):
    """

    Args:
        image: 前端给到的原始草图，尺寸不固定，可能比 dst_width, dst_height 大，也可能比它们小
        dst_width: 生成的图片的宽度
        dst_height: 生成的图片的高度

    Returns:
        一个尺寸为 (dst_height, dst_width, 3) 的 numpy array
    """
    original_h, original_w = image.shape[:2]

    # 图片的长边缩放小于目标尺寸的短边
    max_side_length = min(dst_width, dst_height)
    if max(original_h, original_w) > max_side_length:
        k = float(max_side_length) / max(original_h, original_w)
        new_h = int(original_h * k)
        new_w = int(original_w * k)
        image = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA,
        )
    else:
        new_h = original_h
        new_w = original_w

    new_img = np.ones((dst_height, dst_width, 3), dtype=np.uint8) * 255
    x = (dst_width - new_w) // 2
    y = (dst_height - new_h) // 2
    # paste image to the center of new_img
    new_img[y : y + new_h, x : x + new_w, :] = image

    return new_img


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img: np.ndarray, mod: int):
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    height, width = img.shape[:2]
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)

    return np.pad(
        img,
        ((0, out_height - height), (0, out_width - width), (0, 0)),
        mode="symmetric",
    )


def pil_to_bytes(pil_img, ext: str) -> bytes:
    with io.BytesIO() as output:
        pil_img.save(output, format=ext, quality=95)
        image_bytes = output.getvalue()
    return image_bytes


def load_img(img_bytes, gray: bool = False, return_exif: bool = False):
    alpha_channel = None
    image = Image.open(io.BytesIO(img_bytes))

    try:
        if return_exif:
            exif = image.getexif()
    except:
        exif = None
        logger.error("Failed to extract exif from image")

    try:
        image = ImageOps.exif_transpose(image)
    except:
        pass

    if gray:
        image = image.convert("L")
        np_img = np.array(image)
    else:
        if image.mode == "RGBA":
            np_img = np.array(image)
            alpha_channel = np_img[:, :, -1]
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
        else:
            image = image.convert("RGB")
            np_img = np.array(image)

    if return_exif:
        return np_img, alpha_channel, exif
    return np_img, alpha_channel


@torch.no_grad()
def process(
    pipe,
    sampler,
    input_image: np.ndarray,
    prompt: str,
    negative_prompt: str,
    num_samples: int = 1,
    steps: int = 20,
    width: int = 640,
    height: int = 640,
    guidance_scale: float = 9.0,
    seed: int = -1,
    callback=None,
):
    # return rgb image
    if seed == -1:
        seed = random.randint(0, 999999999)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if sampler == "uni_pc":
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    elif sampler == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    img = preprocess_image(input_image, width, height)
    resized_h, resized_w = img.shape[:2]
    img = pad_img_to_modulo(img, 32)
    logger.info(
        f"Original image shape: {input_image.shape}, Resized & Padded image shape: {img.shape}"
    )

    output = pipe(
        prompt,
        Image.fromarray(img),
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        generator=generator,
        callback=callback,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_samples,
    )
    res_image = output.images[0]
    logger.info(f"Result image shape: {res_image.size}")

    res_rgb_img = np.asarray(res_image)
    # remove padding
    res_rgb_img = res_rgb_img[:resized_h, :resized_w]
    return res_rgb_img


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_ip() -> str:
    # https://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        # doesn't even have to be reachable
        s.connect(("10.254.254.254", 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP
