import io
import os
import random
from typing import Optional

import cv2
import einops
import numpy as np
import torch
from PIL import Image, ImageOps
from loguru import logger
from pytorch_lightning import seed_everything

from annotator.util import resize_image, HWC3
from ldm.models.diffusion.ddim import DDIMSampler


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
async def process(
    model,
    device: str,
    input_image: np.ndarray,
    prompt: str,
    negative_prompt: str,
    num_samples: int = 1,
    max_side_length: int = 512,
    ddim_steps: int = 20,
    scale: float = 9.0,
    seed: int = -1,
    eta: float = 0.0,
    low_vram: bool = True,
    callback=None,
):
    # return rgb image
    ddim_sampler = DDIMSampler(model, device)
    logger.info(f"Original image shape: {input_image.shape}")
    img = resize_image(HWC3(input_image), max_side_length)
    logger.info(f"Resized image shape: {img.shape}")
    H, W, C = img.shape

    detected_map = np.zeros_like(img, dtype=np.uint8)
    detected_map[np.min(img, axis=2) < 127] = 255

    control = torch.from_numpy(detected_map.copy()).float().to(device) / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, "b h w c -> b c h w").clone()

    if seed == -1:
        seed = random.randint(0, 9999999999)
    seed_everything(seed)

    if low_vram:
        model.low_vram_shift(is_diffusing=False)

    cond = {
        "c_concat": [control],
        "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)],
    }
    un_cond = {
        "c_concat": [control],
        "c_crossattn": [
            model.get_learned_conditioning([negative_prompt] * num_samples)
        ],
    }
    shape = (4, H // 8, W // 8)

    if low_vram:
        model.low_vram_shift(is_diffusing=True)

    samples, intermediates = await ddim_sampler.sample(
        ddim_steps,
        num_samples,
        shape,
        cond,
        verbose=False,
        eta=eta,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=un_cond,
        callback=callback,
    )

    if low_vram:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (
        (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )

    results = [x_samples[i] for i in range(num_samples)]
    return results


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
