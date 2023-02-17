import io
import random

import cv2
import einops
import numpy as np
import torch
from PIL import Image, ImageOps
from loguru import logger
from pytorch_lightning import seed_everything

from tldream.ldm.models.diffusion.ddim import DDIMSampler


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, max_side_length):
    # keep ratio, resize max side of image to resolution
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(max_side_length) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(
        input_image,
        (W, H),
        interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA,
    )
    return img


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
    img = HWC3(input_image)
    # img = resize_image(, max_side_length)

    crop_pad = 16
    original_h, original_w = img.shape[:2]
    img = img[crop_pad : original_h - crop_pad, crop_pad : original_w - crop_pad]
    img = pad_img_to_modulo(img, 64)
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
