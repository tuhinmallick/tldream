import imghdr
import os
from dataclasses import dataclass

from tldream import shared

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import socket
import io
import random
from pathlib import Path
from urllib.parse import urlparse

import cv2
import einops
import numpy as np
import torch
from PIL import Image, ImageOps
from diffusers import UniPCMultistepScheduler, DDIMScheduler, DiffusionPipeline
from loguru import logger
from diffusers.utils import DIFFUSERS_CACHE
import safetensors
from torch.hub import download_url_to_file

from tldream.cldm.hack import enable_sliced_attention
from tldream.cldm.model import create_model, load_state_dict
from tldream.ldm.models.diffusion.ddim import DDIMSampler
from tldream.ldm.models.diffusion.uni_pc import UniPCSampler

current_dir = Path(__file__).parent.absolute().resolve()

CONTROlNET_SCRIBBLE_DIFF_MODEL_URL = os.environ.get(
    "CONTROlNET_SCRIBBLE_DIFF_MODEL_URL",
    "https://huggingface.co/kohya-ss/ControlNet-diff-modules/resolve/main/diff_control_sd15_scribble_fp16.safetensors",
)


def get_model_path(model_url_or_path, save_dir):
    if os.path.exists(model_url_or_path):
        logger.info(f"Loading model from: {model_url_or_path}")
        return model_url_or_path

    if model_url_or_path.startswith("http"):
        parts = urlparse(model_url_or_path)
        filename = os.path.basename(parts.path)
        dst_p = str(save_dir / filename)
        if os.path.exists(dst_p):
            logger.info(f"Loading model from: {dst_p}")
            return dst_p

        logger.info(f"Downloading {filename} to {save_dir.absolute()}")
        download_url_to_file(model_url_or_path, dst_p, progress=True)
        return dst_p

    raise ValueError(
        f"model {model_url_or_path} is invalid, please pass a local file path or a url."
    )


def init_ckpt_pipe(model_path, device, torch_dtype, low_vram):
    if low_vram:
        enable_sliced_attention()

    cfg_path = current_dir / "cldm_v15.yaml"
    model = create_model(str(cfg_path), str(device)).cpu()
    model.low_vram = low_vram
    model.device = device

    logger.info(f"Loading model from: {model_path}")
    model_state_dict = load_state_dict(model_path, location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(
        model_state_dict, strict=False
    )

    diff_model_path = get_model_path(
        CONTROlNET_SCRIBBLE_DIFF_MODEL_URL, Path(DIFFUSERS_CACHE)
    )
    diff_state_dict = safetensors.torch.load_file(diff_model_path, device="cpu")
    model.load_control_diff(diff_state_dict)
    # if len(missing_keys) > 0:
    #     logger.error(f"Missing keys in state_dict: {missing_keys}")
    #     exit(-1)

    model = model.to(torch_dtype).eval()
    model = model.to(device)

    return model


def init_pipe(
    model_id,
    device,
    torch_dtype,
    cpu_offload=False,
    nsfw_filter=True,
):
    if Path(model_id).is_file():
        return init_ckpt_pipe(model_id, device, torch_dtype, cpu_offload)

    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

    logger.info(f"Loading model: {model_id}")

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble",
        torch_dtype=torch_dtype,
    )

    kwargs = {}
    if not nsfw_filter:
        kwargs.update({"safety_checker": None})

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
        **kwargs,
    )
    pipe.enable_attention_slicing()

    if cpu_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(device)

    if shared.use_xformers:
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


all_ckpt_samplers = {
    "uni_pc": UniPCSampler,
    "ddim": DDIMSampler,
}


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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def ckpt_process(
    model,
    device,
    torch_dtype,
    sampler: str,
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
    sampler = all_ckpt_samplers[sampler](model, device, torch_dtype)
    logger.info(f"Original image shape: {input_image.shape}")
    img = HWC3(input_image)
    img = preprocess_image(img, width, height)

    original_h, original_w = img.shape[:2]
    img = pad_img_to_modulo(img, 64)
    logger.info(f"Resized image shape: {img.shape}")
    H, W, C = img.shape

    detected_map = np.zeros_like(img, dtype=np.uint8)
    detected_map[np.min(img, axis=2) < 127] = 255

    control = torch.from_numpy(detected_map.copy()).to(torch_dtype).to(device) / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, "b h w c -> b c h w").clone()

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    seed_everything(seed)

    low_vram = model.low_vram
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

    samples = sampler.sample(
        steps,
        num_samples,
        shape,
        cond,
        verbose=False,
        eta=0.0,
        unconditional_guidance_scale=guidance_scale,
        unconditional_conditioning=un_cond,
        img_callback=callback,
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
    res_rgb_img = results[0]
    # remove padding
    res_rgb_img = res_rgb_img[:original_h, :original_w]
    return res_rgb_img


@torch.no_grad()
def process(
    pipe,
    device,
    torch_dtype,
    sampler: str,
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
    if not isinstance(pipe, DiffusionPipeline):
        return ckpt_process(
            pipe,
            device,
            torch_dtype,
            sampler,
            input_image,
            prompt,
            negative_prompt,
            num_samples,
            steps,
            width,
            height,
            guidance_scale,
            seed,
            callback,
        )
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
