import random
from pathlib import Path

import cv2
import einops
import numpy as np
import torch
from typer import Typer, Option

from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.tldream_util import seed_everything

current_dir = Path(__file__).parent.absolute().resolve()


def init_model(model_path, device):
    cfg_path = current_dir / "models" / "cldm_v15.yaml"
    model = create_model(str(cfg_path), device).cpu()
    model.load_state_dict(load_state_dict(model_path, location="cpu"))
    model = model.to(device)
    return model


@torch.no_grad()
def process(
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
):
    # return rgb image
    ddim_sampler = DDIMSampler(model, device)
    img = resize_image(HWC3(input_image), max_side_length)
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
        "c_crossattn": [
            model.get_learned_conditioning([prompt] * num_samples)
        ],
    }
    un_cond = {
        "c_concat": [control],
        "c_crossattn": [model.get_learned_conditioning([negative_prompt] * num_samples)],
    }
    shape = (4, H // 8, W // 8)

    if low_vram:
        model.low_vram_shift(is_diffusing=True)

    samples, intermediates = ddim_sampler.sample(
        ddim_steps,
        num_samples,
        shape,
        cond,
        verbose=False,
        eta=eta,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=un_cond,
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


app = Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def main(
    device: str = Option("mps", help="Device to use (cuda, cpu or mps)"),
    model: Path = Option(
        "/Users/cwq/code/github/ControlNet/models/control_any3_better_scribble.pth",
        help="Path to model",
    ),
    low_vram: bool = Option(True, help="Use low vram mode"),
):
    model = init_model(model, device)
    img = cv2.imread("/Users/cwq/code/github/ControlNet/test_imgs/user_1.png")
    rgb_images = process(
        model,
        device,
        img,
        "a turtle in river",
        "",
        low_vram=low_vram,
    )
    cv2.imwrite("test.jpg", cv2.cvtColor(rgb_images[0], cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    app()
