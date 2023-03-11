import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from tldream import shared
from pathlib import Path

import pytest
import torch
import numpy as np


from tldream.util import init_pipe, process


current_dir = Path(__file__).parent.absolute().resolve()

model = current_dir / "anything-v4.5-pruned.safetensors"


@pytest.mark.parametrize("device", ["mps", "cpu", "cuda"])
@pytest.mark.parametrize("sampler", ["ddim", "uni_pc"])
@pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("low_vram", [False, True])
def test_ckpt_model(device, sampler, torch_dtype, low_vram):
    shared.use_xformers = device == "cuda"
    if device == "mps":
        if not torch.backends.mps.is_available():
            return
        if torch_dtype == torch.float16:
            return

    if device == "cuda" and not torch.cuda.is_available():
        return
    if device == "cpu" and torch_dtype == torch.float16:
        return

    device = torch.device(device)
    controlled_model = init_pipe(
        str(model), device, torch_dtype=torch_dtype, cpu_offload=low_vram
    )

    image = np.zeros((254, 267, 3), dtype=np.uint8)

    process(
        controlled_model,
        sampler,
        image,
        "Hello",
        negative_prompt="hello",
        guidance_scale=9,
        steps=1,
        width=256,
        height=256,
    )
