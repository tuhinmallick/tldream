import pytest as pytest
import torch

from tldream import shared
from tldream.util import process, init_pipe
import numpy as np

model = "runwayml/stable-diffusion-v1-5"


@pytest.mark.parametrize("device", ["mps", "cpu", "cuda"])
@pytest.mark.parametrize("sampler", ["ddim", "uni_pc"])
def test_model(device, sampler):
    shared.use_xformers = device == "cuda"

    if device == "mps" and not torch.backends.mps.is_available():
        return
    if device == "cuda" and not torch.cuda.is_available():
        return

    device = torch.device(device)
    controlled_model = init_pipe(
        model, device, torch_dtype=torch.float32, cpu_offload=False
    )

    image = np.zeros((254, 267, 3), dtype=np.uint8)

    process(
        controlled_model,
        device,
        torch.float32,
        sampler,
        image,
        "Hello",
        negative_prompt="hello",
        guidance_scale=9,
        steps=1,
        width=256,
        height=256,
    )


@pytest.mark.parametrize("sampler", ["ddim", "uni_pc"])
@pytest.mark.parametrize("torch_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("cpu_offload", [True, False])
@pytest.mark.parametrize("nsfw_filter", [True, False])
def test_cuda_model(sampler, torch_dtype, cpu_offload, nsfw_filter):
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    controlled_model = init_pipe(
        model,
        device,
        torch_dtype=torch_dtype,
        cpu_offload=cpu_offload,
        nsfw_filter=nsfw_filter,
    )

    image = np.zeros((254, 267, 3), dtype=np.uint8)

    process(
        controlled_model,
        device,
        torch_dtype,
        sampler,
        image,
        "Hello",
        negative_prompt="hello",
        guidance_scale=9,
        steps=2,
        width=256,
        height=256,
    )
