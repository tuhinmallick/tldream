import os
from pathlib import Path

from safetensors.torch import save_file
import time

import torch
from loguru import logger

from omegaconf import OmegaConf
from tldream.ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get("state_dict", d)


def load_state_dict(ckpt_path, location="cpu"):
    _, extension = os.path.splitext(ckpt_path)
    start = time.time()
    if extension.lower() == ".safetensors":
        import safetensors.torch

        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(
            torch.load(ckpt_path, map_location=torch.device(location))
        )
    state_dict = get_state_dict(state_dict)

    logger.info(f"State dict load time: {time.time() - start:.2f}s")

    # fp16_state_dict = {}
    # for k, v in state_dict.items():
    #     fp16_state_dict[k] = v.to(torch.float16)
    # save_file(fp16_state_dict, f"{Path(ckpt_path).stem}_fp16.safetensors")
    # exit(-1)
    # print(f"Loaded state_dict from [{ckpt_path}]")

    return state_dict


def create_model(config_path, device):
    config = OmegaConf.load(config_path)
    config["model"]["params"]["cond_stage_config"]["params"] = {"device": device}
    model = instantiate_from_config(config.model).cpu()
    print(f"Loaded model config from [{config_path}]")
    return model
