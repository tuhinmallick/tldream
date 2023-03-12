import json
import os
import webbrowser
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image
from loguru import logger

from tldream.const import *

_config_file = None

current_dir = Path(__file__).parent.absolute().resolve()
assets_dir = current_dir / "assets"


@dataclass
class Config:
    listen: bool = DEFAULT_LISTEN
    port: int = DEFAULT_PORT
    device: str = DEFAULT_DEVICE
    model: str = DEFAULT_MODEL
    low_vram: bool = DEFAULT_LOW_VRAM
    fp32: bool = DEFAULT_FP32
    nsfw_filter: bool = DEFAULT_NSFW_FILTER
    cache_dir: str = None
    local_files_only: bool = DEFAULT_LOCAL_FILES_ONLY


def load_config(installer_config: str):
    if os.path.exists(installer_config):
        with open(installer_config, "r", encoding="utf-8") as f:
            return Config(**json.load(f))
    else:
        return Config()


def save_config(
    listen,
    port,
    device,
    model,
    low_vram,
    fp32,
    nsfw_filter,
    cache_dir,
    local_files_only,
):
    config = Config(**locals())
    print(config)

    current_time = datetime.now().strftime("%H:%M:%S")
    try:
        with open(_config_file, "w", encoding="utf-8") as f:
            json.dump((asdict(config)), f, indent=4, ensure_ascii=False)
        msg = f"[{current_time}] Successful save config to: {os.path.abspath(_config_file)}"
        logger.info(msg)
    except Exception as e:
        return f"Save failed: {str(e)}"
    return msg


def build_config_ui(init_config):
    save_btn = gr.Button(value="Save configurations")
    message = gr.HTML()
    with gr.Row():
        model = gr.Textbox(
            init_config.model,
            label=MODEL_HELP,
            scale=2,
        )
        device = gr.Radio(AVAILABLE_DEVICES, label="Device", value=init_config.device)

    cache_dir = gr.Textbox(
        init_config.cache_dir,
        label=CACHE_DIR_HELP,
        scale=2,
    )

    listen = gr.Checkbox(init_config.listen, label=f"Listen ({LISTEN_HELP})")
    port = gr.Number(init_config.port, label="Port", precision=0)

    fp32 = gr.Checkbox(init_config.fp32, label=FP32_HELP)
    low_vram = gr.Checkbox(init_config.low_vram, label=LOW_VRAM_HELP)
    nsfw_filter = gr.Checkbox(
        init_config.nsfw_filter, label=f"NSFW Filter ({NSFW_FILTER_HELP})"
    )
    local_files_only = gr.Checkbox(
        init_config.local_files_only, label=f"Local files only ({LOCAL_FILES_ONLY})"
    )

    save_btn.click(
        save_config,
        [
            listen,
            port,
            device,
            model,
            low_vram,
            fp32,
            nsfw_filter,
            cache_dir,
            local_files_only,
        ],
        message,
    )
    return model


def build_gallery_ui(gr_model):
    def img_block(image, label):
        with gr.Column():
            image = Image.open(assets_dir / image)
            gr.Image(image, label=label, shape=(None, 512)).style(height=512)
            with gr.Row():
                copy_btn = gr.Button(f"Use: {label}")
                open_url_btn = gr.Button("HuggingFace Page")
                # gr.HTML(f"<a href='https://huggingface.co/{label}' target='_blank'> HF Page </a>")
        copy_btn.click(lambda it: label, [], gr_model)
        open_url_btn.click(
            lambda: webbrowser.open(f"https://huggingface.co/{label}", new=2)
        )

    with gr.Column():
        for i in list(range(len(all_hf_models)))[::2]:
            with gr.Row():
                with gr.Column():
                    it1 = all_hf_models[i]
                    img_block(it1[0], it1[1])
                if i + 1 < len(all_hf_models):
                    with gr.Column():
                        it2 = all_hf_models[i + 1]
                        img_block(it2[0], it2[1])


def main(config_file: str):
    global _config_file
    _config_file = config_file

    init_config = load_config(config_file)

    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.Tab(label="Configuration"):
                model = build_config_ui(init_config)
            with gr.Tab(label="Model Gallery"):
                build_gallery_ui(model)

    demo.launch(inbrowser=True, show_api=False)
