import asyncio
import imghdr
import io
import os
import threading
from urllib.parse import urlparse

import time
from enum import Enum
from pathlib import Path
from typing import List, Dict

import torch
from torch.hub import download_url_to_file
import uvicorn
from PIL import Image
from fastapi import FastAPI, File, Form
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel
from starlette.responses import FileResponse, StreamingResponse
from starlette.websockets import WebSocketDisconnect, WebSocket
from typer import Typer, Option

from tldream.cldm.hack import disable_verbosity, enable_sliced_attention
from tldream.cldm.model import create_model, load_state_dict
from tldream.util import process, load_img, torch_gc, pil_to_bytes

disable_verbosity()
current_dir = Path(__file__).parent.absolute().resolve()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
web_static_folder = os.path.join("app/build", "static")
# app.mount("/static", StaticFiles(directory=web_static_folder), name="static")

typer_app = Typer(add_completion=False, pretty_exceptions_show_locals=False)

controlled_model = None
_device = "cpu"
_low_vram = False


def get_image_ext(img_bytes):
    w = imghdr.what("", img_bytes)
    if w is None:
        w = "jpeg"
    return w


def init_model(model_path, device):
    cfg_path = current_dir / "cldm_v15.yaml"
    model = create_model(str(cfg_path), device).cpu()
    model.load_state_dict(load_state_dict(model_path, location="cpu"))
    model = model.to(device)
    return model


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, data: Dict):
        for connection in self.active_connections:
            await connection.send_json(data)


manager = ConnectionManager()


class RunningState(str, Enum):
    NONE = "none"
    DIFFUSION = "diffusion"
    DOWNLOADING = "downloading"


class AppState(BaseModel):
    running_state: RunningState = RunningState.NONE
    step_i: int = 0
    steps: int = 0
    model_size: int = 0
    model_downloaded_size: int = 0

    async def reset(self):
        self.running_state = RunningState.NONE
        self.step_i = 0
        self.steps = 0
        self.model_size = 0
        self.model_downloaded_size = 0
        await manager.broadcast(self.dict())


state = AppState()
lock = threading.Lock()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def diffusion_callback(step_i):
    print(f"step_i: {step_i}")
    await manager.broadcast(state.dict())


@app.get("/")
async def main():
    return FileResponse(os.path.join(web_static_folder, "index.html"))


@app.post("/run")
async def run(
        image: bytes = File(...),
        steps: int = Form(20),
        prompt: str = Form(...),
        negative_prompt: str = Form(""),
        guidance_scale: float = Form(9.0),
):
    logger.info({
        "steps": steps,
        "guidance_scale": guidance_scale,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
    })
    origin_image_bytes = image
    image, alpha_channel, exif = load_img(origin_image_bytes, return_exif=True)
    start = time.time()
    state.steps = steps
    with lock:
        state.running_state = RunningState.DIFFUSION
        try:
            rgb_images = await process(
                controlled_model,
                _device,
                image,
                prompt,
                negative_prompt=negative_prompt,
                scale=guidance_scale,
                ddim_steps=steps,
                low_vram=_low_vram,
                callback=diffusion_callback,
            )

        except RuntimeError as e:
            torch.cuda.empty_cache()
            if "CUDA out of memory. " in str(e):
                # NOTE: the string may change?
                return "CUDA out of memory", 500
            else:
                logger.exception(e)
                return "Internal Server Error", 500
        finally:
            logger.info(f"process time: {(time.time() - start) * 1000}ms")
            torch_gc()
            await state.reset()

    res_rgb_img = rgb_images[0]
    bytes_io = io.BytesIO(pil_to_bytes(Image.fromarray(res_rgb_img), "jpeg"))
    response = StreamingResponse(bytes_io)
    return response


PRE_DEFINE_MODELS = {
    "sd15": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_scribble.pth",
    "any3": "/Users/cwq/code/github/ControlNet/models/control_any3_better_scribble.pth"
}


def get_model_path(model_name, save_dir):
    if os.path.exists(model_name):
        return model_name
    if model_name in PRE_DEFINE_MODELS:
        model_p = PRE_DEFINE_MODELS[model_name]
        if os.path.exists(model_p):
            return model_p

        url = model_p
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
        dst_p = str(save_dir / filename)
        if os.path.exists(dst_p):
            logger.info(f"Loading model from: {dst_p}")
            return dst_p

        logger.info(f"Downloading {filename} to {save_dir.absolute()}")
        download_url_to_file(model_p, dst_p, progress=True)
        return dst_p
    raise ValueError(f"model {model_name} is invalid, available models: {list(PRE_DEFINE_MODELS.keys())}")


@typer_app.command()
def start(
        host: str = Option("127.0.0.1"),
        port: int = Option(4242),
        device: str = Option("mps", help="Device to use (cuda, cpu or mps)"),
        model_id: str = Option(
            "any3", help="Local path to model or model name(will downloaded when start)"
        ),
        low_vram: bool = Option(True, help="Use low vram mode"),
        model_dir: Path = Option("./models", help="Directory to store models"),
):
    if not model_dir.exists():
        logger.info(f"create model dir: {model_dir}")
        model_dir.mkdir(parents=True, exist_ok=True)
    global controlled_model
    global _device
    global _low_vram
    if low_vram:
        enable_sliced_attention()

    # TODO: lazy load model after server started to get download progress
    model_path = get_model_path(model_id, model_dir)
    controlled_model = init_model(model_path, device)
    _device = device
    _low_vram = low_vram
    uvicorn.run(app, host=host, port=port)


def main():
    typer_app()
