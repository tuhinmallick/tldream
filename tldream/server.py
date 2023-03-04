import asyncio
import io
import os

from starlette.staticfiles import StaticFiles

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, File, Form
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from starlette.responses import FileResponse, StreamingResponse
from torch.hub import download_url_to_file
from typer import Typer, Option

from tldream.socket_manager import SocketManager
from tldream.util import process, load_img, torch_gc, pil_to_bytes, init_pipe
from tldream._version import __version__


current_dir = Path(__file__).parent.absolute().resolve()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
web_static_folder = os.path.join(current_dir, "out")
app.mount("/static", StaticFiles(directory=web_static_folder), name="static")
sio = SocketManager(app=app)

typer_app = Typer(
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

controlled_model = None
lock = threading.Lock()


def diffusion_callback(step: int, timestep: int, latents: torch.FloatTensor):
    asyncio.run(sio.emit("progress", {"step": step}))


@sio.on("join")
async def handle_join(sid, *args, **kwargs):
    logger.info(f"join: {sid}")
    # await socket_manager.emit("lobby", "User joined")


@sio.on("leave")
async def handle_leave(sid, *args, **kwargs):
    logger.info(f"leave: {sid}")
    # await socket_manager.emit("lobby", "User joined")


@app.get("/")
async def root():
    return FileResponse(os.path.join(web_static_folder, "index.html"))


@app.post("/run")
def run(
    image: bytes = File(...),
    steps: int = Form(20),
    sampler: str = Form(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    guidance_scale: float = Form(9.0),
    width: int = Form(512),
    height: int = Form(512),
):
    logger.info(
        {
            "steps": steps,
            "guidance_scale": guidance_scale,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
        }
    )
    origin_image_bytes = image
    image, alpha_channel, exif = load_img(origin_image_bytes, return_exif=True)
    start = time.time()
    with lock:
        try:
            res_rgb_img = process(
                controlled_model,
                sampler,
                image,
                prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                steps=steps,
                width=width,
                height=height,
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
            asyncio.run(sio.emit("finish"))

    bytes_io = io.BytesIO(pil_to_bytes(Image.fromarray(res_rgb_img), "jpeg"))
    response = StreamingResponse(bytes_io)
    return response


@typer_app.command()
def start(
    listen: bool = Option(False, help="If true, start server at 0.0.0.0"),
    port: int = Option(4242),
    device: str = Option("cuda", help="Device to use (cuda, cpu or mps)"),
    model: str = Option(
        "runwayml/stable-diffusion-v1-5",
        help="Any HuggingFace Stable Diffusion model id",
    ),
    low_vram: bool = Option(False, help="Use low vram mode"),
    no_half: bool = Option(False, help="Not use float16 mode"),
):
    from diffusers.utils import DIFFUSERS_CACHE

    logger.info(f"tldream {__version__}")
    logger.info(f"Model cache dir: {DIFFUSERS_CACHE}")

    global controlled_model
    torch_dtype = torch.float32
    if device == "cuda" and not no_half:
        torch_dtype = torch.float16

    # TODO: lazy load model after server started to get download progress
    controlled_model = init_pipe(
        model,
        device,
        torch_dtype=torch_dtype,
        cpu_offload=low_vram and device == "cuda",
    )
    host = "0.0.0.0" if listen else "127.0.0.1"
    uvicorn.run(app, host=host, port=port)


def main():
    typer_app()
