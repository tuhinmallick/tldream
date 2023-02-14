import imghdr
import io
import os
import time
from pathlib import Path

import torch
from PIL import Image
from flask import (
    Flask,
    request,
    send_file,
    cli,
    make_response,
)
from flask_cors import CORS
from loguru import logger
from typer import Typer, Option

from cldm.model import create_model, load_state_dict
from ldm.tldream_util import process, load_img, torch_gc, pil_to_bytes

app = Flask(__name__, static_folder=os.path.join("app/build", "static"))
app.config["JSON_AS_ASCII"] = False
CORS(app, expose_headers=["Content-Disposition"])

# Disable ability for Flask to display warning about using a development server in a production environment.
# https://gist.github.com/jerblack/735b9953ba1ab6234abb43174210d356
cli.show_server_banner = lambda *_: None

current_dir = Path(__file__).parent.absolute().resolve()
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
    cfg_path = current_dir / "models" / "cldm_v15.yaml"
    model = create_model(str(cfg_path), device).cpu()
    model.load_state_dict(load_state_dict(model_path, location="cpu"))
    model = model.to(device)
    return model


@app.route("/run", methods=["POST"])
def process():
    input = request.files
    # form = request.form

    origin_image_bytes = input["image"].read()
    image, alpha_channel, exif = load_img(origin_image_bytes, return_exif=True)
    start = time.time()
    try:
        rgb_images = process(
            controlled_model,
            _device,
            image,
            "a turtle in river",
            "",
            low_vram=_low_vram,
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

    res_rgb_img = rgb_images[0]
    bytes_io = io.BytesIO(pil_to_bytes(Image.fromarray(res_rgb_img), "jpeg"))

    response = make_response(
        send_file(
            bytes_io,
            mimetype=f"image/jpeg",
        )
    )
    return response


@typer_app.command()
def main(
    host: str = Option("127.0.0.1"),
    port: int = Option(4242),
    device: str = Option("mps", help="Device to use (cuda, cpu or mps)"),
    model: Path = Option(
        "/Users/cwq/code/github/ControlNet/models/control_any3_better_scribble.pth",
        help="Path to model",
    ),
    low_vram: bool = Option(True, help="Use low vram mode"),
):
    global controlled_model
    global _device
    global _low_vram
    controlled_model = init_model(model, device)
    _device = device
    _low_vram = low_vram
    app.run(host=host, port=port)


if __name__ == "__main__":
    typer_app()
