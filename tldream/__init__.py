import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import warnings

warnings.simplefilter("ignore", UserWarning)

from typer import Option, Typer

typer_app = Typer(
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


@typer_app.command()
def start(
    listen: bool = Option(False, help="If true, start server at 0.0.0.0"),
    port: int = Option(4242),
    device: str = Option("cuda", help="Device to use (cuda, cpu or mps)"),
    model: str = Option(
        "runwayml/stable-diffusion-v1-5",
        help="Any HuggingFace Stable Diffusion model id. Or local ckpt/safetensors path",
    ),
    low_vram: bool = Option(False, help="Use low vram mode"),
    fp32: bool = Option(False, help="Use float32 mode"),
    nsfw_filter: bool = Option(True),
    local_files_only: bool = Option(
        False,
        help="Not connect to HuggingFace server, add this flag if model has been downloaded",
    ),
):
    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"

    from .server import main

    main(
        listen=listen,
        port=port,
        device=device,
        model=model,
        low_vram=low_vram,
        fp32=fp32,
        nsfw_filter=nsfw_filter,
    )


def entry_point():
    typer_app()
