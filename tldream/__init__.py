import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from loguru import logger

import warnings

warnings.simplefilter("ignore", UserWarning)

from typer import Option, Typer
from .const import *

typer_app = Typer(
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


@typer_app.command()
def start(
    listen: bool = Option(DEFAULT_LISTEN, help=LISTEN_HELP),
    port: int = Option(DEFAULT_PORT),
    device: str = Option(DEFAULT_DEVICE, help="Device to use (cuda, cpu or mps)"),
    model: str = Option(DEFAULT_MODEL, help=MODEL_HELP),
    low_vram: bool = Option(DEFAULT_LOW_VRAM, help=LOW_VRAM_HELP),
    fp32: bool = Option(DEFAULT_FP32, help=FP32_HELP),
    nsfw_filter: bool = Option(DEFAULT_NSFW_FILTER, help=NSFW_FILTER_HELP),
    cache_dir: str = Option(None, help=CACHE_DIR_HELP),
    local_files_only: bool = Option(DEFAULT_LOCAL_FILES_ONLY, help=LOCAL_FILES_ONLY),
    start_web_config: bool = Option(False, help="Start web config server"),
    load_config: bool = Option(False, help="Load config from file"),
    config_file: str = Option("config.json", help="Config file path"),
):
    if start_web_config:
        from .web_config import main

        main(config_file)
        return

    if load_config:
        if not os.path.exists(config_file):
            logger.error(f"Config file {config_file} not found")
            exit(-1)

        from .web_config import load_config as load_config_from_file

        config = load_config_from_file(config_file)
        listen = config.listen
        port = config.port
        device = config.device
        model = config.model
        low_vram = config.low_vram
        fp32 = config.fp32
        nsfw_filter = config.nsfw_filter
        cache_dir = config.cache_dir
        local_files_only = config.local_files_only

    if device not in AVAILABLE_DEVICES:
        logger.error(
            f"Device {device} is not supported, use one of {AVAILABLE_DEVICES}"
        )
        exit(-1)

    from . import shared

    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if cache_dir:
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

    from diffusers.utils import is_xformers_available

    shared.use_xformers = device == "cuda" and is_xformers_available()

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
