DEFAULT_LISTEN = False
LISTEN_HELP = "If true, start server at 0.0.0.0"

DEFAULT_PORT = 4242

DEFAULT_DEVICE = "cuda"
AVAILABLE_DEVICES = ["cuda", "cpu", "mps"]

DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"
MODEL_HELP = "Any HuggingFace Stable Diffusion model id. Or local ckpt/safetensors path"

DEFAULT_LOW_VRAM = False
LOW_VRAM_HELP = "Use low vram mode"

DEFAULT_FP32 = False
FP32_HELP = "Use float32 mode (For NVIDIA 16xx GPU)"

DEFAULT_NSFW_FILTER = True
NSFW_FILTER_HELP = "Enable nsfw filter"

CACHE_DIR_HELP = (
    "Model cache directory, by default model downloaded to ~/.cache/huggingface/hub"
)

DEFAULT_LOCAL_FILES_ONLY = False
LOCAL_FILES_ONLY = (
    "Not connect to HuggingFace server, add this flag if model has been downloaded"
)


all_hf_models = [
    ("stable_diffusion_12_1.png", "runwayml/stable-diffusion-v1-5"),
    ("anything-v3.0.png", "Linaqruf/anything-v3.0"),
    ("dreamshaper.jpg", "Lykon/DreamShaper"),
    ("waifu.png", "hakurei/waifu-diffusion-v1-4"),
    ("artstation.jpg", "hakurei/artstation-diffusion"),
    ("pokemon.png", "lambdalabs/sd-pokemon-diffusers"),
    ("modi-samples-02s.jpg", "nitrosocke/mo-di-diffusion", "modern disney style"),
    ("epic.png", "johnslegers/epic-diffusion"),
    ("inkpunk.jpg", "Envvi/Inkpunk-Diffusion", "nvinkpunk"),
    ("ghibli-diffusion-samples-02s.jpg", "nitrosocke/Ghibli-Diffusion", "ghibli style"),
    ("Van-Gogh.jpeg", "dallinmackay/Van-Gogh-diffusion", "lvngvncnt"),
    ("robo.png", "nousr/robo-diffusion", "nousr robot"),
    ("vectorartz.png", "coder119/Vectorartz_Diffusion", "vectorartz"),
    ("guofeng.jpg", "xiaolxl/GuoFeng3"),
    ("knollingcase.jpg", "Aybeeceedee/knollingcase", "knollingcase"),
    ("snthwve.jpg", "ItsJayQz/SynthwavePunk-v2", "snthwve style"),
    ("16bitscene.png", "PublicPrompts/All-In-One-Pixel-Model", "16bitscene"),
    ("Complex-Lineart.png", "Conflictx/Complex-Lineart", "ComplexLA style"),
]
all_hf_models_map = {it[1]: it for it in all_hf_models}


def merge_prompt_with_model_keywords(model_id, prompt):
    if model_id not in all_hf_models_map:
        return prompt
    if len(all_hf_models_map[model_id]) == 2:
        return prompt

    keyword = all_hf_models_map[model_id][2]
    return f"{keyword}, {prompt}"
