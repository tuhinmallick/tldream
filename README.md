<p align="center">
<div align="center">
  <img height=96 src="https://user-images.githubusercontent.com/3998421/222906944-40a06042-e6d5-48f5-905e-b635469e7005.svg"/> 
  <h1>tldream</h1>
</div>
</p>

<p align="center">A tiny little diffusion drawing app</p>

<p align="center">
  <a href="https://github.com/Sanster/tldream">
    <img alt="total download" src="https://pepy.tech/badge/tldream" />
  </a>
  <a href="https://pypi.org/project/tldream/">
    <img alt="version" src="https://img.shields.io/pypi/v/tldream" />
  </a>
   <a href="https://colab.research.google.com/drive/1m1qBE3N8VWDqE__8zRP8hvEE0JPzk7rp?usp=sharing">
    <img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" />
  </a>
</p>


https://user-images.githubusercontent.com/3998421/223580181-6375fc76-414a-4837-b7ab-4fb07509afd3.mp4


### Quick Start

```bash
pip install tldream
tldream --model runwayml/stable-diffusion-v1-5 --device cuda
```

### Command line arguments

* `--listen / --no-listen`: If true, start server at 0.0.0.0  [default: no-listen]
* `--port INTEGER`: [default: 4242]
* `--device TEXT`: Device to use (cuda, cpu or mps)  [default: cuda]
* `--model TEXT`: Any HuggingFace Stable Diffusion model id. Or local ckpt/safetensors path  [default: runwayml/stable-diffusion-v1-5]
* `--low-vram / --no-low-vram`: Use low vram mode  [default: no-low-vram]
* `--fp32 / --no-fp32`: Use float32 mode  [default: no-fp32]
* `--nsfw-filter / --no-nsfw-filter`: [default: nsfw-filter]
* `--cache-dir TEXT`: Model cache directory, by default model downloaded to ~/.cache/huggingface/hub
* `--local-files-only / --no-local-files-only`: Not connect to HuggingFace server, add this flag if model has been downloaded  [default: no-local-files-only]