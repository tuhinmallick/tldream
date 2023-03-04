<h1 align="center">tldream</h1>
<p align="center">A tiny little diffusion drawing app</p>

<p align="center">
  <a href="https://github.com/Sanster/tldream">
    <img alt="total download" src="https://pepy.tech/badge/tldream" />
  </a>
  <a href="https://pypi.org/project/tldream/">
    <img alt="version" src="https://img.shields.io/pypi/v/tldream" />
  </a>
</p>

![A screenshot of the tldream web app](https://github.com/Sanster/tldream-frontend/blob/tldream/assets/tldream.png)

### Quick Start

```bash
pip install tldream
tldream --device cuda
```

### Command line arguments

```bash
╭─ Options ────────────────────────────────────────────────────────────────────────────────╮
│ --listen         --no-listen                  If true, start server at 0.0.0.0           │
│                                               [default: no-listen]                       │
│ --port                               INTEGER  [default: 4242]                            │
│ --device                             TEXT     Device to use (cuda, cpu or mps)           │
│                                               [default: cuda]                            │
│ --model                              TEXT     Any HuggingFace Stable Diffusion model id  │
│                                               [default: runwayml/stable-diffusion-v1-5]  │
│ --low-vram       --no-low-vram                Use low vram mode [default: no-low-vram]   │
│ --fp32           --no-fp32                    Use float32 mode [default: no-fp32]        │
│ --nsfw-filter    --no-nsfw-filter             [default: nsfw-filter]                     │
│ --help                                        Show this message and exit.                │
╰──────────────────────────────────────────────────────────────────────────────────────────╯
```
