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
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────╮
│ --listen       --no-listen                     If true, start server at 0.0.0.0 [default: no-listen] │
│ --port                          INTEGER        [default: 4242]                                       │
│ --device                        TEXT           Device to use (cuda, cpu or mps) [default: cuda]      │
│ --model                         TEXT           Local path to model or model download link or model   │
│                                                name(sd15, any3)                                      │
│                                                [default: sd15]                                       │
│ --low-vram     --no-low-vram                   Use low vram mode [default: low-vram]                 │
│ --no-half      --no-no-half                    Not use float16 mode [default: no-no-half]            │
│ --model-dir                     PATH           Directory to store models [default: ./models]         │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
