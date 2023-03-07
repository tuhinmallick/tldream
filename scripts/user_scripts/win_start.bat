@echo off

set PATH=C:\Windows\System32;%PATH%

@call installer\Scripts\activate.bat

@REM SET XDG_CACHE_HOME=
SET CUDA_VISIBLE_DEVICES=0
@call tldream --model runwayml/stable-diffusion-v1-5 --device cuda --port 4242

PAUSE