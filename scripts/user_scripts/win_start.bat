@echo off

set PATH=C:\Windows\System32;%PATH%

@call installer\Scripts\activate.bat

@REM SET XDG_CACHE_HOME=
SET CUDA_VISIBLE_DEVICES=0
@call tldream --load-config --config-file %0\..\installer_config.json

PAUSE
