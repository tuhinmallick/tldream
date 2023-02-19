import os
import re

import setuptools
from pathlib import Path

web_files = Path("tldream/out").glob("**/*")
web_files = [str(it).replace("tldream/", "") for it in web_files]
web_files += ["cldm_v15.yaml"]


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def load_requirements():
    requirements_file_name = "requirements.txt"
    requires = []
    with open(requirements_file_name) as f:
        for line in f:
            if line:
                requires.append(line.strip())
    return requires


PKG = "tldream"
VERSIONFILE = os.path.join(PKG, "_version.py")
verstr = "unknown"
try:
    verstrline = open(VERSIONFILE, "rt").read()
except EnvironmentError:
    pass  # Okay, there is no version file.
else:
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, verstrline, re.M)
    if mo:
        verstr = mo.group(1)
    else:
        print(f"unable to find version in {VERSIONFILE}")
        raise RuntimeError(
            f"if {VERSIONFILE}.py exists, it is required to be well-formed"
        )


# https://setuptools.readthedocs.io/en/latest/setuptools.html#including-data-files
setuptools.setup(
    name="tldream",
    version=verstr,
    author="PanicByte",
    author_email="cwq1913@gmail.com",
    description="A tiny little diffusion drawing app",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sanster/tldream",
    packages=setuptools.find_packages("./"),
    package_data={"tldream": web_files},
    install_requires=load_requirements(),
    python_requires=">=3.7",
    entry_points={"console_scripts": ["tldream=tldream:entry_point"]},
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
