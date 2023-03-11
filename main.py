import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from tldream import entry_point

if __name__ == "__main__":
    entry_point()
