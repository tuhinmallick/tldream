from functools import lru_cache

import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

LANG_MODELS = {
    "zh": "Helsinki-NLP/opus-mt-zh-en",
    "ja": "Helsinki-NLP/opus-mt-ja-en",
    "ko": "Helsinki-NLP/opus-mt-ko-en",
    "ru": "Helsinki-NLP/opus-mt-ru-en",
}


class TranslationModel:
    def __init__(self, lang: str):
        super().__init__()
        self.lang = lang
        if lang != "en":
            self.max_length = 77
            self.device = torch.device("cpu")
            model_name = LANG_MODELS[lang]
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"loading translation model {model_name}...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, torch_dtype=torch.float32
            )
            self.model = self.model.to(self.device)

    @lru_cache(maxsize=512)
    def __call__(self, prompt: str) -> str:
        if not prompt or self.lang == "en":
            return prompt

        inputs = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        outputs = self.model.generate(
            inputs["input_ids"].to(self.device),
            max_length=self.max_length,
            num_beams=4,
            early_stopping=True,
        )
        new_prompt = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return new_prompt
