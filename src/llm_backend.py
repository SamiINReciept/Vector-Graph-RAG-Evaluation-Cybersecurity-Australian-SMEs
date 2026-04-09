from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


# ---------- LLM backend ----------

@dataclass
class LLMConfig:
    model_name: str
    dtype: str = "bfloat16"  # "float16" or "bfloat16"
    device_map: str = "auto"
    max_new_tokens: int = 128
    temperature: float = 0.1
    top_p: float = 1.0

    # new guardrail knobs
    max_time: float | None = 6.0           # abort generation after N seconds
    no_repeat_ngram_size: int | None = 6   # reduce loops
    repetition_penalty: float | None = 1.05


class LLMBackend:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

        # Choose torch dtype
        if cfg.dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif cfg.dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        # Some models need this
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch_dtype,
            device_map=cfg.device_map,
        )

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """
        Safe, constrained generation with guardrails to prevent rambling,
        repetition, and runaway generation time.
        """

        # Use config value if caller does not specify
        max_new_tokens = max_new_tokens or self.cfg.max_new_tokens

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        # --- Guardrails ---
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=self.cfg.temperature > 0,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Add guardrails only if configured
        if hasattr(self.cfg, "max_time") and self.cfg.max_time is not None:
            gen_kwargs["max_time"] = self.cfg.max_time

        if hasattr(self.cfg, "no_repeat_ngram_size") and self.cfg.no_repeat_ngram_size is not None:
            gen_kwargs["no_repeat_ngram_size"] = self.cfg.no_repeat_ngram_size

        if hasattr(self.cfg, "repetition_penalty") and self.cfg.repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = self.cfg.repetition_penalty

        # --- Generate ---
        output_ids = self.model.generate(
            **inputs,
            **gen_kwargs,
        )

        # Extract *new* tokens only
        gen_start = inputs["input_ids"].shape[1]
        generated_tokens = output_ids[0, gen_start:]

        # Decode safely
        text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return text.strip()



# ---------- Embedding backend ----------

@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-m3"
    device: str = "cuda"
    batch_size: int = 16


class EmbeddingBackend:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model_name, device=cfg.device)

    def embed(self, texts: List[str]) -> List[list]:
        """
        Returns a list of embedding vectors (as Python lists for easy JSON/Chroma).
        """
        vectors = self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()
