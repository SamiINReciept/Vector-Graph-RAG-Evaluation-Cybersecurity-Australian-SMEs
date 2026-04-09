from src.config_loader import load_config
from src.llm_backend import LLMBackend, LLMConfig, EmbeddingBackend, EmbeddingConfig

cfg = load_config("config/base.yaml")

llm_cfg = LLMConfig(**cfg["llm"])
llm = LLMBackend(llm_cfg)

print("LLM loaded")

resp = llm.generate("You are a concise assistant. Say hello in one sentence.")
print("Response:", resp)

emb_cfg = EmbeddingConfig(**cfg["embedding"])
emb = EmbeddingBackend(emb_cfg)
vecs = emb.embed(["additive manufacturing", "laser power"])
print("Got embeddings shape:", len(vecs), "x", len(vecs[0]))
