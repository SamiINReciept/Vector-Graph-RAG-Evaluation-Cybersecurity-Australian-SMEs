from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings

from .config_loader import load_config
from .llm_backend import LLMBackend, LLMConfig, EmbeddingBackend, EmbeddingConfig
from .ingest import Chunk


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_name: str
    text: str
    score: float


class VectorRAG:
    def __init__(self, cfg_path: str = "config/base.yaml"):
        self.cfg = load_config(cfg_path)

        # LLM backend
        llm_cfg = LLMConfig(**self.cfg["llm"])
        self.llm = LLMBackend(llm_cfg)

        # Embedding backend
        emb_cfg = EmbeddingConfig(**self.cfg["embedding"])
        self.embedder = EmbeddingBackend(emb_cfg)

        # ChromaDB client & collection
        vs_cfg = self.cfg["vector_store"]
        persist_dir = vs_cfg["persist_dir"]
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection_name = "corpus"
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

        # Corpus path
        self.corpus_cache = Path(self.cfg["data"]["corpus_cache"])

    def build_index(self, force_rebuild: bool = False):
        """
        Build or rebuild the vector index from corpus.pkl.
        """
        if not self.corpus_cache.exists():
            raise FileNotFoundError(
                f"Corpus cache not found at {self.corpus_cache}. "
                "Run `python -m src.ingest` first."
            )

        if not force_rebuild and self.collection.count() > 0:
            print(
                f"Collection '{self.collection_name}' already has "
                f"{self.collection.count()} records. Skipping rebuild."
            )
            return

        # Optional: clear existing collection
        if force_rebuild:
            print(f"Clearing existing collection '{self.collection_name}'")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)

        print(f"Loading corpus from {self.corpus_cache}")
        with self.corpus_cache.open("rb") as f:
            chunks: List[Chunk] = pickle.load(f)

        texts = [c.text for c in chunks]
        ids = [c.id for c in chunks]
        metadatas = [{"doc_name": c.doc_name} for c in chunks]

        print(f"Embedding {len(texts)} chunks...")
        embeddings = self.embedder.embed(texts)

        print(f"Adding to Chroma collection '{self.collection_name}'...")
        # Chroma expects lists
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print("Index build complete.")

    def retrieve(self, question: str, top_k: int = 5) -> List[RetrievedChunk]:
        """
        Embed the question, retrieve top-k chunks from Chroma.
        """
        query_emb = self.embedder.embed([question])[0]

        res = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: List[RetrievedChunk] = []
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        distances = res["distances"][0]

        for cid, doc, meta, dist in zip(ids, docs, metas, distances):
            retrieved.append(
                RetrievedChunk(
                    chunk_id=cid,
                    doc_name=meta.get("doc_name", ""),
                    text=doc,
                    score=float(dist),
                )
            )

        return retrieved

    def build_prompt(self, question: str, contexts: List[RetrievedChunk]) -> str:
        """
        Build a strict RAG prompt to minimize hallucinations.
        """
        context_strs = []
        for i, c in enumerate(contexts, start=1):
            context_strs.append(f"[Chunk {i} from {c.doc_name}]\n{c.text}")

        context_block = "\n\n".join(context_strs)

        prompt = f"""You are a helpful assistant for a question-answering task.

RULES:
- Answer ONLY using the given context.
- Give a SINGLE, concise answer in 1–3 sentences.
- Do NOT explain your reasoning.
- Do NOT mention chunks, documents, or context.
- Do NOT say things like "I do not have enough information" more than once.
- Do NOT write notes, self-corrections, or alternative versions.
- Do NOT ask the user if the answer is acceptable.
- Answer in English.
Your entire reply must be just the final answer.

CONTEXT:
{context_block}

QUESTION:
{question}

ANSWER:
"""
        return prompt

    def answer(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Full pipeline: retrieve, then generate answer with LLM.
        Returns answer, contexts, and latencies.
        """
        t0 = time.time()
        retrieved = self.retrieve(question, top_k=top_k)
        t1 = time.time()

        prompt = self.build_prompt(question, retrieved)
        answer = self.llm.generate(prompt)
        t2 = time.time()

        return {
            "question": question,
            "answer": answer,
            "contexts": retrieved,
            "latency_retrieval": t1 - t0,
            "latency_generation": t2 - t1,
            "latency_total": t2 - t0,
        }
    
    def get_contexts(self, question: str, top_k: int = 5) -> list[str]:
        """
        Return retrieved chunk texts for this question.
        Used by RAGAS as 'contexts'.
        """
        # If you already have a `retrieve()` that returns the docs, you can reuse it.
        # Otherwise, this uses the Chroma collection directly.

        emb = self.embedder.embed([question])[0]
        res = self.collection.query(
            query_embeddings=[emb],
            n_results=top_k,
            include=["documents"],
        )
        if not res or "documents" not in res or not res["documents"]:
            return []
        docs = res["documents"][0]  # first query
        # ensure string list
        return [str(d) for d in docs]



if __name__ == "__main__":
    # Small manual test
    rag = VectorRAG("config/vector_qwen.yaml") #temporary hard coding of config, will be passed as argparse later
    rag.build_index(force_rebuild=False)
    q = "Summarize the main topic of these documents."
    result = rag.answer(q, top_k=5)
    print("Q:", result["question"])
    print("A:", result["answer"])
    print("Used chunks:", len(result["contexts"]))


# python -m src.vector_rag --config config/vector_llama.yaml 
