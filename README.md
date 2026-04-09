# Vector RAG vs Graph RAG for Cybersecurity QA on Australian SMBs

A comparative study of **Vector-based Retrieval-Augmented Generation (RAG)** and **Graph-based RAG** pipelines for answering cybersecurity questions in the context of Australian Small & Medium Businesses (SMBs). The project benchmarks three open-source LLMs across both retrieval paradigms and evaluates them with a comprehensive suite of metrics.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Vector RAG Pipeline](#vector-rag-pipeline)
  - [Graph RAG Pipeline](#graph-rag-pipeline)
- [Project Structure](#project-structure)
- [Knowledge Graph Schema](#knowledge-graph-schema)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Reproduction Guide](#reproduction-guide)
  - [Step 1: Ingest PDFs into Chunks](#step-1-ingest-pdfs-into-chunks)
  - [Step 2: Build the Vector Index](#step-2-build-the-vector-index)
  - [Step 3: Build the Knowledge Graph](#step-3-build-the-knowledge-graph)
  - [Step 4: Run the Evaluation Pipelines](#step-4-run-the-evaluation-pipelines)
  - [Step 5: Compute Post-hoc CPU Metrics](#step-5-compute-post-hoc-cpu-metrics)
  - [Step 6: Produce Final Metrics](#step-6-produce-final-metrics)
  - [Step 7: Visualize the Knowledge Graph](#step-7-visualize-the-knowledge-graph)
- [Configuration](#configuration)
- [License](#license)

---

## Overview

This project investigates how two fundamentally different retrieval strategies — dense vector similarity search and structured knowledge graph traversal — perform for domain-specific question answering in the cybersecurity + Australian SMB space.

**Key research questions:**
- Does structured graph-based retrieval outperform vector similarity search for domain-specific cybersecurity QA?
- How do different open-source LLMs compare when paired with each retrieval strategy?
- What are the trade-offs in hallucination, completeness, relevance, and latency between Vector RAG and Graph RAG?

**Domain:** The corpus consists of PDF documents covering Australian government cybersecurity guidelines, academic papers on SMB cyber risk, policy frameworks (e.g., ACSC Small Business Guide, Australian Cyber Security Strategy 2023–2030, ISM, Privacy Act), threat landscape reports, and research on RAG/LLM techniques for cybersecurity.

---

## Architecture

### Vector RAG Pipeline

```
PDF Documents
     │
     ▼
[Text Extraction] ──► PyMuPDF (fitz)
     │
     ▼
[Chunking] ──► RecursiveCharacterTextSplitter (1000 chars, 150 overlap)
     │
     ▼
[Embedding] ──► BAAI/bge-m3 (SentenceTransformer)
     │
     ▼
[Vector Store] ──► ChromaDB (persistent, on-disk)
     │
     ▼
[Retrieval] ──► Top-k cosine similarity search (k=5)
     │
     ▼
[Generation] ──► LLM generates answer from retrieved chunks
```

### Graph RAG Pipeline

```
PDF Documents
     │
     ├──► [LLM Triple Extraction] ──► Structured (head, relation, tail) triples
     │         per chunk with constrained cybersecurity schema
     │
     └──► [Manual Triples] ──► NotebookLM-curated triples (triples_1/2/3.json)
              │
              ▼
         [Merge & Import] ──► Neo4j Knowledge Graph
                                    │
                                    ▼
                           [Intent Extraction] ──► LLM maps question to
                                    │               entity types + relations + keywords
                                    ▼
                           [Cypher Template] ──► Deterministic parameterized query
                                    │               (NO natural-language-to-Cypher)
                                    ▼
                           [Neo4j Execution] ──► Retrieved graph facts
                                    │
                                    ▼
                           [Generation] ──► LLM generates answer from graph facts
```

The Graph RAG pipeline deliberately avoids LLM-generated Cypher. Instead, it uses the LLM only for **semantic intent extraction** (mapping free-text questions to structured entity types, relation types, and keywords), then builds Cypher queries from **deterministic templates**. This approach is more reliable and reproducible than NL-to-Cypher.

---

## Project Structure

```
├── config/
│   ├── base.yaml                 # Base configuration (shared defaults)
│   ├── vector_llama.yaml         # Config for LLaMA 3.1 8B experiments
│   ├── vector_mistral.yaml       # Config for Mistral 7B v0.3 experiments
│   └── vector_qwen.yaml          # Config for Qwen3 8B experiments
│
├── data/
│   ├── pdfs/                     # Source PDF documents (not included in repo)
│   ├── vector_store/             # ChromaDB persistent storage
│   ├── test_qas.csv              # Full test set (60 QA pairs with supporting triples)
│   └── test_qas_60.csv           # Alternate test set variant
│
├── src/
│   ├── __init__.py
│   ├── config_loader.py          # YAML config loading with run metadata
│   ├── llm_backend.py            # LLM (HuggingFace) and Embedding (SentenceTransformer) backends
│   ├── ingest.py                 # PDF text extraction and chunking pipeline
│   ├── text_utils.py             # Text cleaning utilities for evaluation
│   ├── vector_rag.py             # Full Vector RAG pipeline (index, retrieve, generate)
│   ├── graph_build.py            # LLM-based KG triple extraction from chunks → Neo4j
│   ├── graph_import_triples.py   # Import manually curated triples (JSON) → Neo4j
│   ├── merge_triples.py          # Merge multiple triple JSON files into one
│   ├── graph_rag.py              # Full Graph RAG pipeline (intent → Cypher → answer)
│   ├── eval_metrics.py           # Core metrics: METEOR, BERTScore, Cosine Similarity
│   ├── evaluate_pipelines.py     # Main evaluation harness: runs all 6 pipelines
│   ├── evaluate_ragas_light.py   # RAGAS-based evaluation (faithfulness, relevancy, recall)
│   ├── posthoc_cpu_metrics.py    # CPU-only post-hoc metrics (hallucination, completeness, etc.)
│   ├── recompute_metrics_from_raw.py  # Recompute summary metrics from raw CSVs
│   ├── finalized_metrics.py      # Merge GPU + CPU metrics into final_metrics.csv
│   ├── kg_viz.py                 # Interactive KG visualization (pyvis → HTML)
│   ├── kg_hairball.py            # Static KG "hairball" + zoomed subgraph visualizations
│   ├── kg_composite.py           # Composite image (hairball + zoom side-by-side)
│   └── neo4j_test.py             # Simple Neo4j connectivity test
│
├── results/                      # Evaluation output CSVs
│   ├── summary_metrics.csv       # GPU-computed aggregate metrics per pipeline
│   ├── summary_cpu_metrics.csv   # CPU-computed aggregate metrics per pipeline
│   ├── final_metrics.csv         # Merged final metrics for all pipelines
│   ├── {pipeline}_raw.csv        # Per-question raw results for each pipeline
│   └── {pipeline}_with_cpu_metrics.csv  # Enriched with post-hoc CPU metrics
│
├── triples_1.json                # Manually curated triples batch 1 (NotebookLM)
├── triples_2.json                # Manually curated triples batch 2
├── triples_3.json                # Manually curated triples batch 3
├── triples.json                  # Merged triples file (all batches combined)
│
├── lib/                          # Frontend libraries for interactive KG visualization
│   ├── vis-9.1.2/                # vis-network JS library
│   ├── tom-select/               # Tom Select dropdown library
│   └── bindings/                 # Utility JS bindings
│
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Knowledge Graph Schema

The knowledge graph uses a constrained cybersecurity/SMB schema designed around the domain.

### Entity Types (Node labels)

| Entity Type | Description |
|---|---|
| `Organization` | Businesses, SMEs, industry bodies, universities, insurers |
| `GovernmentBody` | Government agencies and regulators (e.g., ACSC, ASD, OAIC, AFP) |
| `PolicyOrRegulation` | Laws, strategies, frameworks, guidelines (e.g., Privacy Act 1988, NIST CSF) |
| `ThreatOrAttack` | Phishing, ransomware, BEC, scams, malware, data breaches |
| `Vulnerability` | Lack of funds, weak passwords, unpatched software, skill gaps |
| `ControlOrMeasure` | MFA, backups, training, patching, cyber insurance, policies |
| `AssetOrData` | Customer data, financial records, IT systems, cloud services |
| `ImpactOrOutcome` | Financial loss, downtime, reputational damage, insolvency |
| `Other` | Anything that does not clearly fit above |

### Relation Types (Edge types)

| Relation | Meaning |
|---|---|
| `TARGETS` | A threat/attack targets an organization or asset |
| `MITIGATES` | A control/measure reduces likelihood or impact of a threat |
| `CAUSES` | A vulnerability or threat directly causes a negative outcome |
| `LEADS_TO` | An event/condition leads to a consequence |
| `REQUIRES` | A policy/regulation requires a control or action |
| `APPLIES_TO` | A policy/guideline applies to a particular scope |
| `RESPONSIBLE_FOR` | An organization/role is responsible for an action |
| `LOCATED_IN` | An entity is located in or scoped to a region |
| `RELATED_TO` | A general conceptual connection (fallback) |

All nodes are stored with the label `:Entity` and a `type` property. All edges use the label `:RELATION` with a `type` property (one of the above).

---

## Models Used

| Component | Model | Details |
|---|---|---|
| **LLM 1** | `meta-llama/Llama-3.1-8B-Instruct` | 8B parameter instruction-tuned LLaMA 3.1 |
| **LLM 2** | `mistralai/Mistral-7B-Instruct-v0.3` | 7B parameter Mistral v0.3 |
| **LLM 3** | `Qwen/Qwen3-8B` | 8B parameter Qwen 3 |
| **Embedding** | `BAAI/bge-m3` | Multi-lingual, multi-granularity embedding via SentenceTransformers |
| **NLI (hallucination)** | `cross-encoder/nli-deberta-v3-small` | DeBERTa cross-encoder for NLI-based hallucination detection |
| **Relevance** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | MS MARCO-trained cross-encoder for relevance scoring |

All LLMs are loaded via HuggingFace Transformers with `bfloat16` precision and `device_map="auto"`. Generation guardrails include `max_time=6s`, `no_repeat_ngram_size=6`, and `repetition_penalty=1.1`.

---

## Evaluation Metrics

### GPU-computed Metrics (via `eval_metrics.py`)

| Metric | Description |
|---|---|
| **METEOR** | Token-level overlap with stemming, synonyms, and word order |
| **BERTScore F1** | Contextual embedding similarity between generated and reference answers |
| **Cosine Similarity** | Embedding-based semantic similarity using BAAI/bge-m3 |

### CPU-computed Post-hoc Metrics (via `posthoc_cpu_metrics.py`)

| Metric | Description | Direction |
|---|---|---|
| **Completeness** | ROUGE-L recall of gold answer content in the generated answer | Higher = better |
| **Hallucination** | NLI contradiction probability (context as premise, answer as hypothesis) | Lower = better |
| **Irrelevance** | 1 − answer relevance score (cross-encoder query-answer similarity) | Lower = better |
| **Faithfulness** | NLI entailment probability (answer grounded in retrieved context) | Higher = better |
| **Context Relevance** | Cross-encoder relevance of retrieved context to the query | Higher = better |

### RAGAS-based Metrics (via `evaluate_ragas_light.py`)

| Metric | Description |
|---|---|
| **Faithfulness** | Whether the answer is grounded in the retrieved contexts |
| **Answer Relevancy** | Whether the answer addresses the question |
| **Context Recall** | Whether retrieved contexts cover the ground truth |

---

## Results

### Final Metrics Summary

| Pipeline | Cosine Sim | BERTScore F1 | Completeness | Hallucination | Irrelevance | Latency (s) |
|---|---|---|---|---|---|---|
| **vector_mistral** | **0.840** | **0.915** | 0.455 | 0.003 | **0.096** | **1.04** |
| vector_qwen | 0.797 | 0.886 | **0.464** | 0.007 | **0.000** | 4.51 |
| vector_llama | 0.783 | 0.891 | 0.462 | **0.003** | 0.333 | 2.16 |
| graph_mistral | 0.713 | 0.870 | 0.289 | 0.105 | 0.039 | 4.55 |
| graph_qwen | 0.684 | 0.858 | 0.312 | 0.103 | 0.112 | 9.46 |
| graph_llama | 0.588 | 0.757 | 0.201 | 0.088 | 0.421 | 7.38 |

**Key findings:**
- **Vector RAG consistently outperforms Graph RAG** across all metrics in this setup, achieving higher semantic similarity, lower hallucination, and lower latency.
- **Mistral 7B** delivers the best overall balance of quality and speed in the vector pipeline.
- **Qwen3 8B** achieves the lowest irrelevance score (near-zero) in the vector pipeline, but at higher latency.
- **Graph RAG pipelines** show higher hallucination rates (~10x) and lower completeness, suggesting the knowledge graph's coverage may be a bottleneck.
- Latency is 2–4x higher for Graph RAG due to the additional LLM call for intent extraction.

---

## Prerequisites

- **Python**: 3.10+
- **GPU**: CUDA-capable GPU with ≥16 GB VRAM (for running 7B/8B parameter LLMs in bfloat16)
- **Neo4j**: A Neo4j instance (local or [Neo4j Aura](https://neo4j.com/cloud/aura/) free tier) for the Graph RAG pipeline
- **HuggingFace**: Access tokens for gated models (LLaMA 3.1 requires accepting Meta's license on HuggingFace)
- **Disk Space**: ~30 GB for model weights (downloaded on first run)

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/SamiINReciept/Vector-Graph-RAG-Evaluation-Cybersecurity-Australian-SMEs.git
cd Vector-Graph-RAG-Evaluation-Cybersecurity-Australian-SMEs
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### 5. Configure Neo4j credentials

Update the `graph` section in each config YAML file (`config/vector_llama.yaml`, etc.) with your Neo4j connection details:

```yaml
graph:
  neo4j_uri: bolt://localhost:7687        # or your Neo4j Aura URI
  neo4j_user: neo4j
  neo4j_password: <your-password>
```

> **Tip:** For security, consider setting credentials via environment variables and loading them in `config_loader.py` instead of hardcoding them.

### 6. Configure HuggingFace access

```bash
huggingface-cli login
```

You need to accept the license agreements for:
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)

### 7. Place your PDF documents

Place your source PDF documents in the `data/pdfs/` directory.

---

## Reproduction Guide

### Step 1: Ingest PDFs into Chunks

Extract text from PDFs and split into overlapping chunks (1000 chars, 150 overlap):

```bash
python -m src.ingest
```

This reads from `data/pdfs/`, produces `data/corpus.pkl`.

### Step 2: Build the Vector Index

Build the ChromaDB vector store by embedding all chunks with BAAI/bge-m3:

```bash
python -m src.vector_rag
```

This loads `data/corpus.pkl`, embeds all chunks, and persists them to `data/vector_store/`. If the collection already has records, it skips the rebuild (use `force_rebuild=True` in code to recreate).

### Step 3: Build the Knowledge Graph

You have two options for populating Neo4j:

**Option A: LLM-based triple extraction from chunks**

```bash
python -m src.graph_build
```

This iterates over all chunks, prompts the LLM to extract structured (head, relation, tail) triples using the constrained schema, and writes them to Neo4j. Also saves `data/kg_nodes.pkl` and `data/kg_edges.pkl` for visualization.

**Option B: Import manually curated triples (recommended)**

If you have pre-curated triples (e.g., from NotebookLM), first merge them:

```bash
python -m src.merge_triples
```

This merges `triples_1.json`, `triples_2.json`, `triples_3.json` into `triples.json`.

Then import into Neo4j:

```bash
python -m src.graph_import_triples
```

### Step 4: Run the Evaluation Pipelines

Run all 6 pipelines (3 LLMs × 2 retrieval types) over the test set:

```bash
python -m src.evaluate_pipelines
```

This:
1. Loads `data/test_qas.csv` (60 QA pairs)
2. For each pipeline, runs every question through retrieval → generation
3. Cleans model outputs for metric computation
4. Computes METEOR, BERTScore F1, and Cosine Similarity
5. Saves per-pipeline CSVs to `results/{pipeline}_raw.csv`
6. Saves aggregated metrics to `results/summary_metrics.csv`

### Step 5: Compute Post-hoc CPU Metrics

Run additional CPU-based evaluation metrics (hallucination, completeness, irrelevance, faithfulness):

```bash
python -m src.posthoc_cpu_metrics
```

This loads each `{pipeline}_raw.csv`, enriches it with NLI-based hallucination scores, cross-encoder relevance scores, and ROUGE-L completeness, then saves:
- `results/{pipeline}_with_cpu_metrics.csv` (per-question enriched data)
- `results/summary_cpu_metrics.csv` (aggregated)

### Step 6: Produce Final Metrics

Merge GPU and CPU metrics into a single final table:

```bash
python -m src.finalized_metrics
```

Produces `results/final_metrics.csv`.

### Step 7: Visualize the Knowledge Graph

**Interactive HTML visualization (pyvis):**

```bash
python -m src.kg_viz
```

Produces `graph.html` — an interactive node-link diagram colored by entity type.

**Static hairball + zoomed subgraph (matplotlib):**

```bash
python -m src.kg_hairball
```

Produces `kg_hairball.jpg` and `kg_zoom.jpg`.

**Composite side-by-side image:**

```bash
python -m src.kg_composite
```

Produces `kg_composite.jpg` — the full hairball with a highlighted region next to a zoomed subgraph.

---

## Configuration

Configuration is managed via YAML files in `config/`. Each file specifies:

| Section | Parameters |
|---|---|
| `data` | PDF directory, corpus cache path |
| `chunking` | `chunk_size` (1000), `chunk_overlap` (150) |
| `embedding` | Model name (`BAAI/bge-m3`), batch size, device |
| `llm` | Model name, dtype, device_map, `max_new_tokens`, `temperature`, `top_p`, guardrails (`max_time`, `no_repeat_ngram_size`, `repetition_penalty`) |
| `vector_store` | Type (`chromadb`), persist directory |
| `graph` | Neo4j URI, user, password |

The `base.yaml` provides defaults. Per-model configs (`vector_llama.yaml`, `vector_mistral.yaml`, `vector_qwen.yaml`) override the LLM section. The same per-model configs are reused for both Vector RAG and Graph RAG (the pipeline kind is determined by `evaluate_pipelines.py`).

### Running Individual Pipelines

**Vector RAG with a specific model:**

```bash
python -m src.vector_rag --config config/vector_llama.yaml
```

**Graph RAG with a specific model and question:**

```bash
python -m src.graph_rag --config config/vector_mistral.yaml --question "What impacts do ransomware incidents have on small businesses in Australia?"
```

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
