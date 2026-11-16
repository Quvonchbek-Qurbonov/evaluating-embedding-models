# evaluating-embedding-models

A pipeline to benchmark sentence-embedding models across many tasks, datasets,
and runtime settings, then report a wide set of metrics.

## Models under test

| id         | HF repo                            | dim  | ctx  | prompts            |
|------------|------------------------------------|------|------|--------------------|
| `bge-m3`   | `BAAI/bge-m3`                      | 1024 | 8192 | none               |
| `e5-large` | `intfloat/multilingual-e5-large`  | 1024 | 512  | `query:`/`passage:`|
| `e5-small` | `intfloat/multilingual-e5-small`  | 384  | 512  | `query:`/`passage:`|

Prompt conventions matter: E5 needs instruction prefixes, BGE-M3 must *not*
have them. This is handled per model in `src/config.py`.

## Setup

```bash
# 1. download weights into models/
.venv/bin/hf download BAAI/bge-m3                      --local-dir models/bge-m3
.venv/bin/hf download intfloat/multilingual-e5-large  --local-dir models/multilingual-e5-large
.venv/bin/hf download intfloat/multilingual-e5-small  --local-dir models/multilingual-e5-small

# 2. deps are already in .venv; otherwise:
pip install -r requirements.txt
```

If a model folder is missing the pipeline falls back to downloading from the HF
repo id on first use.

## Run

```bash
.venv/bin/python main.py --list                       # show models + tasks
.venv/bin/python main.py --models e5-small --tasks sts-en --sweep none   # smoke test
.venv/bin/python main.py --sweep full                 # everything
.venv/bin/python main.py --sweep key --truncate-dims 256 512   # Matryoshka study
```

Results land in `results/<timestamp>/`: `results_long.csv`, `summary.md`,
`comparison.png`, `throughput.png`.

## Tasks & datasets

| kind           | datasets                                             |
|----------------|------------------------------------------------------|
| STS            | `stsb_multi_mt` (en, es, ru, zh)                     |
| Retrieval (QA) | `sentence-transformers/natural-questions`            |
| Bitext mining  | `Helsinki-NLP/opus-100` (en-ru, en-ar, en-zh)        |
| Classification | `ag_news`, `papluca/language-identification`         |
| Clustering     | `ag_news`, `papluca/language-identification`         |
| Reranking      | built from Natural Questions                         |

## Metrics

- **Retrieval / bitext**: nDCG@k, MAP, MRR, Recall@k, Precision@k, Hit@k (k=1,5,10)
- **STS**: Spearman, Pearson (cosine vs gold)
- **Classification**: accuracy, macro-F1, weighted-F1 (linear probe)
- **Clustering**: V-measure, ARI, NMI (KMeans)
- **Reranking**: MAP, MRR
- **Efficiency**: load time, throughput (texts/sec), latency (ms/text), peak GPU memory

## Settings sweep

Each model is evaluated across the Cartesian product of:
normalization (on/off) x precision (fp32/fp16) x batch size x Matryoshka
truncation dim. See `src/config.py::build_sweep`.

## Layout

```
src/config.py        model registry, prompt conventions, sweep grid
src/embedder.py      model wrapper: prompting, precision, truncation, efficiency
src/data_loaders.py  dataset loaders + task containers
src/metrics.py       all metric implementations
src/evaluate.py      runner (dispatch by task kind)
src/report.py        CSV / markdown / plots
main.py              CLI
```
