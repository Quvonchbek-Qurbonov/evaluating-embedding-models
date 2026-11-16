"""Dataset loaders. Each returns a Task object (or raises, in which case the
runner skips it with a warning). Subsets are kept small so a full sweep is
tractable on a single GPU; raise the `limit` values for more rigorous numbers.

Chosen for reliability + multilingual coverage:
  STS              stsb_multi_mt            en, es, ru, zh   (0-5 similarity)
  Retrieval (QA)   sentence-transformers/natural-questions   en
  Bitext mining    Helsinki-NLP/opus-100    en-ru, en-ar, en-zh  (crosslingual)
  Classification   ag_news (en) + papluca/language-identification (multi)
  Clustering       ag_news + papluca/language-identification
  Reranking        built from natural-questions (1 positive + N negatives)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from datasets import load_dataset

from .config import CACHE_DIR

RNG = random.Random(42)


# ---------------------------------------------------------------------------
# Task containers
# ---------------------------------------------------------------------------
@dataclass
class RetrievalTask:
    name: str
    queries: list[str]
    docs: list[str]
    qrels: dict[int, set[int]]
    kind: str = "retrieval"


@dataclass
class STSTask:
    name: str
    a: list[str]
    b: list[str]
    gold: np.ndarray
    kind: str = "sts"


@dataclass
class ClassificationTask:
    name: str
    train_texts: list[str]
    train_y: np.ndarray
    test_texts: list[str]
    test_y: np.ndarray
    kind: str = "classification"


@dataclass
class ClusteringTask:
    name: str
    texts: list[str]
    labels: np.ndarray
    kind: str = "clustering"


@dataclass
class RerankingTask:
    name: str
    queries: list[str]
    candidates: list[list[str]]        # per-query candidate docs
    relevance: list[np.ndarray]        # per-query binary relevance
    kind: str = "reranking"


def _load(*args, **kwargs):
    kwargs.setdefault("cache_dir", str(CACHE_DIR))
    return load_dataset(*args, **kwargs)


# ---------------------------------------------------------------------------
# STS
# ---------------------------------------------------------------------------
def load_sts(lang: str = "en", limit: int = 1000) -> STSTask:
    ds = _load("PhilipMay/stsb_multi_mt", name=lang, split="test")
    ds = ds.select(range(min(limit, len(ds))))
    a = [r["sentence1"] for r in ds]
    b = [r["sentence2"] for r in ds]
    gold = np.array([r["similarity_score"] for r in ds], dtype=np.float32)
    return STSTask(name=f"sts-{lang}", a=a, b=b, gold=gold)


# ---------------------------------------------------------------------------
# Retrieval (Natural Questions QA): each query has exactly one gold answer doc.
# ---------------------------------------------------------------------------
def load_nq_retrieval(limit_queries: int = 500, extra_docs: int = 2000) -> RetrievalTask:
    ds = _load("sentence-transformers/natural-questions", split="train")
    n = min(limit_queries + extra_docs, len(ds))
    ds = ds.select(range(n))
    answers = [r["answer"] for r in ds]
    queries = [r["query"] for r in ds][:limit_queries]
    # Doc pool = all answers (unique); qrel maps query i -> its own answer idx.
    docs = answers
    qrels = {i: {i} for i in range(len(queries))}
    return RetrievalTask(name="nq-retrieval", queries=queries, docs=docs, qrels=qrels)


# ---------------------------------------------------------------------------
# Bitext mining as crosslingual retrieval: source sentence retrieves its
# translation out of the full target pool.
# ---------------------------------------------------------------------------
def load_bitext(pair: str = "en-ru", limit: int = 1000) -> RetrievalTask:
    src, tgt = pair.split("-")
    ds = _load("Helsinki-NLP/opus-100", name=pair, split="test")
    ds = ds.select(range(min(limit, len(ds))))
    queries = [r["translation"][src] for r in ds]
    docs = [r["translation"][tgt] for r in ds]
    qrels = {i: {i} for i in range(len(queries))}
    return RetrievalTask(name=f"bitext-{pair}", queries=queries, docs=docs, qrels=qrels)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def load_agnews_classification(train_n: int = 4000, test_n: int = 2000) -> ClassificationTask:
    tr = _load("fancyzhx/ag_news", split="train").shuffle(seed=42).select(range(train_n))
    te = _load("fancyzhx/ag_news", split="test").shuffle(seed=42).select(range(test_n))
    return ClassificationTask(
        name="agnews-clf",
        train_texts=[r["text"] for r in tr],
        train_y=np.array([int(r["label"]) for r in tr]),
        test_texts=[r["text"] for r in te],
        test_y=np.array([int(r["label"]) for r in te]),
    )


def load_langid_classification(train_n: int = 4000, test_n: int = 2000) -> ClassificationTask:
    tr = _load("papluca/language-identification", split="train").shuffle(seed=42).select(range(train_n))
    te = _load("papluca/language-identification", split="test").shuffle(seed=42).select(range(test_n))
    labels = sorted({r["labels"] for r in tr})
    idx = {l: i for i, l in enumerate(labels)}
    return ClassificationTask(
        name="langid-clf",
        train_texts=[r["text"] for r in tr],
        train_y=np.array([idx[r["labels"]] for r in tr]),
        test_texts=[r["text"] for r in te],
        test_y=np.array([idx[r["labels"]] for r in te]),
    )


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def load_agnews_clustering(limit: int = 2000) -> ClusteringTask:
    ds = _load("fancyzhx/ag_news", split="test").shuffle(seed=42).select(range(limit))
    return ClusteringTask(name="agnews-cluster",
                          texts=[r["text"] for r in ds],
                          labels=np.array([int(r["label"]) for r in ds]))


def load_langid_clustering(limit: int = 2000) -> ClusteringTask:
    ds = _load("papluca/language-identification", split="test").shuffle(seed=42).select(range(limit))
    labels = sorted({r["labels"] for r in ds})
    idx = {l: i for i, l in enumerate(labels)}
    return ClusteringTask(name="langid-cluster",
                          texts=[r["text"] for r in ds],
                          labels=np.array([idx[r["labels"]] for r in ds]))


# ---------------------------------------------------------------------------
# Reranking built from Natural Questions.
# ---------------------------------------------------------------------------
def load_nq_reranking(n_queries: int = 300, n_candidates: int = 20) -> RerankingTask:
    ds = _load("sentence-transformers/natural-questions", split="train")
    n = min(n_queries * 4, len(ds))
    ds = ds.select(range(n))
    rows = [{"q": r["query"], "a": r["answer"]} for r in ds]
    all_answers = [r["a"] for r in rows]
    queries, cands, rels = [], [], []
    for i in range(min(n_queries, len(rows))):
        pos = rows[i]["a"]
        negs = RNG.sample([a for j, a in enumerate(all_answers) if j != i],
                          k=min(n_candidates - 1, len(all_answers) - 1))
        candidate_list = [pos] + negs
        order = list(range(len(candidate_list)))
        RNG.shuffle(order)
        shuffled = [candidate_list[k] for k in order]
        rel = np.array([1 if candidate_list[k] == pos else 0 for k in order])
        queries.append(rows[i]["q"])
        cands.append(shuffled)
        rels.append(rel)
    return RerankingTask(name="nq-rerank", queries=queries, candidates=cands, relevance=rels)


# ---------------------------------------------------------------------------
# Registry: name -> (loader, kwargs). Edit freely to add/remove tasks.
# ---------------------------------------------------------------------------
TASK_LOADERS: dict[str, tuple] = {
    "sts-en": (load_sts, {"lang": "en"}),
    "sts-es": (load_sts, {"lang": "es"}),
    "sts-ru": (load_sts, {"lang": "ru"}),
    "sts-zh": (load_sts, {"lang": "zh"}),
    "nq-retrieval": (load_nq_retrieval, {}),
    "bitext-en-ru": (load_bitext, {"pair": "en-ru"}),
    "bitext-en-ar": (load_bitext, {"pair": "en-ar"}),
    "bitext-en-zh": (load_bitext, {"pair": "en-zh"}),
    "agnews-clf": (load_agnews_classification, {}),
    "langid-clf": (load_langid_classification, {}),
    "agnews-cluster": (load_agnews_clustering, {}),
    "langid-cluster": (load_langid_clustering, {}),
    "nq-rerank": (load_nq_reranking, {}),
}
