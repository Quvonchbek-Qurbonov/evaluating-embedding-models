"""Metric implementations for every task type.

All functions take numpy embeddings and return a flat dict of {metric_name: value}
so results can be concatenated into a single table.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    v_measure_score,
)

K_VALUES = (1, 5, 10)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def _dcg(rels: np.ndarray) -> float:
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(rels * discounts))


def retrieval_metrics(
    q_emb: np.ndarray,
    d_emb: np.ndarray,
    qrels: dict[int, set[int]],
    k_values: tuple[int, ...] = K_VALUES,
) -> dict[str, float]:
    """Dense retrieval over the full doc pool. qrels maps query idx -> relevant
    doc indices. Uses cosine similarity (dot product on normalised vectors)."""
    qn = q_emb / np.clip(np.linalg.norm(q_emb, axis=1, keepdims=True), 1e-12, None)
    dn = d_emb / np.clip(np.linalg.norm(d_emb, axis=1, keepdims=True), 1e-12, None)
    sims = qn @ dn.T                      # (nq, nd)
    order = np.argsort(-sims, axis=1)     # ranked doc idx per query

    maxk = max(k_values)
    agg: dict[str, list[float]] = {}

    def push(name, val):
        agg.setdefault(name, []).append(val)

    aps, rrs = [], []
    for qi, ranked in enumerate(order):
        rel = qrels.get(qi, set())
        if not rel:
            continue
        ranked_rel = np.array([1.0 if d in rel else 0.0 for d in ranked[:maxk]])

        # MRR / AP over the full (truncated) ranking
        hit_pos = np.where(ranked_rel == 1.0)[0]
        rrs.append(1.0 / (hit_pos[0] + 1) if hit_pos.size else 0.0)
        if hit_pos.size:
            precisions = [(i + 1) / (p + 1) for i, p in enumerate(hit_pos)]
            aps.append(float(np.mean(precisions)) if precisions else 0.0)
        else:
            aps.append(0.0)

        for k in k_values:
            topk = ranked_rel[:k]
            n_rel = len(rel)
            push(f"recall@{k}", topk.sum() / n_rel)
            push(f"precision@{k}", topk.sum() / k)
            push(f"hit@{k}", 1.0 if topk.sum() > 0 else 0.0)
            ideal = np.ones(min(k, n_rel))
            idcg = _dcg(ideal)
            push(f"ndcg@{k}", _dcg(topk) / idcg if idcg else 0.0)

    out = {name: float(np.mean(vals)) for name, vals in agg.items()}
    out["map"] = float(np.mean(aps)) if aps else 0.0
    out["mrr"] = float(np.mean(rrs)) if rrs else 0.0
    return out


# ---------------------------------------------------------------------------
# Semantic Textual Similarity
# ---------------------------------------------------------------------------
def sts_metrics(a_emb: np.ndarray, b_emb: np.ndarray, gold: np.ndarray) -> dict[str, float]:
    an = a_emb / np.clip(np.linalg.norm(a_emb, axis=1, keepdims=True), 1e-12, None)
    bn = b_emb / np.clip(np.linalg.norm(b_emb, axis=1, keepdims=True), 1e-12, None)
    cos = np.sum(an * bn, axis=1)
    return {
        "spearman": float(spearmanr(cos, gold).correlation),
        "pearson": float(pearsonr(cos, gold)[0]),
    }


# ---------------------------------------------------------------------------
# Classification (linear probe on frozen embeddings)
# ---------------------------------------------------------------------------
def classification_metrics(
    train_emb: np.ndarray,
    train_y: np.ndarray,
    test_emb: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, float]:
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(train_emb, train_y)
    pred = clf.predict(test_emb)
    return {
        "accuracy": float((pred == test_y).mean()),
        "f1_macro": float(f1_score(test_y, pred, average="macro")),
        "f1_weighted": float(f1_score(test_y, pred, average="weighted")),
    }


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def clustering_metrics(emb: np.ndarray, labels: np.ndarray, seed: int = 42) -> dict[str, float]:
    n_clusters = len(np.unique(labels))
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    pred = km.fit_predict(emb)
    return {
        "v_measure": float(v_measure_score(labels, pred)),
        "ari": float(adjusted_rand_score(labels, pred)),
        "nmi": float(normalized_mutual_info_score(labels, pred)),
    }


# ---------------------------------------------------------------------------
# Reranking: each sample has a query + candidate list with binary relevance.
# ---------------------------------------------------------------------------
def reranking_metrics(samples: list[dict]) -> dict[str, float]:
    """samples: [{"sim": np.ndarray(n_cand,), "rel": np.ndarray(n_cand,)}]."""
    aps, rrs = [], []
    for s in samples:
        order = np.argsort(-s["sim"])
        rel = s["rel"][order]
        hit_pos = np.where(rel == 1)[0]
        rrs.append(1.0 / (hit_pos[0] + 1) if hit_pos.size else 0.0)
        if hit_pos.size:
            precisions = [(i + 1) / (p + 1) for i, p in enumerate(hit_pos)]
            aps.append(float(np.mean(precisions)))
        else:
            aps.append(0.0)
    return {"map": float(np.mean(aps)) if aps else 0.0,
            "mrr": float(np.mean(rrs)) if rrs else 0.0}
