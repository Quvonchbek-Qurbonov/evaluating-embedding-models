"""Evaluation runner: loads tasks once, then for every (model, sweep-config)
encodes the data, computes task metrics, and records efficiency stats.

Produces a long-format list of result rows:
    model, config, task, kind, metric, value, plus efficiency columns.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from . import metrics as M
from .config import DEFAULT_SWEEP, MODELS, ModelSpec, SweepConfig
from .data_loaders import (
    TASK_LOADERS,
    ClassificationTask,
    ClusteringTask,
    RerankingTask,
    RetrievalTask,
    STSTask,
)
from .embedder import Embedder

log = logging.getLogger("eval")


# ---------------------------------------------------------------------------
# Task loading (with graceful skipping of broken datasets)
# ---------------------------------------------------------------------------
def load_tasks(names: list[str] | None = None) -> list:
    names = names or list(TASK_LOADERS)
    tasks = []
    for name in names:
        loader, kwargs = TASK_LOADERS[name]
        try:
            t0 = time.monotonic()
            task = loader(**kwargs)
            log.info("loaded %-16s (%.1fs)", name, time.monotonic() - t0)
            tasks.append(task)
        except Exception as exc:  # noqa: BLE001 - skip unavailable datasets
            log.warning("SKIP task %s: %s", name, exc)
    return tasks


# ---------------------------------------------------------------------------
# Per-task evaluation (dispatch by kind)
# ---------------------------------------------------------------------------
def eval_task(emb: Embedder, task) -> dict[str, float]:
    if isinstance(task, STSTask):
        a = emb.encode_symmetric(task.a)
        b = emb.encode_symmetric(task.b)
        return M.sts_metrics(a, b, task.gold)

    if isinstance(task, RetrievalTask):
        q = emb.encode_queries(task.queries)
        d = emb.encode_docs(task.docs)
        return M.retrieval_metrics(q, d, task.qrels)

    if isinstance(task, ClassificationTask):
        tr = emb.encode_symmetric(task.train_texts)
        te = emb.encode_symmetric(task.test_texts)
        return M.classification_metrics(tr, task.train_y, te, task.test_y)

    if isinstance(task, ClusteringTask):
        x = emb.encode_symmetric(task.texts)
        return M.clustering_metrics(x, task.labels)

    if isinstance(task, RerankingTask):
        q = emb.encode_queries(task.queries)
        samples = []
        for i, cand in enumerate(task.candidates):
            c = emb.encode_docs(cand)
            qn = q[i] / max(np.linalg.norm(q[i]), 1e-12)
            cn = c / np.clip(np.linalg.norm(c, axis=1, keepdims=True), 1e-12, None)
            samples.append({"sim": cn @ qn, "rel": task.relevance[i]})
        return M.reranking_metrics(samples)

    raise TypeError(f"unknown task type {type(task)}")


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------
def run(
    model_names: list[str] | None = None,
    task_names: list[str] | None = None,
    sweep: list[SweepConfig] | None = None,
    device: str = "cuda",
) -> list[dict]:
    model_names = model_names or list(MODELS)
    sweep = sweep or DEFAULT_SWEEP
    tasks = load_tasks(task_names)

    rows: list[dict] = []
    for mname in model_names:
        spec: ModelSpec = MODELS[mname]
        for cfg in sweep:
            # fp16 is a CUDA-only concept; skip on CPU to avoid dup rows.
            if cfg.precision == "fp16" and device != "cuda":
                continue
            log.info("=== %s | %s ===", mname, cfg.label)
            try:
                emb = Embedder(spec, cfg, device=device)
            except Exception as exc:  # noqa: BLE001
                log.error("load failed %s/%s: %s", mname, cfg.label, exc)
                continue

            for task in tasks:
                try:
                    scores = eval_task(emb, task)
                except Exception as exc:  # noqa: BLE001
                    log.error("task %s failed on %s/%s: %s", task.name, mname, cfg.label, exc)
                    continue
                for metric, value in scores.items():
                    rows.append({
                        "model": mname,
                        "config": cfg.label,
                        "normalize": cfg.normalize,
                        "precision": cfg.precision,
                        "batch_size": cfg.batch_size,
                        "truncate_dim": cfg.truncate_dim or spec.native_dim,
                        "effective_dim": emb.effective_dim,
                        "task": task.name,
                        "kind": task.kind,
                        "metric": metric,
                        "value": value,
                    })

            # One efficiency row per (model, config).
            rows.append({
                "model": mname,
                "config": cfg.label,
                "normalize": cfg.normalize,
                "precision": cfg.precision,
                "batch_size": cfg.batch_size,
                "truncate_dim": cfg.truncate_dim or spec.native_dim,
                "effective_dim": emb.effective_dim,
                "task": "_efficiency",
                "kind": "efficiency",
                "metric": "load_seconds",
                "value": emb.load_seconds,
            })
            rows.append({**rows[-1], "metric": "texts_per_sec", "value": emb.stats.texts_per_sec})
            rows.append({**rows[-1], "metric": "ms_per_text", "value": emb.stats.ms_per_text})
            rows.append({**rows[-1], "metric": "peak_gpu_mem_mb", "value": emb.stats.peak_mem_mb})

            emb.unload()
    return rows
