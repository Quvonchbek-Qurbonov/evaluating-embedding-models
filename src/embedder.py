"""Unified embedder wrapper that applies each model's prompt convention and the
current sweep settings (precision, normalization, Matryoshka truncation), while
recording efficiency metrics (load time, throughput, peak GPU memory)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .config import ModelSpec, SweepConfig


@dataclass
class EncodeStats:
    n_texts: int = 0
    total_seconds: float = 0.0
    peak_mem_mb: float = 0.0

    @property
    def texts_per_sec(self) -> float:
        return self.n_texts / self.total_seconds if self.total_seconds else 0.0

    @property
    def ms_per_text(self) -> float:
        return 1000.0 * self.total_seconds / self.n_texts if self.n_texts else 0.0


class Embedder:
    """Wraps a SentenceTransformer for one model under one sweep config."""

    def __init__(self, spec: ModelSpec, sweep: SweepConfig, device: str = "cuda"):
        self.spec = spec
        self.sweep = sweep
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"

        t0 = time.monotonic()
        self.model = SentenceTransformer(spec.resolve_path(), device=self.device)
        if spec.max_seq_length:
            self.model.max_seq_length = spec.max_seq_length
        if sweep.precision == "fp16" and self.device == "cuda":
            self.model.half()
        self.load_seconds = time.monotonic() - t0

        self.dim = self.model.get_sentence_embedding_dimension()
        self.stats = EncodeStats()

    # -- prompting --------------------------------------------------------
    def _apply(self, texts: list[str], prefix: str) -> list[str]:
        return [f"{prefix}{t}" for t in texts] if prefix else list(texts)

    # -- encoding ---------------------------------------------------------
    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prepared = self._apply(texts, prefix)

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        vecs = self.model.encode(
            prepared,
            batch_size=self.sweep.batch_size,
            normalize_embeddings=False,   # we normalize after truncation
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        elapsed = time.monotonic() - t0

        self.stats.n_texts += len(texts)
        self.stats.total_seconds += elapsed
        if self.device == "cuda":
            peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self.stats.peak_mem_mb = max(self.stats.peak_mem_mb, peak)

        # Matryoshka truncation, then (optional) re-normalisation.
        if self.sweep.truncate_dim and self.sweep.truncate_dim < vecs.shape[1]:
            vecs = vecs[:, : self.sweep.truncate_dim]
        if self.sweep.normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.clip(norms, 1e-12, None)
        return vecs

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.spec.query_prefix)

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.spec.doc_prefix)

    def encode_symmetric(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.spec.symmetric_prefix)

    @property
    def effective_dim(self) -> int:
        return self.sweep.truncate_dim or self.dim

    def unload(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
