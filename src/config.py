"""Central configuration: model registry, prompt conventions, and sweep grid.

Each embedding model has its own *prompt convention*. Getting this wrong is the
single most common cause of bad numbers, so it is encoded here explicitly:

* E5 models (intfloat/multilingual-e5-*) were trained with instruction prefixes.
  - retrieval:  "query: " on the query side, "passage: " on the document side
  - symmetric tasks (STS / classification / clustering): both sides get "query: "
* BGE-M3 (BAAI/bge-m3) was trained WITHOUT prefixes; adding them hurts. All
  prefixes are therefore empty strings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = PROJECT_ROOT / "data" / "_hf_cache"


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to load and correctly prompt a model."""

    name: str                       # short id used in reports / filenames
    path: str                       # local dir under models/ (or a HF repo id)
    query_prefix: str = ""          # prepended to queries in retrieval
    doc_prefix: str = ""            # prepended to documents in retrieval
    symmetric_prefix: str = ""      # prepended to text in STS/clf/clustering
    # Matryoshka: truncation dims to test (empty -> only native dim).
    matryoshka_dims: tuple[int, ...] = field(default_factory=tuple)
    native_dim: int | None = None
    max_seq_length: int | None = None  # override model default if set
    notes: str = ""

    def resolve_path(self) -> str:
        """Prefer a locally downloaded copy; fall back to the HF repo id."""
        local = MODELS_DIR / self.path
        return str(local) if local.exists() else self.path


# ---------------------------------------------------------------------------
# Model registry.  `path` is the folder name under models/ that the download
# commands create; if it is missing we fall back to the HF repo id (download
# on demand), so the pipeline still runs before you fetch the weights.
# ---------------------------------------------------------------------------
MODELS: dict[str, ModelSpec] = {
    "bge-m3": ModelSpec(
        name="bge-m3",
        path="bge-m3",  # falls back to "BAAI/bge-m3"
        query_prefix="",
        doc_prefix="",
        symmetric_prefix="",
        native_dim=1024,
        max_seq_length=8192,
        notes="BAAI/bge-m3 — no instruction prefixes, 8192 ctx, 1024-dim.",
    ),
    "e5-large": ModelSpec(
        name="e5-large",
        path="multilingual-e5-large",  # falls back to "intfloat/multilingual-e5-large"
        query_prefix="query: ",
        doc_prefix="passage: ",
        symmetric_prefix="query: ",
        native_dim=1024,
        max_seq_length=512,
        notes="intfloat/multilingual-e5-large — query:/passage: prefixes, 512 ctx.",
    ),
    "e5-small": ModelSpec(
        name="e5-small",
        path="multilingual-e5-small",  # falls back to "intfloat/multilingual-e5-small"
        query_prefix="query: ",
        doc_prefix="passage: ",
        symmetric_prefix="query: ",
        native_dim=384,
        max_seq_length=512,
        notes="intfloat/multilingual-e5-small — query:/passage: prefixes, 384 ctx.",
    ),
}

# Fall-back HF repo ids when a local folder is absent.
HF_REPO_FALLBACK = {
    "bge-m3": "BAAI/bge-m3",
    "multilingual-e5-large": "intfloat/multilingual-e5-large",
    "multilingual-e5-small": "intfloat/multilingual-e5-small",
}


@dataclass(frozen=True)
class SweepConfig:
    """One point in the settings grid."""

    normalize: bool
    precision: str      # "fp32" | "fp16"
    batch_size: int
    truncate_dim: int | None = None  # Matryoshka truncation; None = native

    @property
    def label(self) -> str:
        dim = self.truncate_dim or "full"
        norm = "norm" if self.normalize else "raw"
        return f"{norm}|{self.precision}|bs{self.batch_size}|d{dim}"


def build_sweep(
    *,
    normalize=(True, False),
    precision=("fp32", "fp16"),
    batch_sizes=(32, 128),
    truncate_dims=(None,),
) -> list[SweepConfig]:
    """Cartesian product of the setting axes."""
    grid: list[SweepConfig] = []
    for n in normalize:
        for p in precision:
            for bs in batch_sizes:
                for td in truncate_dims:
                    grid.append(SweepConfig(n, p, bs, td))
    return grid


# A compact default sweep (16 combos). Override from the CLI for bigger grids.
DEFAULT_SWEEP = build_sweep()
