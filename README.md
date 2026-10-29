# Evaluating Embedding Models

This repository is for experimenting with different embedding models and
evaluating how they perform across different types of data.

## Goals

- Compare embedding model quality across multiple datasets.
- Measure performance by task and data modality (for example: text, code, or
  domain-specific corpora).
- Make model evaluations reproducible and easy to extend with new models and
  datasets.

## What to evaluate

- Retrieval quality (for example: Recall@K, MRR, nDCG)
- Clustering/separation behavior
- Similarity consistency on different data distributions

## Planned workflow

1. Add or select datasets for the target data type.
2. Generate embeddings with one or more models.
3. Run the same evaluation pipeline for each model.
4. Compare metrics and summarize trade-offs.