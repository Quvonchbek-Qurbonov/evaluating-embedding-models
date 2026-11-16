"""CLI for evaluating embedding models.

Examples
--------
# list available models and tasks
python main.py --list

# quick smoke test: one model, one task, minimal sweep
python main.py --models e5-small --tasks sts-en --sweep none

# full comparison across all models, all tasks, full setting sweep
python main.py --sweep full

# add Matryoshka truncation dims (dimension-reduction study)
python main.py --models bge-m3 e5-large --sweep key --truncate-dims 256 512
"""
from __future__ import annotations

import argparse
import logging

from src.config import MODELS, build_sweep
from src.data_loaders import TASK_LOADERS
from src.evaluate import run
from src.report import save_all


def make_sweep(preset: str, truncate_dims: list[int] | None):
    dims = tuple([None] + list(truncate_dims)) if truncate_dims else (None,)
    if preset == "none":
        return build_sweep(normalize=(True,), precision=("fp32",),
                           batch_sizes=(64,), truncate_dims=dims)
    if preset == "key":
        return build_sweep(normalize=(True, False), precision=("fp32", "fp16"),
                           batch_sizes=(64,), truncate_dims=dims)
    return build_sweep(normalize=(True, False), precision=("fp32", "fp16"),
                       batch_sizes=(32, 128), truncate_dims=dims)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate embedding models.")
    p.add_argument("--models", nargs="*", choices=list(MODELS), default=None)
    p.add_argument("--tasks", nargs="*", choices=list(TASK_LOADERS), default=None)
    p.add_argument("--sweep", choices=["none", "key", "full"], default="key")
    p.add_argument("--truncate-dims", nargs="*", type=int, default=None,
                   help="extra Matryoshka truncation dims to test, e.g. 256 512")
    p.add_argument("--device", default="cuda")
    p.add_argument("--list", action="store_true", help="list models/tasks and exit")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.list:
        print("Models:")
        for name, spec in MODELS.items():
            print(f"  {name:10s} {spec.notes}")
        print("\nTasks:")
        for name in TASK_LOADERS:
            print(f"  {name}")
        return

    sweep = make_sweep(args.sweep, args.truncate_dims)
    logging.info("sweep: %d configs", len(sweep))
    rows = run(args.models, args.tasks, sweep, device=args.device)
    out = save_all(rows)
    print(f"\nDone. Results written to: {out}")
    print(f"  - {out/'results_long.csv'}")
    print(f"  - {out/'summary.md'}")
    print(f"  - {out/'comparison.png'}, {out/'throughput.png'}")


if __name__ == "__main__":
    main()
