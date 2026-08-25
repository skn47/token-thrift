"""Trains and persists a new versioned pruner artifact for one bundled
corpus.

Run via `uv run python scripts/train_model.py [corpus_id]` after that
corpus's splits exist (see scripts/build_splits.py). corpus_id defaults to
"lighthouse" when omitted.
"""

import sys

from tokenthrift.config import DEFAULT_CORPUS_ID
from tokenthrift.pruner.training import train_and_evaluate


def main() -> None:
    corpus_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CORPUS_ID
    model = train_and_evaluate(seed=0, corpus_id=corpus_id)
    print(f"corpus_id={model.metadata.corpus_id}")
    print(f"threshold={model.metadata.threshold}")
    print(f"train_examples={model.metadata.train_examples} "
          f"val_examples={model.metadata.val_examples} "
          f"test_examples={model.metadata.test_examples}")
    print(f"val_metrics={model.metadata.val_metrics}")
    print(f"test_metrics={model.metadata.test_metrics}")


if __name__ == "__main__":
    main()
