"""Computes and persists the train/val/test document+question splits for
one bundled corpus.

Run after scripts/generate_corpus.py (or its content is edited) via
`uv run python scripts/build_splits.py [corpus_id]`. corpus_id defaults to
"lighthouse" when omitted.
"""

import sys

from tokenthrift.config import DEFAULT_CORPUS_ID
from tokenthrift.corpus.registry import resolve_corpus
from tokenthrift.corpus.splits import compute_splits, persist_splits


def main() -> None:
    corpus_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CORPUS_ID
    spec = resolve_corpus(corpus_id)
    splits = compute_splits(
        seed=0, corpus_dir=spec.corpus_dir, labels_path=spec.labels_path)
    persist_splits(splits, splits_dir=spec.splits_dir)
    print(f"corpus: {corpus_id}")
    for split in ("train", "val", "test"):
        print(f"{split}: {len(splits.doc_ids_for(split))} docs, "
              f"{len(splits.question_ids_for(split))} questions")
    print(f"excluded (cross-split evidence): "
          f"{sorted(splits.excluded_question_ids)}")


if __name__ == "__main__":
    main()
