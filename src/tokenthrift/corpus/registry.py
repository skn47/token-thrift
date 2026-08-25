from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tokenthrift.config import ARTIFACTS_ROOT, CORPORA_DIR, DEFAULT_CORPUS_ID

CUSTOM_CORPUS_SENTINEL = "custom"


class UnknownCorpusError(Exception):
    """Raised when a corpus_id doesn't match any bundled corpus — never
    silently falls back to some other corpus."""


@dataclass(frozen=True)
class CorpusSpec:
    corpus_id: str
    display_name: str
    description: str
    corpus_dir: Path
    labels_path: Path | None
    splits_dir: Path | None
    artifacts_dir: Path
    labeled: bool


def _manifest_spec(corpus_root: Path) -> CorpusSpec:
    manifest = json.loads((corpus_root / "manifest.json").read_text())
    corpus_id = corpus_root.name
    return CorpusSpec(
        corpus_id=corpus_id,
        display_name=manifest["display_name"],
        description=manifest["description"],
        corpus_dir=corpus_root / "documents",
        labels_path=corpus_root / "labels.jsonl",
        splits_dir=corpus_root / "splits",
        artifacts_dir=ARTIFACTS_ROOT / corpus_id,
        labeled=True,
    )


def list_bundled_corpora() -> list[CorpusSpec]:
    """Every corpus with a manifest.json under data/corpora/ — each one is
    hand-labeled and gets its own trained pruner + honest held-out metrics."""
    if not CORPORA_DIR.exists():
        return []
    specs = []
    for child in sorted(CORPORA_DIR.iterdir()):
        if child.is_dir() and (child / "manifest.json").exists():
            specs.append(_manifest_spec(child))
    return specs


def resolve_corpus(corpus_id: str = DEFAULT_CORPUS_ID) -> CorpusSpec:
    corpus_root = CORPORA_DIR / corpus_id
    if not (corpus_root / "manifest.json").exists():
        available = ", ".join(s.corpus_id for s in list_bundled_corpora())
        raise UnknownCorpusError(
            f"unknown corpus_id {corpus_id!r}; bundled corpora: "
            f"{available or '(none found)'}")
    return _manifest_spec(corpus_root)


def ad_hoc_corpus(
    folder: Path, model_source_id: str, display_name: str | None = None,
) -> CorpusSpec:
    """A user-supplied folder of unlabeled documents: no labels/splits of
    its own, so no held-out evaluation is possible for it. Pruning against
    it must reuse an existing labeled corpus's trained model, chosen
    explicitly via `model_source_id` — never a silent default, since there
    is no principled "closest" corpus to guess."""
    model_spec = resolve_corpus(model_source_id)
    return CorpusSpec(
        corpus_id=CUSTOM_CORPUS_SENTINEL,
        display_name=display_name or folder.name,
        description=(
            f"Custom folder ({folder}) — unlabeled, no held-out evaluation. "
            f"Pruning uses the model trained on {model_spec.display_name}."),
        corpus_dir=folder,
        labels_path=None,
        splits_dir=None,
        artifacts_dir=model_spec.artifacts_dir,
        labeled=False,
    )
