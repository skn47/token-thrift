from __future__ import annotations

import subprocess
import sys

import pytest

from tokenthrift.config import ARTIFACTS_DIR, CORPUS_DIR, REPO_ROOT, SPLITS_DIR
from tokenthrift.corpus.registry import resolve_corpus
from tokenthrift.pruner.model import resolve_latest_version


def _run_script(name: str, *args: str) -> None:
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / name), *args],
        check=True, cwd=REPO_ROOT,
    )


@pytest.fixture(scope="session", autouse=True)
def ensure_fixtures_exist():
    """Makes `uv run pytest` work from a clean checkout: generates each
    bundled corpus, computes its splits, and trains an artifact for it if
    none of them exist yet. No-ops (cheap glob/file checks) once they do."""
    if not any(CORPUS_DIR.glob("*.json")):
        _run_script("generate_corpus.py")
    if not (SPLITS_DIR / "train.json").exists():
        _run_script("build_splits.py")
    if resolve_latest_version(ARTIFACTS_DIR) is None:
        _run_script("train_model.py")

    nimbus = resolve_corpus("nimbus") if _nimbus_corpus_generated() else None
    if nimbus is None:
        _run_script("generate_corpus_nimbus.py")
        nimbus = resolve_corpus("nimbus")
    if not (nimbus.splits_dir / "train.json").exists():
        _run_script("build_splits.py", "nimbus")
    if resolve_latest_version(nimbus.artifacts_dir) is None:
        _run_script("train_model.py", "nimbus")


def _nimbus_corpus_generated() -> bool:
    from tokenthrift.config import CORPORA_DIR
    return (CORPORA_DIR / "nimbus" / "manifest.json").exists()
