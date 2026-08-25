from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
CORPORA_DIR = DATA_DIR / "corpora"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "pruner"
PRICING_PATH = REPO_ROOT / "src" / "tokenthrift" / "pricing" / "model_pricing.json"

DEFAULT_CORPUS_ID = "lighthouse"

# Back-compat single-corpus constants: point at the default bundled corpus
# so every loader's existing zero-argument call site (and every Stage 1-4
# test) keeps working unchanged. Multi-corpus callers instead resolve an
# explicit tokenthrift.corpus.registry.CorpusSpec and pass its paths.
CORPUS_DIR = CORPORA_DIR / DEFAULT_CORPUS_ID / "documents"
LABELS_PATH = CORPORA_DIR / DEFAULT_CORPUS_ID / "labels.jsonl"
SPLITS_DIR = CORPORA_DIR / DEFAULT_CORPUS_ID / "splits"
ARTIFACTS_DIR = ARTIFACTS_ROOT / DEFAULT_CORPUS_ID

FEATURE_VERSION = "v1"

# Broad-recall retrieval width: how many chunks the retriever hands to the
# pruner before scoring. Must be >= any demo query's evidence-chunk count.
RETRIEVAL_TOP_K = 12
