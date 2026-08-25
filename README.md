# ⚡ TokenThrift

**Same answer, a fraction of the context.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![Managed by uv](https://img.shields.io/badge/managed%20by-uv-8A2BE2)](https://github.com/astral-sh/uv)

TokenThrift is a lightweight context gatekeeper for RAG: it sits between
retrieval and generation, scores every retrieved chunk with a small trained
classifier, and hands the LLM only what it needs. Bring your own API key and
watch it happen live, side by side with the unpruned baseline — same
retrieval, same model, same prompt, only the context differs.

![TokenThrift: a real Compare run, unpruned vs. pruned, side by side](docs/assets/demo.gif)

*Real capture, not staged: 571 → 194 input tokens (66% saved) on one
Lighthouse-corpus question, run locally against Ollama.*

## Why TokenThrift

Most RAG setups either send every retrieved chunk to the model every time,
or bolt on a heavy re-ranking stack (a second vector index, an embedding
service, a hosted reranker API) just to trim context.

| | Status quo (no pruning) | Heavy RAG frameworks | **TokenThrift** |
|---|---|---|---|
| Setup | — | new service(s) to run/pay for | `uv sync`, one process |
| Context selection | none — full recall set, every call | learned reranker / embeddings | 10-feature logistic-regression classifier |
| Infra beyond Python | — | vector DB, embedding API | none required (TF-IDF retriever ships in-process) |
| Safety net | — | usually none | deterministic rules: never drop the top hit, honor a min-context floor, never silently truncate |
| Provider lock-in | whatever you already use | often one platform | any OpenAI-compatible endpoint, or Anthropic — BYOK |
| Cost per query | 100% of context, every time | varies | targets **40–70% token reduction** on suitable workloads *(measured, not guaranteed)* |

The model is a single `SGDClassifier(loss="log_loss")` — inference is a dot
product, not a network call. No GPU, no vector database, no extra service
to run. It's a library-sized addition to a RAG pipeline you already have.

## How it works

```
Query -> Retrieval -> Feature Extraction -> ML Pruner -> Safety Rules -> Pruned Prompt -> BYOK LLM -> Answer
                                                              ^                                          |
                                                              +---------- session feedback loop ---------+
```

![Architecture](docs/assets/architecture.svg)

Ten inexpensive signals per chunk (query overlap, TF-IDF similarity,
retrieval rank, heading/entity overlap, neighbor relevance, and more) feed
a trained classifier, then deterministic safety rules take over: the
top-ranked chunk is never dropped, a minimum context floor is always kept,
and a mandatory chunk that would blow the token budget is reported, never
silently cut. Accept/Incorrect feedback during a session can calibrate the
threshold, or — opt-in — drive a guarded, session-local model update that's
discarded on reset.

Full technical depth — the pruner's I/O contract, all 10 features, the
safety-rule list, the feedback-attribution table, and the evaluation
protocol — lives in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## Quickstart

```bash
uv sync
uv run streamlit run src/tokenthrift/ui/app.py
```

Pick a provider in the sidebar (Groq, OpenAI, Anthropic, OpenRouter, a
local Ollama/custom OpenAI-compatible endpoint, or anything else that
speaks the OpenAI or Anthropic wire format) and paste its API key — held
only in that session's memory, never persisted. As a dev convenience, each
preset also reads a matching `.env` variable as its default
(`GROQ_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`).

Point it at your own docs instead of the bundled demo corpora: choose
"My own folder" in the sidebar and pick (or type) a path to a folder of
`.md`/`.txt` files — pruning borrows a trained model from one of the
bundled corpora since ad hoc content has no relevance labels of its own.

### Rebuilding a bundled corpus / model artifact

Each bundled corpus (`data/corpora/{corpus_id}/`) has its own generator,
splits, and trained pruner artifact (`artifacts/pruner/{corpus_id}/v{n}/`):

```bash
uv run python scripts/generate_corpus.py            # lighthouse corpus
uv run python scripts/generate_corpus_nimbus.py     # nimbus corpus
uv run python scripts/build_splits.py [corpus_id]   # defaults to lighthouse
uv run python scripts/train_model.py [corpus_id]    # defaults to lighthouse
```

## Built in stages, tested at every step

Each increment shipped with its own test suite rather than as one big
change — `tests/stage1_mvp/` through `tests/stage8_proxy_ui/` map directly
onto what was actually built:

1. **MVP** — TF-IDF retrieval, a trained pruner, deterministic safety
   rules, Groq BYOK, the two-column comparison UI.
2. **Safety & calibration** — boundary buffering, budget-conflict
   handling, conservative fallback, bounded session calibration.
3. **Feedback** — the feedback-attribution table, layered answer
   checking, explicit chunk labeling.
4. **Online learning** — guarded session-local SGD updates, a
   chronological (no-leakage) online evaluation protocol.
5. **Multi-corpus** — a corpus registry plus ad hoc ingestion of any
   local folder.
6. **Generalized BYOK** — Anthropic and arbitrary OpenAI-compatible
   clients, unknown-model cost handling.
7. **Reverse proxy** — a generic, model-free TF-IDF scorer that prunes
   real agent traffic in flight.
8. **Proxy UI** — sidebar connectivity check and a live pass-through
   toggle for the running proxy.

## Proxy: using TokenThrift with a real coding agent

`src/tokenthrift/proxy/` is a **passive local proxy, not an autonomous
background agent.** It runs continuously on a local port, but it only
does work when an API request actually passes through it — no file
watching, no polling, no deciding on its own when to act.

```
Coding agent
    -> API request
TokenThrift proxy (localhost:8787)
    -> pruned API request
Upstream provider (Ollama / OpenAI / Anthropic / Groq / ...)
```

### The usage loop

You don't "install" TokenThrift into an agent. Instead:

1. Start the proxy in one terminal, pointed at your real upstream.
2. Configure the coding agent's API base URL to point at the proxy
   instead.
3. Use the agent normally in another terminal — every model call it
   makes now passes through the proxy first.

```bash
# Terminal 1 — against a hosted provider
TOKENTHRIFT_UPSTREAM_BASE_URL=https://api.groq.com/openai/v1 \
  uv run python -m tokenthrift.proxy.server

# Terminal 1 — or against a local model, no API key needed
TOKENTHRIFT_UPSTREAM_BASE_URL=http://localhost:11434/v1 \
  uv run python -m tokenthrift.proxy.server
```

```bash
# Terminal 2 — point the agent at the proxy instead of the provider
export OPENAI_BASE_URL=http://localhost:8787/v1        # OpenAI-compatible agents
# or: export ANTHROPIC_BASE_URL=http://localhost:8787   # Claude Code
your-coding-agent
```

Same API key, same wire format either way — the exact environment
variable your agent reads depends on the agent.

### When does it prune?

Once per model request, immediately before forwarding it upstream — the
proxy never schedules or triggers a call itself, it only intercepts each
one as it happens:

```
Time 1: you ask a question         -> little or no marked context yet
Time 2: agent finishes reading docs -> sends results to the model
                                     -> proxy prunes the marked block
Time 3: agent reads more files      -> next request pruned independently
Time 4: agent runs tests            -> that request pruned independently
```

### The catch: pruning is opt-in per request

TokenThrift never guesses what's safe to prune. It only touches text
wrapped like this:

```
<tokenthrift:context>
large documentation, file contents, or logs
</tokenthrift:context>
```

Unmarked text is forwarded byte-for-byte unchanged. That means **the
calling agent (or an adapter in front of it) has to add the markers** —
this repo isn't yet a drop-in "turn it on and every agent automatically
uses less context" tool. For example, if an agent reading five docs
files sends:

```
docs/install.md ...
docs/authentication.md ...
docs/webhooks.md ...

Question: Explain authentication.
```

an integrated agent instead wraps the file contents:

```
<tokenthrift:context>
docs/install.md ...
docs/authentication.md ...
docs/webhooks.md ...
</tokenthrift:context>

Question: Explain authentication.
```

and only *then* does the proxy prune the marked material down to
whatever's actually relevant to "Explain authentication" before it
reaches the model.

### What "running" means

| Proxy state | What happens |
|---|---|
| Idle | Nothing — no filesystem watching, no polling |
| Request with no markers | Forwarded unchanged |
| Request with `<tokenthrift:context>` markers | Pruned, then forwarded |
| Upstream responds | Response returned to the agent |

It does **not** monitor your filesystem, watch terminal commands, decide
when the agent should call the model, summarize conversation history on
its own, or modify an agent's memory. It's context-filtering HTTP
middleware: `agent request -> filter marked context -> upstream`.

The Streamlit app's sidebar has a read-only "Proxy" panel to check a
running proxy's connectivity/config, and a toggle that routes the current
Compare query through it live so you can watch it prune a real request
end to end.

## Tests

```bash
uv run pytest
```

## License

[MIT](LICENSE)
