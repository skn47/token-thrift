# Codex CLI

**Not yet directly compatible — this page is a status report, not a
recipe.** As of the current Codex CLI release, `wire_api` in
`config.toml` only supports `"responses"`: Codex speaks OpenAI's
Responses API, not the Chat Completions format
(`POST {base_url}/chat/completions`, `messages: [{role, content}]`) that
TokenThrift's proxy implements at `/v1/chat/completions`. Pointing
Codex's `base_url` at the proxy today would send request/response shapes
the proxy doesn't parse (the Responses API uses `input`, not `messages`,
among other structural differences) — it wouldn't prune anything, and
would most likely just fail.

This was checked against Codex's own current configuration
documentation while writing this page, not assumed from memory —
`wire_api = "chat"` was removed in a recent Codex release and now hard
errors, so there's no fallback path to the format this proxy speaks.

```toml
# For reference — what today's Codex custom-provider syntax looks like.
# Pointing base_url at TokenThrift's proxy here does NOT work yet.
[model_providers.example]
name = "example"
base_url = "http://localhost:8787"
env_key = "OPENAI_API_KEY"
wire_api = "responses"   # the proxy has no /v1/responses handler
```

## What would make this work

TokenThrift's proxy would need a `/v1/responses`-compatible endpoint —
parsing the Responses API's `input` structure and however it represents
tool output — alongside the existing `/v1/chat/completions` and
`/v1/messages` handlers. That's not built yet.

## What works today

- **Claude Code** — see [claude-code.md](claude-code.md); fully working,
  same wire format the proxy already implements.
- Any other tool that still speaks Chat Completions
  (`role`/`content` messages, not the Responses API) against
  `/v1/chat/completions` works the same way — see the README's
  [Proxy section](../../README.md#proxy-using-tokenthrift-with-a-real-coding-agent).
