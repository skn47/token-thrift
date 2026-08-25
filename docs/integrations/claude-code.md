# Claude Code

Claude Code speaks Anthropic's native Messages API — the same wire format
TokenThrift's proxy implements at `POST /v1/messages` — so this is a
working integration today, not a future one.

## Setup

1. Start the proxy, pointed at the real Anthropic API:

   ```bash
   TOKENTHRIFT_UPSTREAM_BASE_URL=https://api.anthropic.com \
     uv run python -m tokenthrift.proxy.server
   ```

2. Point Claude Code at the proxy instead of `api.anthropic.com`.
   `ANTHROPIC_BASE_URL` is read once when the Claude Code process starts,
   so set it before launching — changing it mid-session won't take effect
   until you restart:

   ```bash
   export ANTHROPIC_BASE_URL=http://localhost:8787
   claude
   ```

   Or set it persistently in `~/.claude/settings.json`:

   ```json
   { "env": { "ANTHROPIC_BASE_URL": "http://localhost:8787" } }
   ```

3. Turn on tool-result auto-marking (below) so Claude Code's own file
   reads, greps, and command output get pruned automatically. Without it,
   only text you've manually wrapped in `<tokenthrift:context>` markers
   ever gets touched.

## Auto-marking tool results

Claude Code's tool calls come back to the model as Anthropic `tool_result`
content blocks — a structural signal TokenThrift's proxy can recognize on
its own, no markers required. Turn it on either from the Streamlit
sidebar's Proxy panel ("Auto-mark tool results") while the proxy is
running, or by setting `TOKENTHRIFT_AUTO_MARK_TOOL_RESULTS=1` before
starting the proxy. This only covers tool-result content — plain text you
paste into a prompt yourself still needs a manual marker. See the
README's
[Proxy section](../../README.md#proxy-using-tokenthrift-with-a-real-coding-agent)
for the full explanation.
