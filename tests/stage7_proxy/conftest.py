import pytest

from tokenthrift.proxy.server import _runtime_config


@pytest.fixture(autouse=True)
def _reset_runtime_config():
    """`_runtime_config` is the proxy's one piece of module-level mutable
    state — reset it around every test in this package so a test that turns
    auto-marking on can't leak that into an unrelated test importing the
    same `server.app` instance."""
    _runtime_config.auto_mark_tool_results = False
    yield
    _runtime_config.auto_mark_tool_results = False
