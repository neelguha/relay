"""Root conftest for relay tests.

Sets asyncio_mode to "auto" so that every coroutine test function is
automatically treated as an asyncio test without requiring the explicit
``@pytest.mark.asyncio`` decorator.  Individual tests that still carry the
decorator are unaffected.
"""

import pytest


# Ensure that pytest-asyncio >= 0.21 uses strict (per-test) mode, which is
# the closest to the legacy behaviour and avoids the deprecation warning about
# the default mode.
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as an asyncio coroutine test",
    )
