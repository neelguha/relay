"""Cache layer for relay.

Exports the abstract base class and the built-in cache backends so that
callers can import them from a single location::

    from relay.cache import CacheBackend, SQLiteCache
"""

from relay.cache.base import CacheBackend
from relay.cache.sqlite import SQLiteCache

__all__ = [
    "CacheBackend",
    "SQLiteCache",
]
