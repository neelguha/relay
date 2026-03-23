"""relay.db — SQLite database layer.

Public exports:

- :class:`JobStore` — async CRUD store for jobs, requests, results, and tags.
- :func:`get_price_per_million` — look up input/output price per million tokens.
- :func:`get_batch_price_per_million` — batch-discounted price lookup.
- :func:`estimate_cost_usd` — cost estimate helper.
"""

from relay.db.store import JobStore
from relay.db.prices import (
    estimate_cost_usd,
    get_batch_discount_pct,
    get_batch_price_per_million,
    get_model_info,
    get_price_per_million,
    list_models,
)

__all__ = [
    "JobStore",
    "estimate_cost_usd",
    "get_batch_discount_pct",
    "get_batch_price_per_million",
    "get_model_info",
    "get_price_per_million",
    "list_models",
]
