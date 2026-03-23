#!/usr/bin/env python3
"""Multi-provider comparison example.

Sends the same prompts to multiple providers and compares responses
side by side.

Prerequisites:
    pip install relay[all]
    export ANTHROPIC_API_KEY=...
    export OPENAI_API_KEY=...

Usage:
    python examples/multi_provider.py
"""

import asyncio

from relay import BatchClient, BatchConfig, BatchRequest, fan_out


requests = [
    BatchRequest(
        id=f"q{i}",
        messages=[{"role": "user", "content": p}],
        max_tokens=256,
    )
    for i, p in enumerate([
        "What is 2+2? Reply with just the number.",
        "Name the largest ocean.",
        "Who wrote Hamlet?",
    ])
]


async def main():
    async with BatchClient() as client:
        results = await fan_out(
            client,
            requests=requests,
            configs=[
                BatchConfig(provider="anthropic", model="claude-haiku-4-5"),
                BatchConfig(provider="openai", model="gpt-4o-mini"),
            ],
            wait=True,
        )

        # Compare responses
        for provider_model, result_list in results.items():
            print(f"\n=== {provider_model} ===")
            if isinstance(result_list, Exception):
                print(f"  Error: {result_list}")
                continue
            for r in result_list:
                print(f"  [{r.request_id}] {r.content[:100]}")


if __name__ == "__main__":
    asyncio.run(main())
