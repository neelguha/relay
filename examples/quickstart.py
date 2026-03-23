#!/usr/bin/env python3
"""Quickstart example for the relay library.

This is split into two scripts you'd run at different times:

  1. submit_batch()  — fire off the job with a name, walk away
  2. check_results() — come back later, look it up by name

Prerequisites:
    pip install relay[anthropic]
    export ANTHROPIC_API_KEY=your-key-here

Usage:
    python examples/quickstart.py submit
    python examples/quickstart.py check
"""

import asyncio
import sys

from relay import BatchClient, BatchConfig, BatchRequest


# -- Prompts & config (shared) ------------------------------------------------

PROMPTS = [
    "What is the capital of France? Reply in one sentence.",
    "Explain photosynthesis in two sentences.",
    "What causes tides? Reply briefly.",
    "Name three prime numbers greater than 100.",
    "What is the speed of light in km/s?",
]

JOB_NAME = "science-quiz"

CONFIG = BatchConfig(
    provider="anthropic",
    model="claude-haiku-4-5",
    name=JOB_NAME,                  # <-- human-readable name
    project="quickstart-demo",
    tags=["example"],
)


# -- Step 1: Submit and walk away ---------------------------------------------

async def submit_batch():
    """Submit the batch job. No need to save a UUID."""
    requests = [
        BatchRequest(
            id=f"q{i}",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.0,
        )
        for i, prompt in enumerate(PROMPTS)
    ]

    async with BatchClient() as client:
        estimate = await client.estimate_cost(requests, CONFIG)
        print(f"Estimated cost: ${estimate.net_usd:.4f}")

        job = await client.submit(requests, CONFIG)
        print(f"\nJob submitted!")
        print(f"  Name:  {job.name}")
        print(f"  ID:    {job.id}")
        print(f"\nCheck on it later with:")
        print(f"  python examples/quickstart.py check")


# -- Step 2: Come back later --------------------------------------------------

async def check_results():
    """Check job status by name. If done, download and print results."""
    async with BatchClient() as client:
        # Look up by name — no UUID needed
        job = await client.get_job(JOB_NAME)

        print(f"Job:      {job.name}")
        print(f"Status:   {job.status.value}")
        print(f"Progress: {job.completed_requests}/{job.total_requests}")

        if not job.status.is_terminal:
            print(f"\nStill running — check back later.")
            return

        if job.status.value == "FAILED":
            print(f"\nJob failed: {job.error}")
            return

        results = await client.download(job.id)
        print(f"\n{len(results)} results:\n")
        for result in results:
            idx = int(result.request_id[1:])
            print(f"  Q: {PROMPTS[idx]}")
            print(f"  A: {result.content[:200]}")
            print()

        path = "./quickstart_results.jsonl"
        await client.export(job.id, format="jsonl", path=path)
        print(f"Exported to {path}")


# -- CLI -----------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python examples/quickstart.py submit")
        print("  python examples/quickstart.py check")
        sys.exit(1)

    command = sys.argv[1]

    if command == "submit":
        asyncio.run(submit_batch())
    elif command == "check":
        asyncio.run(check_results())
    else:
        print(f"Unknown command: {command}")
        print("Use 'submit' or 'check'")
        sys.exit(1)


if __name__ == "__main__":
    main()
