"""Entry point: ``uv run python -m src``."""

import sys

# Suppress Windows asyncio connection reset errors (cosmetic, not critical).
if sys.platform == "win32":
    import asyncio

    # Use selector event loop policy to avoid ProactorEventLoop issues.
    # This prevents spurious "ConnectionResetError" messages in logs.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from src.app import main

main()
