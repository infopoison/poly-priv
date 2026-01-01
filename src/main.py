"""
Entry point for Railway deployment.
"""

import asyncio

from src import config
from src.utils import logging as log


async def main() -> None:
    """Main entry point."""
    log.info(
        "startup",
        mode=config.MODE,
        threshold_entry=config.THRESHOLD_ENTRY,
        dip_threshold_points=config.DIP_THRESHOLD_POINTS,
        position_size_usd=config.POSITION_SIZE_USD,
    )

    if config.IS_PAPER_MODE:
        log.info("running_paper_mode")
    else:
        log.info("running_live_mode")
        if not config.POLYMARKET_API_KEY:
            log.error("missing_credentials", msg="POLYMARKET_API_KEY required for live mode")
            return

    # TODO: Initialize components
    # - Market discovery (REST API to find eligible markets)
    # - WebSocket monitor (subscribe to price updates)
    # - Signal detector (threshold crossing logic)
    # - Paper trader or live trader based on mode

    log.info("placeholder_loop_starting", msg="Replace with actual monitoring loop")
    
    # Placeholder: keep process alive
    while True:
        await asyncio.sleep(60)
        log.debug("heartbeat")


if __name__ == "__main__":
    asyncio.run(main())