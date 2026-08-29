import asyncio
import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import check_api_keys
from src.simulation import run_simulation

async def main():
    check_api_keys()
    await run_simulation()

if __name__ == "__main__":
    asyncio.run(main())