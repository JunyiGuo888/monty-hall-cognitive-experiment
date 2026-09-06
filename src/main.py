import asyncio
import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation import run_simulation

async def main():
    await run_simulation()

if __name__ == "__main__":
    asyncio.run(main())