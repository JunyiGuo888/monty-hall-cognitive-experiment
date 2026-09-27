import asyncio
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation import run_simulation
from src.analyzer import run_analysis, AnalysisError
from config.settings import OLLAMA_ENABLED, PROGRESS_FILE
from src.api_client import check_ollama_available


async def main():
    if OLLAMA_ENABLED:
        if not await check_ollama_available():
            print("[Init] OLLAMA_ENABLED=true but Ollama is not usable. Aborting.")
            sys.exit(1)

    await run_simulation()

    print(f"\n{'=' * 80}\nStarting CoT analysis\n{'=' * 80}")
    try:
        paths = await run_analysis(PROGRESS_FILE)
        print("\n[Analyzer] All outputs written:")
        for k, v in paths.items():
            print(f"  {k}: {v}")
    except AnalysisError as e:
        print(f"\n[Analyzer] FAILED: {e}")
        print("[Analyzer] Simulation results are intact; re-run analysis later with:")
        print(f"    python src/analyzer.py {PROGRESS_FILE}")
    except Exception as e:
        print(f"\n[Analyzer] Unexpected error: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("[Analyzer] Simulation results are intact.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)