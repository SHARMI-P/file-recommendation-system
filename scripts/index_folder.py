"""Index a folder from the command line (close the Streamlit app first: the vector store allows one process).

Usage:  python scripts/index_folder.py demo_workspace
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filerec.pipeline import Engine  # noqa: E402


def main(folder: str):
    eng = Engine()
    t0 = time.time()

    def progress(i, n, name):
        print(f"\r[{i}/{n}] {name[:60]:<60}", end="", flush=True)

    counts = eng.index_folder(Path(folder), progress)
    print(f"\nDone in {time.time() - t0:.1f}s: {counts}")
    eng.vec.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo_workspace")
