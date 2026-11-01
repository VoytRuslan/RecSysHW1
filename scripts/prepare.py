from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.data import download, prepare


if __name__ == "__main__":
    raw = ROOT / "data/raw/goodreads"
    download(raw)
    _, summary = prepare(raw, ROOT / "data/processed", rebuild=True)
    for row in summary["funnel"]:
        print(row)
