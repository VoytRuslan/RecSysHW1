"""Повторный подбор только по validation; сохранённые финальные настройки не меняет."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from threadpoolctl import threadpool_limits
from utils.data import download, prepare
from utils.experiment import Experiment, search_spaces


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(search_spaces()))
    args = parser.parse_args()
    torch.set_num_threads(1)
    download(ROOT / "data/raw/goodreads")
    data, _ = prepare(ROOT / "data/raw/goodreads", ROOT / "data/processed")
    experiment = Experiment(data)
    selected, rows = {}, []
    with threadpool_limits(limits=1):
        for name, space in search_spaces().items():
            if args.model and args.model != name:
                continue
            log, selected[name] = experiment.search(name, space)
            rows.extend(log)
            print(name, selected[name], flush=True)
    result = dict(selected=selected, validation=rows)
    (ROOT / "results/search_repeated.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
