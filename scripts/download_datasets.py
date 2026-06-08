from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import PRIMARY_KAGGLE_DATASETS, RAW_DATA_DIR


def download_dataset(slug: str, output_dir: Path = RAW_DATA_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = ["kaggle", "datasets", "download", "-d", slug, "-p", str(output_dir), "--unzip"]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download verified Kaggle datasets.")
    parser.add_argument("--dataset", choices=list(PRIMARY_KAGGLE_DATASETS.keys()) + ["all"], default="ranking_pairs")
    args = parser.parse_args()

    selected = PRIMARY_KAGGLE_DATASETS if args.dataset == "all" else {args.dataset: PRIMARY_KAGGLE_DATASETS[args.dataset]}
    for name, slug in selected.items():
        print(f"Downloading {name}: {slug}")
        download_dataset(slug)


if __name__ == "__main__":
    main()
