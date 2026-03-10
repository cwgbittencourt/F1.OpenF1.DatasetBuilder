from __future__ import annotations

import argparse
import os
from pathlib import Path
import pandas as pd


def main() -> None:
    data_dir = os.getenv("DATA_DIR", "./f1_dataset/data")
    parser = argparse.ArgumentParser(description="Consolidar datasets gold")
    parser.add_argument("--gold-dir", default=str(Path(data_dir) / "gold"))
    parser.add_argument("--output", default=str(Path(data_dir) / "gold" / "consolidated.parquet"))
    parser.add_argument("--output-csv", default=None, help="Opcional: caminho para salvar CSV consolidado")
    args = parser.parse_args()

    gold_dir = Path(args.gold_dir)
    frames = []
    for path in gold_dir.rglob("dataset.parquet"):
        frames.append(pd.read_parquet(path))
    if not frames:
        print("Nenhum arquivo gold encontrado.")
        return

    df = pd.concat(frames, ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    if args.output_csv:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Consolidado CSV salvo em {csv_path}")
    print(f"Consolidado salvo em {output}")


if __name__ == "__main__":
    main()
