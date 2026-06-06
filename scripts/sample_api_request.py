"""Send a sample request to the local Rossmann API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import request

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/rossmann/predict")
    parser.add_argument("--input", default="data/sample/store_1_scoring.csv")
    args = parser.parse_args()

    df = pd.read_csv(Path(args.input))
    body = json.dumps({"records": df.to_dict(orient="records")}, default=str).encode("utf-8")
    req = request.Request(args.url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=30) as response:
        print(response.read().decode("utf-8"))


if __name__ == "__main__":
    main()
