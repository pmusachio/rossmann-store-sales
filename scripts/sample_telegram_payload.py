"""Simulate a Telegram webhook update."""

from __future__ import annotations

import argparse
import json
from urllib import request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5000/rossmann/bot")
    parser.add_argument("--store", default="/1")
    args = parser.parse_args()
    payload = {
        "message": {
            "chat": {"id": 123456},
            "text": args.store,
        }
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(args.url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=30) as response:
        print(response.read().decode("utf-8"))


if __name__ == "__main__":
    main()
