"""Set the Telegram webhook for the Rossmann bot."""

from __future__ import annotations

import argparse
import os
from urllib import parse, request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Public HTTPS URL ending with /rossmann/bot")
    parser.add_argument("--token", default=os.environ.get("TELEGRAM_TOKEN"))
    args = parser.parse_args()
    if not args.token:
        raise SystemExit("Provide --token or set TELEGRAM_TOKEN.")
    endpoint = f"https://api.telegram.org/bot{args.token}/setWebhook?{parse.urlencode({'url': args.url})}"
    with request.urlopen(endpoint, timeout=30) as response:
        print(response.read().decode("utf-8"))


if __name__ == "__main__":
    main()
