"""Command line interface for the Rossmann workflow."""

from __future__ import annotations

import argparse
import json

from .config import load_config
from .data import profile
from .models import predict_file, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Rossmann Store Sales workflow.")
    parser.add_argument("--config", default=None, help="Path to configs/project.toml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-config")
    sub.add_parser("profile")
    sub.add_parser("train")
    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.command == "validate-config":
        print(json.dumps(load_config(args.config)["project"], indent=2))
    elif args.command == "profile":
        print(json.dumps(profile(args.config), indent=2))
    elif args.command == "train":
        print(json.dumps(train(args.config), indent=2))
    elif args.command == "predict":
        print(json.dumps(predict_file(args.input, args.output, args.config), indent=2))


if __name__ == "__main__":
    main()
