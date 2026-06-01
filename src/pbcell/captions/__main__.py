"""Command-line entrypoint for pbcell.captions."""

from __future__ import annotations

import argparse
import importlib
from typing import List

COMMANDS = {
    "captions": "Generate morphology captions using a vision-language model."
}


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Captioning utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in COMMANDS.items():
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    module = importlib.import_module(f"pbcell.captions.{args.command}")
    module.main(args.args)


if __name__ == "__main__":
    main()
