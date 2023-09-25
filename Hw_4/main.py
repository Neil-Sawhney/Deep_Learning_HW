#!/usr/bin/env python3

import argparse
import importlib
from pathlib import Path

import argcomplete


def main():
    parser = argparse.ArgumentParser(description="Choose an example to train:")
    parser.add_argument("runner", type=Path,
                        help="Path to the runner file")
    parser.add_argument("--config", "-c", type=Path,
                        nargs='?', help="Path to the config file")
    parser.add_argument("--use_last_checkpoint", "-ch", action="store_true",
                        help="Whether or not to use the last checkpoint")

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    runner = importlib.import_module(f"runners.{args.runner.stem}")
    runner.run(args.config, args.use_last_checkpoint)


if __name__ == "__main__":
    main()
