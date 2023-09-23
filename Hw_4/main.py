#!/usr/bin/env python3

import argparse
import argcomplete
import importlib
import os


def main():
    # Get a list of all files in the 'runners' directory
    runners = os.listdir('runners')
    configs = os.listdir('configs')

    # remove the .py extension from the files
    runners = [runner[:-3] for runner in runners]
    configs = [config[:-5] for config in configs]
    # remove __init__.py
    runners.remove('__init__')

    parser = argparse.ArgumentParser(description="Choose an example to train:")
    parser.add_argument("model", type=str, choices=runners,
                        help="Name of the example to train")
    parser.add_argument("config", type=str, choices=configs,
                        nargs='?', help="Path to the config file")

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    runner = importlib.import_module(f"runners.{args.model}")
    if args.config is None:
        print(f"You chose to train the {args.model} " +
              "model with the default config file.")
        runner.run()
    else:
        print(f"You chose to train the {args.model} " +
              "model with the config file at \
            {args.config}.")
        runner.run(args.config)


if __name__ == "__main__":
    main()
