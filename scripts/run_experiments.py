#!/usr/bin/env python3
"""Run a batch of ReelRank experiments from a config directory.

Stub: the experiment runner is implemented in a later phase.
"""
import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ReelRank experiments")
    parser.add_argument("--configs", default="configs", help="directory of experiment configs")
    parser.add_argument("--results", default="results", help="output directory for results")
    parser.parse_args()
    print("run_experiments: not implemented until later phases", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
