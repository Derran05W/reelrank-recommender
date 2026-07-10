#!/usr/bin/env python3
"""Plot metric time series and distributions from a ReelRank experiment result directory.

Stub: plotting is implemented in a later phase.
"""
import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot ReelRank experiment results")
    parser.add_argument("result", help="experiment result directory")
    parser.add_argument("--out", default="plots", help="output directory for plots")
    parser.parse_args()
    print("plot_results: not implemented until later phases", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
