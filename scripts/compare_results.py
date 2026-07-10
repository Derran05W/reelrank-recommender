#!/usr/bin/env python3
"""Compare metric summaries across two or more ReelRank experiment result directories.

Stub: comparison is implemented in a later phase.
"""
import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ReelRank experiment results")
    parser.add_argument("results", nargs="*", help="experiment result directories to compare")
    parser.parse_args()
    print("compare_results: not implemented until later phases", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
