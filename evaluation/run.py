"""Run every evaluation scenario and print a pass/fail table: python -m evaluation.run"""

import logging
import sys

from evaluation.scenarios import run_all


def main() -> int:
    # Tools log expected failures (invalid tickers, short history).
    logging.disable(logging.CRITICAL)
    results = run_all()
    width = max(len(r.name) for r in results)
    for r in results:
        print(f"{'PASS' if r.passed else 'FAIL'}  {r.name.ljust(width)}  {len(r.checks)} checks")
        for c in r.checks:
            if not c.passed:
                print(f"      - {c.name}: {c.detail}")
    passed = sum(r.passed for r in results)
    print(f"\n{passed}/{len(results)} scenarios passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
