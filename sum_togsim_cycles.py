#!/usr/bin/env python3
import argparse
import os
import re
import sys
from typing import Optional

TOTAL_RE = re.compile(r"Total execution cycles:\s*(\d+)")
LATENCY_RE = re.compile(r"\[SSD\]\s+latency replay enabled")


def resolve_dir(base_dir: str, trace_name: Optional[str]) -> str:
    if not trace_name:
        trace_name = os.environ.get("TOGSIM_SSD_TRACE_NAME")
    if not trace_name:
        raise RuntimeError("TOGSIM_SSD_TRACE_NAME must be set or passed via --trace-name")
    return os.path.join(base_dir, trace_name)


def sum_cycles(log_dir: str) -> tuple[int, int]:
    total = 0
    latency_total = 0
    for name in os.listdir(log_dir):
        if not name.endswith(".log"):
            continue
        path = os.path.join(log_dir, name)
        if not os.path.isfile(path):
            continue
        has_latency = False
        file_total = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not has_latency and LATENCY_RE.search(line):
                    has_latency = True
                if file_total is None:
                    match = TOTAL_RE.search(line)
                    if match:
                        file_total = int(match.group(1))

        if file_total is None:
            continue
        total += file_total
        if has_latency:
            latency_total += file_total
    return total, latency_total


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sum total execution cycles from TOGSim .log files."
    )
    parser.add_argument(
        "--base-dir",
        default="/workspace/PyTorchSim/togsim_results",
        help="Base TOGSim results directory (default: /workspace/PyTorchSim/togsim_results)",
    )
    parser.add_argument(
        "--trace-name",
        default=None,
        help="Trace name directory under base-dir (defaults to $TOGSIM_SSD_TRACE_NAME)",
    )
    args = parser.parse_args()

    try:
        log_dir = resolve_dir(args.base_dir, args.trace_name)
    except RuntimeError as exc:
        print(str(exc))
        return 1

    if not os.path.isdir(log_dir):
        print(f"Log directory not found: {log_dir}")
        return 1

    total, latency_total = sum_cycles(log_dir)
    print(f"total_execution_cycles=\n{total}")
    print(f"ssd_latency_execution_cycles=\n{latency_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
