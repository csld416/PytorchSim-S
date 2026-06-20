#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from typing import Optional


def resolve_paths(trace_dir: str, trace_name: Optional[str]) -> tuple[str, str, str]:
    if not trace_name:
        trace_name = os.environ.get("TOGSIM_SSD_TRACE_NAME")
    if not trace_name:
        raise RuntimeError("TOGSIM_SSD_TRACE_NAME must be set or passed via --trace-name")
    base_dir = os.path.join(trace_dir, trace_name)
    input_dir = os.path.join(base_dir, "model_weight_traces")
    output_dir = os.path.join(base_dir, "ssd_latency")
    return base_dir, input_dir, output_dir


def add_latency_to_trace(src_path: str, dst_path: str, latency_ns: int) -> None:
    with open(src_path, "r", encoding="utf-8") as src, open(
        dst_path, "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        header = next(reader, None)
        if header is None:
            return

        latency_idx = None
        if "latency" in header:
            latency_idx = header.index("latency")
        else:
            header.append("latency")
            latency_idx = len(header) - 1
        writer.writerow(header)

        for row in reader:
            if latency_idx >= len(row):
                row.extend([""] * (latency_idx - len(row) + 1))
            row[latency_idx] = str(latency_ns)
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add a latency column to SSD model weight trace CSV files."
    )
    parser.add_argument(
        "--trace-dir",
        default="./ssd_traces",
        help="Base SSD trace directory (default: ./ssd_traces)",
    )
    parser.add_argument(
        "--trace-name",
        default=None,
        help="Trace name directory under trace-dir (defaults to $TOGSIM_SSD_TRACE_NAME)",
    )
    parser.add_argument(
        "--latency-ns",
        type=int,
        default=50000,
        help="Latency in ns to write into the latency column (default: 50000)",
    )
    args = parser.parse_args()

    _, input_dir, output_dir = resolve_paths(args.trace_dir, args.trace_name)
    if not os.path.isdir(input_dir):
        print(f"Input directory not found: {input_dir}")
        return 1

    os.makedirs(output_dir, exist_ok=True)

    wrote_any = False
    for name in os.listdir(input_dir):
        if not name.endswith(".csv"):
            continue
        src_path = os.path.join(input_dir, name)
        if not os.path.isfile(src_path):
            continue
        dst_path = os.path.join(output_dir, name)
        add_latency_to_trace(src_path, dst_path, args.latency_ns)
        wrote_any = True

    if not wrote_any:
        print(f"No CSV files found under {input_dir}")
        return 1

    print(f"Wrote latency traces to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
