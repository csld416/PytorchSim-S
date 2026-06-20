import argparse
import os
import re
from typing import List, Tuple

RANGE_RE = re.compile(r"\bbase=(\d+)\b.*\bend=(\d+)\b")

def parse_ranges(path: str) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    with open(path, "r") as f:
        for line in f:
            match = RANGE_RE.search(line)
            if not match:
                continue
            base = int(match.group(1))
            end = int(match.group(2))
            if end < base:
                base, end = end, base
            ranges.append((base, end))
    return ranges


def merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    ranges = sorted(ranges, key=lambda r: (r[0], r[1]))
    merged: List[Tuple[int, int]] = [ranges[0]]
    for base, end in ranges[1:]:
        last_base, last_end = merged[-1]
        if base <= last_end:
            merged[-1] = (last_base, max(last_end, end))
            continue
        if base == last_end:
            merged[-1] = (last_base, end)
            continue
        merged.append((base, end))
    return merged


def default_paths() -> Tuple[str, str]:
    trace_dir = os.environ.get("TOGSIM_SSD_TRACE_DIR")
    trace_name = os.environ.get("TOGSIM_SSD_TRACE_NAME")
    if not trace_dir or not trace_name:
        raise RuntimeError("TOGSIM_SSD_TRACE_DIR and TOGSIM_SSD_TRACE_NAME must be set")
    base_dir = os.path.join(trace_dir, trace_name)
    input_path = os.path.join(base_dir, "model_weight_ranges.txt")
    output_path = os.path.join(base_dir, "model_weight_ranges_merged.txt")
    return input_path, output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge model weight address ranges")
    parser.add_argument("--input", dest="input_path", default=None, help="Path to model_weight_ranges.txt")
    parser.add_argument("--output", dest="output_path", default=None, help="Output path for merged ranges")
    args = parser.parse_args()

    if args.input_path is None or args.output_path is None:
        input_path, output_path = default_paths()
    else:
        input_path = args.input_path
        output_path = args.output_path

    ranges = parse_ranges(input_path)
    merged = merge_ranges(ranges)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for base, end in merged:
            f.write(f"{base}\t{end}\n")

    print(f"Parsed {len(ranges)} ranges, merged to {len(merged)} ranges.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
