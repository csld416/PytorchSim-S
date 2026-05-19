#!/usr/bin/env python3
import argparse
import glob
import os
import sys

try:
    import yaml
except ImportError:
    yaml = None


def parse_meta(meta_path):
    names = set()
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                name = line.split("=", 1)[0].strip()
                if name:
                    names.add(name)
    except OSError:
        return set()
    return names


def parse_attribute(attr_path):
    if yaml is None:
        raise RuntimeError("PyYAML is required. Please install with `pip install pyyaml`.")
    try:
        with open(attr_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except OSError:
        return set()
    if not isinstance(data, dict):
        return set()
    address_info = data.get("address_info", {})
    if isinstance(address_info, dict):
        return set(address_info.keys())
    return set()


def collect_weights(outputs_root):
    weights_union = set()
    per_hash = {}

    entries = []
    try:
        entries = os.listdir(outputs_root)
    except OSError as exc:
        raise RuntimeError(f"Failed to list outputs root: {exc}") from exc

    for entry in entries:
        hash_dir = os.path.join(outputs_root, entry)
        if not os.path.isdir(hash_dir):
            continue
        meta_path = os.path.join(hash_dir, "meta.txt")
        if not os.path.isfile(meta_path):
            continue

        meta_names = parse_meta(meta_path)
        if not meta_names:
            continue

        attr_paths = glob.glob(os.path.join(hash_dir, "runtime_*", "attribute", "0"))
        input_names = set()
        for attr_path in attr_paths:
            input_names |= parse_attribute(attr_path)

        weights = meta_names - input_names
        if weights:
            per_hash[entry] = weights
            weights_union |= weights

    return weights_union, per_hash


def write_weights_file(trace_dir, trace_name, weights, include_header):
    out_dir = os.path.join(trace_dir, trace_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "weights.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        if include_header:
            f.write("# Auto-generated weights list\n")
            f.write("# Lines starting with # are ignored\n")
        for name in sorted(weights):
            f.write(f"{name}\n")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate SSD weights.txt by excluding runtime inputs from meta.txt entries."
    )
    parser.add_argument(
        "--outputs-root",
        default="/workspace/PyTorchSim/outputs",
        help="Path to outputs directory (default: /workspace/PyTorchSim/outputs)",
    )
    parser.add_argument(
        "--trace-name",
        required=True,
        help="Trace name directory under /workspace/PyTorchSim/ssd_traces",
    )
    parser.add_argument(
        "--trace-dir",
        default="/workspace/PyTorchSim/ssd_traces",
        help="Base SSD trace directory (default: /workspace/PyTorchSim/ssd_traces)",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Do not write header comments in weights.txt",
    )

    args = parser.parse_args()

    weights, per_hash = collect_weights(args.outputs_root)
    if not weights:
        print("No weights found. Check outputs root and attribute files.")
        return 1

    out_path = write_weights_file(args.trace_dir, args.trace_name, weights, not args.no_header)
    print(f"Wrote {len(weights)} weight names to {out_path}")
    print(f"Scanned {len(per_hash)} output directories with meta.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
