"""Runtime host-address classification and logical SSD placement for weights.

The active manifest contains only weights that are currently materialized in
host-backed NPU memory. Logical SSD offsets remain stable by tensor name even
when streamed layers are unloaded and later allocations reuse their addresses.
"""

import os
import threading


_DEFAULT_ALIGNMENT = 4096
_registries = {}
_registries_lock = threading.Lock()


def _align_up(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment


class _PlacementRegistry:
    def __init__(self, output_dir, alignment):
        self.output_dir = output_dir
        self.alignment = alignment
        self.next_ssd_offset = 0
        self.placements = {}
        self.lock = threading.Lock()

    def _placement_for(self, name, size_bytes):
        existing = self.placements.get(name)
        if existing is not None:
            ssd_base, existing_size = existing
            if existing_size != size_bytes:
                raise RuntimeError(
                    f"Weight {name} changed size from {existing_size} to {size_bytes} bytes"
                )
            return ssd_base

        ssd_base = _align_up(self.next_ssd_offset, self.alignment)
        self.placements[name] = (ssd_base, size_bytes)
        self.next_ssd_offset = ssd_base + size_bytes
        return ssd_base

    @staticmethod
    def _atomic_write(path, lines):
        temporary = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
        with open(temporary, "w", encoding="utf-8") as output:
            output.writelines(lines)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)

    def write_tensors(self, named_tensors):
        active_rows = []
        with self.lock:
            for full_name, parameter in named_tensors:
                if parameter is None:
                    continue
                size_bytes = int(parameter.numel()) * int(parameter.element_size())
                if size_bytes == 0:
                    continue
                if not parameter.is_contiguous():
                    raise RuntimeError(
                        f"Weight {full_name} must be contiguous for SSD placement"
                    )
                host_base = int(parameter.data_ptr())
                if host_base == 0:
                    raise RuntimeError(
                        f"Weight {full_name} has no materialized storage"
                    )
                ssd_base = self._placement_for(full_name, size_bytes)
                active_rows.append(
                    (full_name, host_base, host_base + size_bytes, ssd_base, size_bytes)
                )

            os.makedirs(self.output_dir, exist_ok=True)
            active_lines = [
                "tensor_name\thost_base\thost_end_exclusive\tssd_base\tsize_bytes\n"
            ]
            active_lines.extend(
                f"{name}\t{host_base}\t{host_end}\t{ssd_base}\t{size_bytes}\n"
                for name, host_base, host_end, ssd_base, size_bytes in active_rows
            )
            self._atomic_write(
                os.path.join(self.output_dir, "model_weight_placements.tsv"),
                active_lines,
            )

            layout_lines = ["tensor_name\tssd_base\tsize_bytes\n"]
            layout_lines.extend(
                f"{name}\t{ssd_base}\t{size_bytes}\n"
                for name, (ssd_base, size_bytes) in sorted(
                    self.placements.items(), key=lambda item: item[1][0]
                )
            )
            self._atomic_write(
                os.path.join(self.output_dir, "model_weight_layout.tsv"),
                layout_lines,
            )


def _configured_registry():
    trace_dir = os.environ.get("TOGSIM_SSD_TRACE_DIR")
    trace_name = os.environ.get("TOGSIM_SSD_TRACE_NAME")
    if not trace_dir or not trace_name:
        return None, None

    alignment = int(
        os.environ.get("TOGSIM_SSD_PLACEMENT_ALIGNMENT", str(_DEFAULT_ALIGNMENT))
    )
    if alignment <= 0:
        raise ValueError("TOGSIM_SSD_PLACEMENT_ALIGNMENT must be positive")

    output_dir = os.path.join(trace_dir, trace_name)
    key = (os.path.abspath(output_dir), alignment)
    with _registries_lock:
        registry = _registries.get(key)
        if registry is None:
            registry = _PlacementRegistry(output_dir, alignment)
            _registries[key] = registry
    return registry, output_dir


def write_weight_placements(named_tensors):
    """Publish an iterable of ``(stable_name, tensor)`` weight placements."""
    registry, output_dir = _configured_registry()
    if registry is None:
        return None
    registry.write_tensors(named_tensors)
    return os.path.join(output_dir, "model_weight_placements.tsv")


def write_module_weight_placements(module, name_prefix):
    """Publish every recursively named parameter belonging to ``module``."""
    named_tensors = (
        (f"{name_prefix}.{parameter_name}", parameter)
        for parameter_name, parameter in module.named_parameters(recurse=True)
    )
    return write_weight_placements(named_tensors)
