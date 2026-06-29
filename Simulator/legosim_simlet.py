"""
LegOSim integration simlet for PyTorchSim.

DummyWeightSimlet replaces TOGSimulator for kernel dispatch during inference.
For every kernel that performs a DMA read from a registered model weight tensor,
it applies a constant-bandwidth latency model instead of spawning a TOGSim
subprocess.  All other DMA traffic (activations, KV-cache) contributes zero
latency by default — plug a real memory simlet in its place as needed.

Typical usage::

    simlet = DummyWeightSimlet(bandwidth_gbps=1000.0)
    simlet.register_weights(model)

    with simlet:
        output = compiled_layer(hidden_state)

    print(f"Weight DMA latency: {simlet.get_total_latency_ns():.1f} ns")

The simlet hooks into PyTorchSim via ``torch.npu.set_tog_simulator()``, which is
the same mechanism used by TOGSimulator.  No changes to compilation or codegen
are required — only ``write_kernel_attribute_file`` is extended to emit an
``arg_meta`` section (direction + byte-size per argument) that the simlet reads.
"""

import logging

import torch
import yaml

# MLIRKernelArgs direction bitmask constants — inlined here to avoid the
# Simulator ↔ mlir_common circular import that exists in the broader codebase.
_MLIR_ARGS_IN = 0x01
_MLIR_ARGS_OUT = 0x02
_MLIR_ARGS_INOUT = 0x04

logger = logging.getLogger(__name__)


class DummyWeightSimlet:
    """
    Simlet that replaces TOGSimulator and models weight DMA with constant latency.

    Parameters
    ----------
    bandwidth_gbps:
        Peak weight-DRAM read bandwidth in GB/s.  The latency for a weight DMA
        is ``bytes / (bandwidth_gbps * 1e9 / 1e9)`` nanoseconds, i.e.
        ``bytes / bandwidth_gbps`` ns (since 1 GB/s == 1 byte/ns).
    stable_threshold:
        Number of times a MVIN arg address must be seen before it is
        auto-detected as a weight (used only when register_weights() was not
        called for that tensor).
    """

    def __init__(self, bandwidth_gbps: float = 1000.0, stable_threshold: int = 2):
        # 1 GB/s == 1 byte/ns, so bandwidth_gbps == bytes/ns directly
        self._bw_bytes_per_ns: float = bandwidth_gbps
        self._stable_threshold: int = stable_threshold

        # Explicitly registered weight addresses: data_ptr -> byte_size
        self._weight_registry: dict[int, int] = {}

        # Auto-detection state
        # ptr -> (seen_count, byte_size_or_None)
        self._candidate: dict[int, list] = {}

        # Accumulated latency across all launch_kernel() calls since last reset
        self._accumulated_latency_ns: float = 0.0
        self._kernel_count: int = 0
        self._weight_dma_count: int = 0

        # Saved simulator for context-manager restore
        self._saved_simulator = None

    # ------------------------------------------------------------------
    # Public API: weight registration
    # ------------------------------------------------------------------

    def register_weights(self, model: torch.nn.Module) -> None:
        """Register every parameter of *model* as a weight tensor."""
        count = 0
        for param in model.parameters():
            self._weight_registry[param.data_ptr()] = param.numel() * param.element_size()
            count += 1
        logger.debug(f"[DummySimlet] Registered {count} weight tensors from model.")

    def register_weight_tensor(self, tensor: torch.Tensor) -> None:
        """Register a single tensor as a weight."""
        self._weight_registry[tensor.data_ptr()] = tensor.numel() * tensor.element_size()

    # ------------------------------------------------------------------
    # TOGSimulator-compatible interface (called by torch.npu.launch_kernel)
    # ------------------------------------------------------------------

    def launch_kernel(
        self,
        device_index: int,
        stream_index: int,
        tog_path: str,
        attribute_path: str,
        timestamp: int = 0,
    ) -> int:
        """
        Handle a kernel launch.

        Reads the attribute YAML written by ``write_kernel_attribute_file``,
        identifies MVIN arguments that correspond to registered (or auto-detected)
        weight tensors, and accumulates a constant-latency estimate for each.

        Returns the sequential kernel ID assigned to this launch.
        """
        kernel_id = self._kernel_count
        self._kernel_count += 1

        try:
            latency_ns = self._kernel_weight_latency(attribute_path)
        except Exception as exc:
            logger.warning(
                f"[DummySimlet] Kernel {kernel_id}: failed to compute latency — {exc}"
            )
            latency_ns = 0.0

        self._accumulated_latency_ns += latency_ns
        if latency_ns > 0:
            self._weight_dma_count += 1
        logger.debug(
            f"[DummySimlet] Kernel {kernel_id}: weight_latency={latency_ns:.1f} ns "
            f"(running_total={self._accumulated_latency_ns:.1f} ns)"
        )
        return kernel_id

    def device_synchronize(self, device_index: int) -> None:
        """No-op — no real device process to synchronize."""
        pass

    # ------------------------------------------------------------------
    # Result queries
    # ------------------------------------------------------------------

    def get_total_latency_ns(self) -> float:
        """Total accumulated weight-DMA latency in nanoseconds."""
        return self._accumulated_latency_ns

    def get_kernel_count(self) -> int:
        """Number of kernels dispatched since last reset."""
        return self._kernel_count

    def get_weight_dma_count(self) -> int:
        """Number of kernels that touched at least one weight tensor."""
        return self._weight_dma_count

    def reset_latency(self) -> None:
        """Reset the latency counter (call between separate inference runs)."""
        self._accumulated_latency_ns = 0.0
        self._kernel_count = 0
        self._weight_dma_count = 0

    # ------------------------------------------------------------------
    # Context-manager interface (mirrors TOGSimulator)
    # ------------------------------------------------------------------

    def __enter__(self) -> "DummyWeightSimlet":
        self._saved_simulator = torch.npu.get_tog_simulator()
        torch.npu.set_tog_simulator(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        torch.npu.synchronize()
        torch.npu.set_tog_simulator(self._saved_simulator)
        logger.info(
            f"[DummySimlet] Session done: {self._kernel_count} kernels dispatched, "
            f"{self._weight_dma_count} had weight DMA, "
            f"total weight-DMA latency = {self._accumulated_latency_ns:.1f} ns"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _kernel_weight_latency(self, attribute_path: str) -> float:
        """
        Return the total dummy latency (ns) for weight MVIN ops in this kernel.

        The attribute YAML (written by ``write_kernel_attribute_file``) contains:

        .. code-block:: yaml

            address_info:
              arg0: <data_ptr>
              arg1: <data_ptr>
            arg_meta:            # written only when arg_attributes is passed
              arg0: {direction: 1, bytes: 67108864}
              arg1: {direction: 2, bytes: 8192}

        When ``arg_meta`` is absent (old-format attribute files), every arg is
        treated as a candidate MVIN and auto-detection logic applies.
        """
        with open(attribute_path) as f:
            attr = yaml.safe_load(f)

        address_info: dict = attr.get("address_info", {})
        arg_meta: dict = attr.get("arg_meta", {})
        has_meta = bool(arg_meta)

        total_bytes = 0

        for arg_key, ptr in address_info.items():
            # ----------------------------------------------------------
            # Direction filtering: skip pure-output (MVOUT) args so we
            # don't count output DMA as weight traffic.
            # ----------------------------------------------------------
            direction = _MLIR_ARGS_IN  # default: assume input
            byte_size: int | None = None

            if has_meta and arg_key in arg_meta:
                meta = arg_meta[arg_key]
                direction = meta.get("direction", _MLIR_ARGS_IN)
                byte_size = meta.get("bytes")

            is_mvin = bool(direction & (_MLIR_ARGS_IN | _MLIR_ARGS_INOUT))
            is_mvout_only = bool(direction & _MLIR_ARGS_OUT) and not is_mvin
            if is_mvout_only:
                continue  # Output-only: not a weight read

            # ----------------------------------------------------------
            # Weight classification
            # ----------------------------------------------------------
            if ptr in self._weight_registry:
                nbytes = self._weight_registry[ptr]
                total_bytes += nbytes
                logger.debug(
                    f"[DummySimlet]   {arg_key} 0x{ptr:x}: weight hit, {nbytes} B"
                )
                continue

            # Auto-detection: track address stability
            if ptr not in self._candidate:
                self._candidate[ptr] = [1, byte_size]
            else:
                self._candidate[ptr][0] += 1
                if byte_size is not None and self._candidate[ptr][1] is None:
                    self._candidate[ptr][1] = byte_size

            seen, tracked_size = self._candidate[ptr]
            if seen >= self._stable_threshold and tracked_size is not None:
                # Seen enough times with a known size → promote to weight
                self._weight_registry[ptr] = tracked_size
                total_bytes += tracked_size
                logger.debug(
                    f"[DummySimlet]   {arg_key} 0x{ptr:x}: auto-detected weight "
                    f"(seen {seen}×), {tracked_size} B"
                )

        return total_bytes / self._bw_bytes_per_ns if total_bytes else 0.0
