"""
NPU simlet class for LegOSim × PyTorchSim integration.

LegOSimNPUInterceptor is a TOGSimulator-compatible object that can be installed
via torch.npu.set_tog_simulator().  When a compiled kernel performs a weight
MVIN, it sends a two-phase request to the DRAM simlet over LegOSim's sync
protocol and blocks until the GM resolves the timing.

Two-phase DMA request protocol per weight read
───────────────────────────────────────────────
                NPU simlet                         DRAM simlet
                (this file)                        (dram_simlet.py)

Phase 1  stdout ─► [INTERCMD] SEND npu→dram        ◄─ stdin
(request)  ◄─ stdin [INTERCMD] RESULT 1 pipe_req   stdout ─►
         write(pipe_req, pack_int64(nbytes))       read(pipe_req, 8) → nbytes
         stdout ─► [INTERCMD] WRITE cycle 8 bytes  ◄─ stdin
                   ← SYNC →                        [INTERCMD] READ 8 bytes

Phase 2  stdout ─► [INTERCMD] RECEIVE npu←dram     ◄─ stdin
(response)  ◄─ stdin [INTERCMD] RESULT 1 pipe_resp stdout ─►
         read(pipe_resp, 8) [dummy]                write(pipe_resp, dummy)
         [INTERCMD] READ nbytes real bytes ─►       ◄─ [INTERCMD] WRITE latency nbytes
                   ←── SYNC (DMA done cycle) ──────→

The DRAM simlet computes the latency independently based on its bandwidth
model.  LegOSim's GM synchronizes the timing and tells both simlets the
cycle at which the DMA completes.

Usage
─────
This class is used by application code that runs as an NPU simlet process:

    interceptor = LegOSimNPUInterceptor(npu_x=0, npu_y=0, dram_x=1, dram_y=0)
    interceptor.register_weights(model)

    with interceptor:
        output = compiled_layer(...)

    # Report final cycle to GM
    interceptor.finalize()

When LegOSim launches this process, its stdin/stdout are wired to the GM.
All logging must go to stderr.
"""

import sys
import logging
import yaml
import torch

from Simulator.legosim import protocol as proto

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="[NPU %(process)d] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# MLIRKernelArgs direction bitmasks — inlined to avoid circular import
# (Simulator.simulator → mlir_common → extension_codecache → Simulator.simulator)
_MLIR_ARGS_IN = 0x01
_MLIR_ARGS_OUT = 0x02
_MLIR_ARGS_INOUT = 0x04


class LegOSimNPUInterceptor:
    """
    Replaces TOGSimulator during inference; routes weight DMA through LegOSim.

    Parameters
    ----------
    npu_x, npu_y:
        Chip coordinates of this NPU simlet in the LegOSim topology.
    dram_x, dram_y:
        Chip coordinates of the DRAM simlet to send weight DMA requests to.
    """

    def __init__(self, npu_x: int, npu_y: int, dram_x: int, dram_y: int):
        self.npu_x = npu_x
        self.npu_y = npu_y
        self.dram_x = dram_x
        self.dram_y = dram_y
        self._current_cycle: int = 0
        self._weight_registry: dict[int, int] = {}  # data_ptr → byte_size
        self._kernel_count: int = 0
        self._dma_count: int = 0
        self._saved_simulator = None

    # ------------------------------------------------------------------
    # Weight registration
    # ------------------------------------------------------------------

    def register_weights(self, model: torch.nn.Module) -> None:
        """Register all parameters of *model* as weight tensors."""
        count = 0
        for p in model.parameters():
            self._weight_registry[p.data_ptr()] = p.numel() * p.element_size()
            count += 1
        logger.info(f"Registered {count} weight tensors "
                    f"({sum(self._weight_registry.values()) / 1e6:.1f} MB total)")

    def register_weight_tensor(self, tensor: torch.Tensor) -> None:
        """Register a single tensor as a weight."""
        self._weight_registry[tensor.data_ptr()] = (
            tensor.numel() * tensor.element_size()
        )

    # ------------------------------------------------------------------
    # TOGSimulator-compatible interface
    # ------------------------------------------------------------------

    def launch_kernel(self, device_index: int, stream_index: int,
                      tog_path: str, attribute_path: str,
                      timestamp: int = 0) -> int:
        """
        Handle one compiled-kernel dispatch.

        Reads the attribute YAML written by write_kernel_attribute_file().
        For each MVIN arg whose pointer matches a registered weight, sends a
        DMA request to the DRAM simlet and waits for the timing response.
        """
        kernel_id = self._kernel_count
        self._kernel_count += 1

        try:
            with open(attribute_path) as f:
                attr = yaml.safe_load(f)
        except OSError as e:
            logger.warning(f"Kernel {kernel_id}: cannot read {attribute_path}: {e}")
            return kernel_id

        address_info: dict = attr.get("address_info", {})
        arg_meta: dict = attr.get("arg_meta", {})

        for arg_key, ptr in address_info.items():
            meta = arg_meta.get(arg_key, {})
            direction = meta.get("direction", _MLIR_ARGS_IN)
            is_mvin = bool(direction & (_MLIR_ARGS_IN | _MLIR_ARGS_INOUT))
            if not is_mvin:
                continue
            if ptr not in self._weight_registry:
                continue

            nbytes = self._weight_registry[ptr]
            logger.info(f"Kernel {kernel_id}: weight DMA {arg_key} "
                        f"ptr=0x{ptr:x} {nbytes}B → requesting DRAM latency")

            synced_cycle = self._request_dma(nbytes)
            self._current_cycle = max(self._current_cycle, synced_cycle)
            self._dma_count += 1

        return kernel_id

    def device_synchronize(self, device_index: int) -> None:
        pass

    # ------------------------------------------------------------------
    # Context manager (mirrors TOGSimulator usage)
    # ------------------------------------------------------------------

    def __enter__(self) -> "LegOSimNPUInterceptor":
        self._saved_simulator = torch.npu.get_tog_simulator()
        torch.npu.set_tog_simulator(self)
        return self

    def __exit__(self, *_) -> None:
        torch.npu.synchronize()
        torch.npu.set_tog_simulator(self._saved_simulator)

    def finalize(self) -> int:
        """
        Report the accumulated cycle to GM.

        Call once after all inference is done.  Returns the GM-synchronized
        final cycle.
        """
        final_cycle = proto.cycle_sync(self._current_cycle)
        logger.info(f"NPU simlet finished: {self._kernel_count} kernels, "
                    f"{self._dma_count} weight DMAs, "
                    f"final_cycle={final_cycle}")
        return final_cycle

    # ------------------------------------------------------------------
    # Internal: two-phase DMA request/response via LegOSim protocol
    # ------------------------------------------------------------------

    def _request_dma(self, nbytes: int) -> int:
        """
        Exchange one weight DMA request with the DRAM simlet.

        Phase 1 — NPU sends nbytes to DRAM (request how much data is needed):
          NPU calls send_sync → writes nbytes to pipe → write_sync.
          DRAM calls receive_sync → reads nbytes from pipe → read_sync.

        Phase 2 — DRAM sends DMA timing back to NPU:
          DRAM calls send_sync → writes dummy payload → write_sync at latency cycle.
          NPU calls receive_sync → read_sync at current cycle → gets synced cycle.
        """
        # ------ Phase 1: send nbytes request to DRAM ------
        pipe_req = proto.send_sync(self.npu_x, self.npu_y,
                                   self.dram_x, self.dram_y)
        proto.write_pipe(pipe_req, proto.pack_int64(nbytes))
        proto.write_sync(self._current_cycle,
                         self.npu_x, self.npu_y,
                         self.dram_x, self.dram_y,
                         8, 0)  # 8-byte metadata payload

        # ------ Phase 2: receive DMA timing from DRAM ------
        pipe_resp = proto.receive_sync(self.npu_x, self.npu_y,
                                       self.dram_x, self.dram_y)
        proto.read_pipe(pipe_resp, 8)  # discard dummy payload
        synced_cycle = proto.read_sync(self._current_cycle,
                                       self.npu_x, self.npu_y,
                                       self.dram_x, self.dram_y,
                                       nbytes, 0)
        logger.info(f"DMA {nbytes}B complete at cycle {synced_cycle}")
        return synced_cycle
