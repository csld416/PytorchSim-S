"""
Dummy constant-bandwidth DRAM simlet for LegOSim × PyTorchSim integration.

Runs as an independent LegOSim simlet process.  Handles weight DMA requests
from the NPU simlet using a simple bandwidth model:

    latency_cycles = ceil(nbytes / bytes_per_cycle)
    bytes_per_cycle = bandwidth_gbps * 1e9 / core_freq_ghz * 1e9

Protocol per request (matches npu_simlet.py _request_dma()):
─────────────────────────────────────────────────────────────
Phase 1 — receive nbytes from NPU (request phase):
  DRAM: receive_sync(dram, from=npu)  → pipe_req  (GM pairs with NPU's send_sync)
  DRAM: read_pipe(pipe_req, 8)        → nbytes    (NPU wrote this)
  DRAM: read_sync(cycle, dram, npu, 8, 0)         (timing ack for request)

Phase 2 — send DMA timing back to NPU (response phase):
  DRAM: send_sync(dram, to=npu)       → pipe_resp (GM pairs with NPU's receive_sync)
  DRAM: write_pipe(pipe_resp, dummy)              (dummy payload, NPU discards it)
  DRAM: write_sync(done_cycle, dram, npu, nbytes, 0)  ← GM uses this to sync NPU

The GM resolves max(NPU_read_cycle, DRAM_write_cycle + noc_latency) and sends
the synchronized completion cycle to both simlets.

Usage (invoked by LegOSim GM):
  python -m Simulator.legosim.dram_simlet \\
         <dram_x> <dram_y> <npu_x> <npu_y> <bandwidth_gbps> [<core_freq_ghz>]
"""

import sys
import math
import logging

from Simulator.legosim import protocol as proto

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="[DRAM %(process)d] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(dram_x: int, dram_y: int,
        npu_x: int, npu_y: int,
        bandwidth_gbps: float,
        core_freq_ghz: float = 1.0) -> None:
    """
    Main event loop for the DRAM simlet.

    Runs until the GM closes stdin (simulation complete or NPU simlet exited).
    """
    bytes_per_cycle: float = (bandwidth_gbps * 1e9) / (core_freq_ghz * 1e9)
    current_cycle: int = 0
    request_count: int = 0

    logger.info(
        f"DRAM simlet at ({dram_x},{dram_y}), partner NPU at ({npu_x},{npu_y}), "
        f"bandwidth={bandwidth_gbps} GB/s @ {core_freq_ghz} GHz "
        f"→ {bytes_per_cycle:.2f} B/cycle"
    )

    while True:
        try:
            # ── Phase 1: receive the nbytes request sent by the NPU simlet ──
            pipe_req = proto.receive_sync(dram_x, dram_y, npu_x, npu_y)
            data = proto.read_pipe(pipe_req, 8)
            nbytes = proto.unpack_int64(data)
            # Ack the request-metadata timing (8-byte payload)
            sync1 = proto.read_sync(current_cycle, dram_x, dram_y,
                                    npu_x, npu_y, 8, 0)
            current_cycle = sync1

            # ── Compute DMA latency ──
            latency_cycles = math.ceil(nbytes / bytes_per_cycle)
            dma_done_cycle = current_cycle + latency_cycles

            # ── Phase 2: send DMA completion timing back to NPU ──
            pipe_resp = proto.send_sync(dram_x, dram_y, npu_x, npu_y)
            proto.write_pipe(pipe_resp, b"\x00" * 8)  # dummy payload
            sync2 = proto.write_sync(dma_done_cycle, dram_x, dram_y,
                                     npu_x, npu_y, nbytes, 0)
            current_cycle = sync2

            request_count += 1
            logger.info(
                f"Request #{request_count}: {nbytes:,}B "
                f"→ latency={latency_cycles} cycles "
                f"(done at {dma_done_cycle}, synced={sync2})"
            )

        except EOFError:
            logger.info(
                f"GM closed connection — served {request_count} DMA requests, "
                f"final_cycle={current_cycle}"
            )
            break


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print(
            f"Usage: python -m Simulator.legosim.dram_simlet "
            f"<dram_x> <dram_y> <npu_x> <npu_y> <bandwidth_gbps> [<core_freq_ghz>]",
            file=sys.stderr,
        )
        sys.exit(1)

    run(
        dram_x=int(sys.argv[1]),
        dram_y=int(sys.argv[2]),
        npu_x=int(sys.argv[3]),
        npu_y=int(sys.argv[4]),
        bandwidth_gbps=float(sys.argv[5]),
        core_freq_ghz=float(sys.argv[6]) if len(sys.argv) > 6 else 1.0,
    )
