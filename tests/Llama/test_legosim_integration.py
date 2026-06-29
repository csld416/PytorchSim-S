"""
Integration test: PyTorchSim × LegOSim two-simlet architecture.

Demonstrates the proper LegOSim integration where PyTorchSim and the dummy
DRAM model are two independent simlets that communicate through LegOSim's
Unified Integration Interface (UII) sync protocol.

─────────────────────────────────────────────────────────────────
Architecture
─────────────────────────────────────────────────────────────────

  ┌──────────────────────────────────────────────────────────┐
  │  LegOSim Global Manager (GM)                             │
  │  - Manages chip topology                                 │
  │  - Routes [INTERCMD] messages between simlets            │
  │  - Resolves timing via WRITE/READ synchronization        │
  └───────────────┬──────────────────────────┬───────────────┘
                  │ stdin/stdout             │ stdin/stdout
        ┌─────────▼────────┐      ┌─────────▼──────────────┐
        │  NPU simlet      │      │  DRAM simlet           │
        │  (0, 0)          │      │  (1, 0)                │
        │                  │      │                        │
        │  PyTorchSim      │      │  Dummy const-BW model  │
        │  torch.compile   │      │  bandwidth_gbps param  │
        │  + TOG intercept │      │                        │
        └──────────────────┘      └────────────────────────┘

For each weight MVIN kernel:
  NPU ─[SEND nbytes]──► DRAM  (Phase 1: request)
  NPU ◄─[WRITE latency]─ DRAM  (Phase 2: response with DMA timing)
  GM resolves timing, both get SYNC cycle.

─────────────────────────────────────────────────────────────────
Standalone test
─────────────────────────────────────────────────────────────────
Without a real LegOSim GM, this file includes a MockGM that:
  - Creates named pipes (FIFOs) for each data direction
  - Routes [INTERCMD] messages between two threads
  - Applies simplified timing (max of both sides + NoC overhead)

Run:
  python tests/Llama/test_legosim_integration.py \\
         --bandwidth_gbps 256 --iters 1

LegOSim YAML configuration (for real GM):
  proc:
    - cmd: ["python", "-m", "Simulator.legosim.npu_simlet_entry", ...]
      x: 0
      y: 0
    - cmd: ["python", "-m", "Simulator.legosim.dram_simlet",
            "1", "0", "0", "0", "256.0", "1.0"]
      x: 1
      y: 0
"""

import os
import sys
import math
import struct
import threading
import tempfile
import argparse
import logging

import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import PyTorchSimFrontend.mlir.mlir_lowering  # registers NPU aten overrides

from Simulator.legosim.npu_simlet import LegOSimNPUInterceptor
from Simulator.legosim.dram_simlet import run as dram_simlet_run

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Mock GM: allows testing both simlets in-process without a real LegOSim GM
# ═══════════════════════════════════════════════════════════════════════════

class MockGM:
    """
    Minimal LegOSim Global Manager for standalone testing.

    Wires two simlets' stdin/stdout together using OS pipes and routes
    [INTERCMD] messages according to the LegOSim sync protocol.

    Only implements the SEND/RECEIVE/WRITE/READ/CYCLE/SYNC subset needed by
    LegOSimNPUInterceptor and dram_simlet.run().
    """

    def __init__(self, noc_cycles: int = 0):
        """
        Parameters
        ----------
        noc_cycles:
            Fixed NoC latency added to every WRITE→READ transfer (in cycles).
        """
        self._noc_cycles = noc_cycles
        # Pending SEND commands keyed by (src_x, src_y, dst_x, dst_y)
        self._pending_sends: dict[tuple, dict] = {}
        # Pending RECEIVE commands keyed by (dst_x, dst_y, src_x, src_y)
        self._pending_receives: dict[tuple, dict] = {}
        # Pending WRITE commands keyed by (src_x, src_y, dst_x, dst_y)
        self._pending_writes: dict[tuple, dict] = {}
        # Pending READ commands keyed by (dst_x, dst_y, src_x, src_y)
        self._pending_reads: dict[tuple, dict] = {}
        self._lock = threading.Lock()

    def _pipe_name(self, src_x, src_y, dst_x, dst_y) -> str:
        return f"./buffer{src_x}_{src_y}_{dst_x}_{dst_y}"

    # ------------------------------------------------------------------
    # Route messages from one simlet to the GM and handle responses
    # ------------------------------------------------------------------

    def handle(self, line: str, stdin_w, partner_stdin_w) -> None:
        """
        Process one [INTERCMD] line from a simlet.
        Write responses to stdin_w (the originating simlet's stdin).
        """
        line = line.strip()
        if not line.startswith("[INTERCMD]"):
            return
        parts = line[len("[INTERCMD]"):].strip().split()
        cmd = parts[0]

        if cmd == "SEND":
            src_x, src_y, dst_x, dst_y = map(int, parts[1:5])
            key = (src_x, src_y, dst_x, dst_y)
            rkey = (dst_x, dst_y, src_x, src_y)
            pipe = self._pipe_name(src_x, src_y, dst_x, dst_y)
            # Create FIFO if not exists
            if not os.path.exists(pipe):
                os.mkfifo(pipe)
            with self._lock:
                if rkey in self._pending_receives:
                    recv_stdin = self._pending_receives.pop(rkey)
                    result = f"[INTERCMD] RESULT 1 {pipe}\n"
                    stdin_w.write(result)
                    stdin_w.flush()
                    recv_stdin.write(result)
                    recv_stdin.flush()
                else:
                    self._pending_sends[key] = {
                        "pipe": pipe, "stdin_w": stdin_w
                    }

        elif cmd == "RECEIVE":
            dst_x, dst_y, src_x, src_y = map(int, parts[1:5])
            key = (dst_x, dst_y, src_x, src_y)
            skey = (src_x, src_y, dst_x, dst_y)
            pipe = self._pipe_name(src_x, src_y, dst_x, dst_y)
            if not os.path.exists(pipe):
                os.mkfifo(pipe)
            with self._lock:
                if skey in self._pending_sends:
                    send_stdin = self._pending_sends.pop(skey)["stdin_w"]
                    result = f"[INTERCMD] RESULT 1 {pipe}\n"
                    stdin_w.write(result)
                    stdin_w.flush()
                    send_stdin.write(result)
                    send_stdin.flush()
                else:
                    self._pending_receives[key] = stdin_w

        elif cmd == "WRITE":
            cycle, src_x, src_y, dst_x, dst_y, nbytes = (
                int(parts[1]), int(parts[2]), int(parts[3]),
                int(parts[4]), int(parts[5]), int(parts[6])
            )
            key = (src_x, src_y, dst_x, dst_y)
            rkey = (dst_x, dst_y, src_x, src_y)
            with self._lock:
                if rkey in self._pending_reads:
                    read_info = self._pending_reads.pop(rkey)
                    synced = max(cycle + self._noc_cycles, read_info["cycle"])
                    sync_msg = f"[INTERCMD] SYNC {synced}\n"
                    stdin_w.write(sync_msg)
                    stdin_w.flush()
                    read_info["stdin_w"].write(sync_msg)
                    read_info["stdin_w"].flush()
                else:
                    self._pending_writes[key] = {
                        "cycle": cycle, "stdin_w": stdin_w
                    }

        elif cmd == "READ":
            cycle, dst_x, dst_y, src_x, src_y, nbytes = (
                int(parts[1]), int(parts[2]), int(parts[3]),
                int(parts[4]), int(parts[5]), int(parts[6])
            )
            key = (dst_x, dst_y, src_x, src_y)
            wkey = (src_x, src_y, dst_x, dst_y)
            with self._lock:
                if wkey in self._pending_writes:
                    write_info = self._pending_writes.pop(wkey)
                    synced = max(write_info["cycle"] + self._noc_cycles, cycle)
                    sync_msg = f"[INTERCMD] SYNC {synced}\n"
                    stdin_w.write(sync_msg)
                    stdin_w.flush()
                    write_info["stdin_w"].write(sync_msg)
                    write_info["stdin_w"].flush()
                else:
                    self._pending_reads[key] = {
                        "cycle": cycle, "stdin_w": stdin_w
                    }

        elif cmd == "CYCLE":
            cycle = int(parts[1])
            stdin_w.write(f"[INTERCMD] SYNC {cycle}\n")
            stdin_w.flush()


def _pump_simlet(simlet_stdout_r, gm: MockGM, simlet_stdin_w, partner_stdin_w):
    """Read [INTERCMD] lines from a simlet and route them through the mock GM."""
    for line in simlet_stdout_r:
        if line.startswith("[INTERCMD]"):
            gm.handle(line, simlet_stdin_w, partner_stdin_w)


# ═══════════════════════════════════════════════════════════════════════════
# Simlet runner helpers
# ═══════════════════════════════════════════════════════════════════════════

def _run_npu_simlet_thread(npu_interceptor, compiled_layer, inputs,
                           real_stdin_r, real_stdout_w):
    """Run NPU inference with the interceptor in a thread with redirected I/O."""
    # Redirect protocol I/O for this thread — we swap sys.stdin/stdout
    # temporarily so the protocol module (which uses sys.stdout/stdin) works.
    old_stdout = sys.stdout
    old_stdin = sys.stdin
    sys.stdout = real_stdout_w
    sys.stdin = real_stdin_r
    try:
        with npu_interceptor:
            hs, pos, attn, pos_emb = inputs
            _ = compiled_layer(
                hidden_states=hs,
                attention_mask=attn,
                position_ids=pos,
                position_embeddings=pos_emb,
            )
        npu_interceptor.finalize()
    finally:
        sys.stdout = old_stdout
        sys.stdin = old_stdin


def _run_dram_simlet_thread(dram_x, dram_y, npu_x, npu_y,
                             bandwidth_gbps, core_freq_ghz,
                             real_stdin_r, real_stdout_w):
    """Run DRAM simlet in a thread with redirected I/O."""
    old_stdout = sys.stdout
    old_stdin = sys.stdin
    sys.stdout = real_stdout_w
    sys.stdin = real_stdin_r
    try:
        dram_simlet_run(dram_x, dram_y, npu_x, npu_y,
                        bandwidth_gbps, core_freq_ghz)
    except Exception as e:
        logger.error(f"DRAM simlet error: {e}")
    finally:
        sys.stdout = old_stdout
        sys.stdin = old_stdin


# ═══════════════════════════════════════════════════════════════════════════
# Model / input construction
# ═══════════════════════════════════════════════════════════════════════════

def _make_llama_decoder(dtype: torch.dtype, device: torch.device):
    cfg = LlamaConfig(
        _name_or_path="custom-llama",
        architectures=["LlamaForCausalLM"],
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=4096,
        initializer_range=0.02,
        intermediate_size=11008,
        max_position_embeddings=4096,
        mlp_bias=False,
        model_type="llama",
        num_attention_heads=32,
        num_hidden_layers=1,
        num_key_value_heads=32,
        pretraining_tp=1,
        rms_norm_eps=1e-6,
        rope_scaling=None,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        torch_dtype=dtype,
        transformers_version="4.43.4",
        use_cache=True,
        vocab_size=8192,
        _attn_implementation="sdpa",
    )
    layer = LlamaDecoderLayer(cfg, layer_idx=0).eval()
    return layer.to(dtype=dtype, device=device), cfg


def _make_inputs(batch, seq_len, cfg, dtype, device):
    g = torch.Generator().manual_seed(42)
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    hidden_states = torch.randn(batch, seq_len, cfg.hidden_size,
                                generator=g, dtype=dtype)
    position_ids = (torch.arange(seq_len, dtype=torch.long)
                    .unsqueeze(0).expand(batch, -1))
    attention_mask = torch.zeros(batch, 1, seq_len, seq_len, dtype=dtype)
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    attention_mask.masked_fill_(mask, torch.finfo(dtype).min)
    cos = torch.randn(1, seq_len, head_dim, generator=g, dtype=dtype)
    sin = torch.randn(1, seq_len, head_dim, generator=g, dtype=dtype)

    return (
        hidden_states.to(device),
        position_ids.to(device),
        attention_mask.to(device),
        (cos.to(device), sin.to(device)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main test
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_two_simlet_test(
    device: torch.device,
    batch: int = 1,
    seq_len: int = 32,
    dtype: str = "float32",
    bandwidth_gbps: float = 256.0,
    core_freq_ghz: float = 1.0,
    noc_cycles: int = 0,
    n_warmup: int = 0,
    n_iters: int = 1,
):
    """
    Run LlamaDecoderLayer with the two-simlet LegOSim architecture.

    The NPU simlet (running in this process with stdin/stdout redirected) and
    the DRAM simlet (running in a background thread) communicate through a
    MockGM that implements the LegOSim sync protocol.

    Parameters
    ----------
    bandwidth_gbps:
        Weight DRAM bandwidth modeled by the DRAM simlet.
    core_freq_ghz:
        NPU clock frequency for cycles↔time conversion.
    noc_cycles:
        NoC latency added by MockGM to each WRITE/READ transfer.
    """
    print(f"\n[LegOSim Two-Simlet Integration Test]")
    print(f"  device={device}, batch={batch}, seq_len={seq_len}, dtype={dtype}")
    print(f"  bandwidth={bandwidth_gbps} GB/s @ {core_freq_ghz} GHz, "
          f"NoC={noc_cycles} cycles")
    print(f"  warmup={n_warmup}, iters={n_iters}")

    dtype_map = {"float32": torch.float32,
                 "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[dtype]

    model, cfg = _make_llama_decoder(torch_dtype, device)
    compiled_layer = torch.compile(model, dynamic=False)
    inputs = _make_inputs(batch, seq_len, cfg, torch_dtype, device)

    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"  Model: {sum(1 for _ in model.parameters())} tensors, "
          f"{weight_bytes / 1e6:.1f} MB")

    # Warm-up: torch.compile traces on first call; simlets NOT active
    for i in range(n_warmup):
        hs, pos, attn, pos_emb = inputs
        _ = compiled_layer(
            hidden_states=hs,
            attention_mask=attn,
            position_ids=pos,
            position_embeddings=pos_emb,
        )
        print(f"  warmup {i}: done")

    # ------------------------------------------------------------------
    # Timed iterations with both simlets active via MockGM
    # ------------------------------------------------------------------
    results = []
    for it in range(n_iters):
        # ---- Create communication pipes (4 text-mode pipe pairs) ----
        # Each pair: (read_fd, write_fd) → wrapped as text file objects
        npu_to_gm_r, npu_to_gm_w = os.pipe()
        gm_to_npu_r, gm_to_npu_w = os.pipe()
        dram_to_gm_r, dram_to_gm_w = os.pipe()
        gm_to_dram_r, gm_to_dram_w = os.pipe()

        npu_stdout_r = os.fdopen(npu_to_gm_r, "r", buffering=1)
        npu_stdout_w = os.fdopen(npu_to_gm_w, "w", buffering=1)
        npu_stdin_r  = os.fdopen(gm_to_npu_r, "r", buffering=1)
        npu_stdin_w  = os.fdopen(gm_to_npu_w, "w", buffering=1)

        dram_stdout_r = os.fdopen(dram_to_gm_r, "r", buffering=1)
        dram_stdout_w = os.fdopen(dram_to_gm_w, "w", buffering=1)
        dram_stdin_r  = os.fdopen(gm_to_dram_r, "r", buffering=1)
        dram_stdin_w  = os.fdopen(gm_to_dram_w, "w", buffering=1)

        gm = MockGM(noc_cycles=noc_cycles)

        # ---- Start GM pump threads ----
        npu_pump = threading.Thread(
            target=_pump_simlet,
            args=(npu_stdout_r, gm, npu_stdin_w, dram_stdin_w),
            daemon=True,
        )
        dram_pump = threading.Thread(
            target=_pump_simlet,
            args=(dram_stdout_r, gm, dram_stdin_w, npu_stdin_w),
            daemon=True,
        )

        # ---- Start DRAM simlet thread ----
        interceptor = LegOSimNPUInterceptor(
            npu_x=0, npu_y=0, dram_x=1, dram_y=0
        )
        interceptor.register_weights(model)

        dram_thread = threading.Thread(
            target=_run_dram_simlet_thread,
            args=(1, 0, 0, 0, bandwidth_gbps, core_freq_ghz,
                  dram_stdin_r, dram_stdout_w),
            daemon=True,
        )

        npu_pump.start()
        dram_pump.start()
        dram_thread.start()

        # ---- Run NPU simlet in main thread with redirected I/O ----
        _run_npu_simlet_thread(
            interceptor, compiled_layer, inputs, npu_stdin_r, npu_stdout_w
        )

        # ---- Tear down: close write ends to unblock DRAM simlet ----
        npu_stdout_w.close()
        dram_stdin_w.close()
        dram_thread.join(timeout=5.0)
        npu_pump.join(timeout=2.0)
        dram_pump.join(timeout=2.0)

        # Read-ends were consumed by threads; close remaining handles
        for fobj in (npu_stdout_r, npu_stdin_r, npu_stdin_w,
                     dram_stdout_r, dram_stdout_w, dram_stdin_r):
            try:
                fobj.close()
            except OSError:
                pass

        final_cycle = interceptor._current_cycle
        dma_count = interceptor._dma_count
        kernel_count = interceptor._kernel_count
        bytes_per_cycle = (bandwidth_gbps * 1e9) / (core_freq_ghz * 1e9)
        latency_ns = (final_cycle / core_freq_ghz) if core_freq_ghz > 0 else 0.0

        print(
            f"  iter {it}: kernels={kernel_count}, weight_dma={dma_count}, "
            f"final_cycle={final_cycle}, "
            f"latency≈{latency_ns:.1f} ns ({latency_ns/1e6:.3f} ms)"
        )
        results.append(final_cycle)

    if results:
        avg = sum(results) / len(results)
        avg_ns = avg / core_freq_ghz
        print(f"\n  Average final cycle: {avg:.0f} "
              f"({avg_ns:.1f} ns @ {core_freq_ghz} GHz)")

    print("\n[Test Done]")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING,
                        format="%(name)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="LegOSim two-simlet integration test (NPU + DRAM simlets)"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--bandwidth_gbps", type=float, default=256.0,
                        help="Weight DRAM bandwidth modeled by the DRAM simlet")
    parser.add_argument("--core_freq_ghz", type=float, default=1.0,
                        help="NPU clock frequency (GHz) for cycle→ns conversion")
    parser.add_argument("--noc_cycles", type=int, default=0,
                        help="NoC latency added by GM to each transfer")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    args = parser.parse_args()

    sys.path.append(
        os.environ.get("PYTORCHSIM_ROOT_PATH",
                       "/workspace/legomerged/eclab_legosim/PyTorchSim")
    )
    device = torch.device("npu:0")

    run_two_simlet_test(
        device=device,
        batch=args.batch,
        seq_len=args.seq_len,
        dtype=args.dtype,
        bandwidth_gbps=args.bandwidth_gbps,
        core_freq_ghz=args.core_freq_ghz,
        noc_cycles=args.noc_cycles,
        n_warmup=args.warmup,
        n_iters=args.iters,
    )
