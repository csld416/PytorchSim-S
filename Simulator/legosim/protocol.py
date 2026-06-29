"""
Python bindings for the LegOSim Inter-Chip Communication (Interchiplet) protocol.

Each simlet communicates with the LegOSim Global Manager (GM) through two channels:
  - stdout  : simlet → GM  :  "[INTERCMD] <CMD> <args>" lines
  - stdin   : GM → simlet  :  "[INTERCMD] RESULT n <items>" or "[INTERCMD] SYNC <cycle>"

All non-protocol output (logging, debug) must go to stderr so it does not
corrupt the stdout protocol channel.

Pipe file naming convention (created by GM):
  ./buffer{src_x}_{src_y}_{dst_x}_{dst_y}

Simlets transfer data by opening that file and reading/writing raw bytes.
Named pipes (FIFOs) or regular files may be used depending on the GM config;
this module handles both by blocking on open() for FIFOs.
"""

import sys
import struct

_CMD_HEAD = "[INTERCMD]"


def _emit(cmd: str) -> None:
    sys.stdout.write(f"{_CMD_HEAD} {cmd}\n")
    sys.stdout.flush()


def _recv() -> list[str]:
    """Read one response line from GM (via stdin) and return its tokens."""
    line = sys.stdin.readline()
    if not line:
        raise EOFError("LegOSim GM closed stdin — simulation ended")
    line = line.strip()
    # Strip the optional [INTERCMD] prefix that the GM includes in its responses
    if line.startswith(_CMD_HEAD):
        line = line[len(_CMD_HEAD):].strip()
    return line.split()


# ---------------------------------------------------------------------------
# Handshake commands (SEND / RECEIVE)
# ---------------------------------------------------------------------------

def send_sync(src_x: int, src_y: int, dst_x: int, dst_y: int) -> str:
    """
    Announce that this simlet will send data to (dst_x, dst_y).

    Sends:  [INTERCMD] SEND src_x src_y dst_x dst_y
    Gets:   [INTERCMD] RESULT 1 <pipe_name>
    Returns the pipe file path to write data into.
    """
    _emit(f"SEND {src_x} {src_y} {dst_x} {dst_y}")
    parts = _recv()  # RESULT 1 <pipe_name>
    return parts[2]


def receive_sync(dst_x: int, dst_y: int, src_x: int, src_y: int) -> str:
    """
    Announce that this simlet will receive data from (src_x, src_y).

    Sends:  [INTERCMD] RECEIVE dst_x dst_y src_x src_y
    Gets:   [INTERCMD] RESULT 1 <pipe_name>
    Returns the pipe file path to read data from.
    """
    _emit(f"RECEIVE {dst_x} {dst_y} {src_x} {src_y}")
    parts = _recv()  # RESULT 1 <pipe_name>
    return parts[2]


# ---------------------------------------------------------------------------
# Timing synchronization commands (WRITE / READ / CYCLE)
# ---------------------------------------------------------------------------

def write_sync(cycle: int, src_x: int, src_y: int, dst_x: int, dst_y: int,
               nbytes: int, desc: int = 0) -> int:
    """
    Report that this simlet finished writing nbytes at the given local cycle.

    Sends:  [INTERCMD] WRITE cycle src_x src_y dst_x dst_y nbytes desc
    Gets:   [INTERCMD] SYNC <synced_cycle>
    Returns the GM-synchronized cycle at which the transfer completes.
    """
    _emit(f"WRITE {cycle} {src_x} {src_y} {dst_x} {dst_y} {nbytes} {desc}")
    parts = _recv()  # SYNC <cycle>
    return int(parts[1])


def read_sync(cycle: int, dst_x: int, dst_y: int, src_x: int, src_y: int,
              nbytes: int, desc: int = 0) -> int:
    """
    Report that this simlet started reading nbytes at the given local cycle.

    Sends:  [INTERCMD] READ cycle dst_x dst_y src_x src_y nbytes desc
    Gets:   [INTERCMD] SYNC <synced_cycle>
    Returns the GM-synchronized cycle at which the transfer completes.
    """
    _emit(f"READ {cycle} {dst_x} {dst_y} {src_x} {src_y} {nbytes} {desc}")
    parts = _recv()  # SYNC <cycle>
    return int(parts[1])


def cycle_sync(cycle: int) -> int:
    """
    Report the current local cycle to GM and get the global synchronized cycle.

    Sends:  [INTERCMD] CYCLE cycle
    Gets:   [INTERCMD] SYNC <synced_cycle>
    """
    _emit(f"CYCLE {cycle}")
    parts = _recv()  # SYNC <cycle>
    return int(parts[1])


# ---------------------------------------------------------------------------
# Pipe data transfer helpers
# ---------------------------------------------------------------------------

def write_pipe(pipe_name: str, data: bytes) -> None:
    """Write bytes into a LegOSim pipe file (blocking open for FIFOs)."""
    with open(pipe_name, "wb") as f:
        f.write(data)


def read_pipe(pipe_name: str, nbytes: int) -> bytes:
    """Read exactly nbytes from a LegOSim pipe file (blocking open for FIFOs)."""
    with open(pipe_name, "rb") as f:
        return f.read(nbytes)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def pack_int64(value: int) -> bytes:
    return struct.pack("<q", value)


def unpack_int64(data: bytes) -> int:
    return struct.unpack("<q", data)[0]
