#pragma once

#include <cstdint>
#include <string>

#include "pipe_comm.h"
#include "ssd_protocol.h"

// Live counterpart to SsdTraceManager's replay path (see SsdTrace.h):
// instead of popping a pre-recorded latency out of a trace file, this asks
// an external LegoSim SSD/DRAM simlet for a latency, over the interchiplet
// sync protocol, and blocks until it answers. Only makes sense when TOGSim
// itself is running as one of interchiplet's phase1 processes (see
// artifact/auto_transformer/CMakeLists.txt's `run` target for the general
// pattern) -- outside of that, there's nothing on the other end of stdin.
//
// Enabled by setting TOGSIM_SSD_LEGOSIM=1. Chiplet coordinates default to
// the (0,0)=compute / (1,0)=DRAM convention used elsewhere in this
// integration, overridable via TOGSIM_LEGOSIM_X/Y and
// TOGSIM_SSD_LEGOSIM_X/Y.
class SsdLegoSimLink {
 public:
  static SsdLegoSimLink& instance();

  bool enabled() const { return _enabled; }

  // Blocks until the SSD simlet answers. `addr`/`nbytes` describe the DMA
  // access; `inst_id`/`addr_name` are only for identification/logging on the
  // simlet side (addr_name is truncated to kSsdAddrNameCapacity-1 bytes).
  uint64_t query_latency_ns(uint64_t addr, uint64_t nbytes, uint64_t inst_id,
                            const std::string& addr_name);

  // Tells the SSD simlet to exit its request loop and waits for its ack.
  // Call once, right before TOGSim's process would otherwise exit. Safe to
  // call multiple times or when disabled (no-op past the first call).
  void shutdown();

 private:
  SsdLegoSimLink();

  SsdLatencyResponse round_trip(const SsdLatencyRequest& req);

  long _self_x = 0;
  long _self_y = 0;
  long _peer_x = 1;
  long _peer_y = 0;
  bool _enabled = false;
  bool _shutdown_sent = false;
  InterChiplet::PipeComm _pipe_comm;
};
