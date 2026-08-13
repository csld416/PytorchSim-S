#pragma once

#include <cstdint>
#include <string>

#include "pipe_comm.h"
#include "ssd_protocol.h"

// Live counterpart to the real Dram/Interconnect timing model: instead of
// simulating DRAM channels and the on-chip NoC cycle by cycle, this asks an
// external LegoSim DRAM simlet for a latency, over the interchiplet sync
// protocol, and blocks until it answers. Where SsdLegoSimLink only covers
// weight-tensor DMA reads, DramLegoSimLink is the catch-all for every DMA
// access -- reads and writes alike -- so enabling it fully replaces
// TOGSim's Dram+Interconnect models for the run. Only makes sense when
// TOGSim itself is running as one of interchiplet's phase1 processes --
// outside of that, there's nothing on the other end of stdin.
//
// Enabled by setting TOGSIM_DRAM_LEGOSIM=1. Self coordinates are shared
// with SsdLegoSimLink (TOGSIM_LEGOSIM_X/Y, default (0,0) -- TOGSim has one
// chiplet identity regardless of how many peers it talks to); the peer
// (this DRAM simlet) defaults to (2,0), distinct from the SSD simlet's
// (1,0), overridable via TOGSIM_DRAM_PEER_LEGOSIM_X/Y.
class DramLegoSimLink {
 public:
  static DramLegoSimLink& instance();

  bool enabled() const { return _enabled; }

  // Blocks until the DRAM simlet answers. `addr`/`nbytes` describe the DMA
  // access; `inst_id`/`addr_name` are only for identification/logging on
  // the simlet side (addr_name is truncated to kSsdAddrNameCapacity-1
  // bytes).
  uint64_t query_latency_ns(uint64_t addr, uint64_t nbytes, uint64_t inst_id,
                            const std::string& addr_name);

  // Tells the DRAM simlet to exit its request loop and waits for its ack.
  // Call once, right before TOGSim's process would otherwise exit. Safe to
  // call multiple times or when disabled (no-op past the first call).
  void shutdown();

 private:
  DramLegoSimLink();

  SsdLatencyResponse round_trip(const SsdLatencyRequest& req);

  long _self_x = 0;
  long _self_y = 0;
  long _peer_x = 2;
  long _peer_y = 0;
  bool _enabled = false;
  bool _shutdown_sent = false;
  InterChiplet::PipeComm _pipe_comm;
};
