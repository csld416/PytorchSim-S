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
//
// Every query is issued at TOGSim's real, caller-supplied core_cycle (not
// an internally-tracked placeholder) -- interchiplet resolves each
// read/write pairing's end cycle in a common domain via each process's own
// clock_rate (see _build_legosim_yaml's per-process clock_rate and
// TOGSIM_DRAM_LEGOSIM_CORE_FREQ_MHZ below), then converts it back into
// TOGSim's own core-cycle domain before acking -- so the cycle
// query_latency_ns() gets back already includes both dram_simlet's modeled
// DRAM latency (folded into the cycle it wrote its response at, mirroring
// artifact/HBM_DDR/DDR.cpp/HBM.cpp's timeNow pattern) and any interconnect
// transport delay interchiplet resolved for the round trip (a real
// phase-2 NoC delay when TOGSIM_LEGOSIM_DRAM_NOC=1, or its generic
// flit-count default otherwise) -- rather than assuming zero transport
// cost, which is what returning resp.latency_ns directly would do.
class DramLegoSimLink {
 public:
  static DramLegoSimLink& instance();

  bool enabled() const { return _enabled; }

  // Blocks until the DRAM simlet answers, and returns the real elapsed
  // interchiplet latency (in ns) for the round trip issued at
  // `core_cycle` -- not just dram_simlet's own modeled DRAM latency, see
  // class comment. `addr`/`nbytes` describe the DMA access; `inst_id`/
  // `addr_name` are only for identification/logging on the simlet side
  // (addr_name is truncated to kSsdAddrNameCapacity-1 bytes).
  uint64_t query_latency_ns(uint64_t addr, uint64_t nbytes, uint64_t inst_id,
                            const std::string& addr_name, uint64_t core_cycle);

  // Tells the DRAM simlet to exit its request loop and waits for its ack.
  // Call once, right before TOGSim's process would otherwise exit. Safe to
  // call multiple times or when disabled (no-op past the first call).
  void shutdown();

 private:
  DramLegoSimLink();

  struct RoundTrip {
    SsdLatencyResponse response;
    // End cycle interchiplet resolved for this round trip, already
    // converted back into TOGSim's own core-cycle domain.
    InterChiplet::TimeType resolved_cycle;
  };
  RoundTrip round_trip(const SsdLatencyRequest& req, InterChiplet::TimeType cycle);

  long _self_x = 0;
  long _self_y = 0;
  long _peer_x = 2;
  long _peer_y = 0;
  bool _enabled = false;
  bool _shutdown_sent = false;
  InterChiplet::PipeComm _pipe_comm;
  // TOGSim's core clock, needed to convert the resolved core-cycle delta
  // back into ns for query_latency_ns()'s return contract. 0 (unset) falls
  // back to resp.latency_ns -- see .cc.
  double _core_freq_mhz = 0.0;
};
