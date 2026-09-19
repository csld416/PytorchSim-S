#pragma once

#include <cstdint>

#include "pipe_comm.h"
#include "ssd_ipc_protocol.h"

// Runtime client for the external SimpleSSD LegoSim process. Semantic request
// and completion metadata travel through PipeComm; matching READ/WRITE events
// tell LegoSim how many bytes cross the modeled interconnect and at what time.
//
// Enabled by setting TOGSIM_SSD_LEGOSIM=1. Chiplet coordinates default to
// the (0,0)=compute / (1,0)=SSD convention used by this integration,
// integration, overridable via TOGSIM_LEGOSIM_X/Y and
// TOGSIM_SSD_LEGOSIM_X/Y.
class SsdLegoSimLink {
 public:
  static SsdLegoSimLink& instance();

  bool enabled() const { return _enabled; }

  // Issues a logical SSD read at `issue_cycle` and returns the absolute
  // TOGSim cycle at which the data response has crossed the interconnect.
  uint64_t issue_read(uint64_t offset_bytes, uint64_t nbytes,
                      uint64_t issue_cycle);

  // Tells the SSD simlet to exit its request loop and waits for its ack.
  // Call once, right before TOGSim's process would otherwise exit. Safe to
  // call multiple times or when disabled (no-op past the first call).
  void shutdown();

 private:
  SsdLegoSimLink();

  struct RoundTrip {
    NUSSD::SsdIpcResponse response;
    InterChiplet::TimeType resolved_cycle;
  };

  RoundTrip round_trip(const NUSSD::SsdIpcRequest& request,
                       InterChiplet::TimeType issue_cycle,
                       uint64_t request_wire_bytes,
                       uint64_t response_wire_bytes);
  uint64_t allocate_request_id();

  long _self_x = 0;
  long _self_y = 0;
  long _peer_x = 1;
  long _peer_y = 0;
  bool _enabled = false;
  bool _shutdown_sent = false;
  uint64_t _next_request_id = 1;
  InterChiplet::TimeType _last_cycle = 1;
  InterChiplet::PipeComm _pipe_comm;
};
