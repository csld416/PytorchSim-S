#include "SsdLegoSimLink.h"

#include <cstdlib>
#include <limits>

#include <spdlog/spdlog.h>

#include "ssd_ipc_trace.h"

namespace {
long env_long(const char* name, long fallback) {
  const char* v = std::getenv(name);
  return v ? std::atol(v) : fallback;
}

NUSSD::SsdIpcTrace& protocol_trace() {
  static NUSSD::SsdIpcTrace trace("NPU", "npu.events.tsv");
  return trace;
}
}  // namespace

SsdLegoSimLink& SsdLegoSimLink::instance() {
  static SsdLegoSimLink link;
  return link;
}

SsdLegoSimLink::SsdLegoSimLink() {
  const char* on = std::getenv("TOGSIM_SSD_LEGOSIM");
  _enabled = on && std::string(on) == "1";
  if (!_enabled) return;

  _self_x = env_long("TOGSIM_LEGOSIM_X", 0);
  _self_y = env_long("TOGSIM_LEGOSIM_Y", 0);
  _peer_x = env_long("TOGSIM_SSD_LEGOSIM_X", 1);
  _peer_y = env_long("TOGSIM_SSD_LEGOSIM_Y", 0);

  spdlog::info("[SsdLegoSimLink] enabled: self=({},{}) ssd_peer=({},{})",
               _self_x, _self_y, _peer_x, _peer_y);
}

uint64_t SsdLegoSimLink::allocate_request_id() {
  if (_next_request_id == 0 ||
      _next_request_id > static_cast<uint64_t>(std::numeric_limits<long>::max())) {
    spdlog::critical("[SsdLegoSimLink] exhausted LegoSim request descriptors");
    std::exit(EXIT_FAILURE);
  }
  return _next_request_id++;
}

SsdLegoSimLink::RoundTrip SsdLegoSimLink::round_trip(
    const NUSSD::SsdIpcRequest& request,
    InterChiplet::TimeType issue_cycle,
    uint64_t request_wire_bytes,
    uint64_t response_wire_bytes) {
  if (request_wire_bytes > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
      response_wire_bytes > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    spdlog::critical("[SsdLegoSimLink] transfer exceeds LegoSim byte-count range");
    std::exit(EXIT_FAILURE);
  }

  // Keep LegoSim's synchronization descriptor at zero. The preceding SEND
  // command has no descriptor, and interchiplet compares that command with
  // PopNet's first delay record at the start of the next round. Encoding the
  // request ID here makes those descriptors differ and causes interchiplet to
  // discard every network delay. The fixed-size FIFO message remains the
  // authoritative request-ID carrier and is validated end to end below.
  constexpr long desc = 0;

  // FIFO carries the semantic request; WRITE models its interconnect leg.
  std::string req_file = InterChiplet::sendSync(_self_x, _self_y, _peer_x, _peer_y);
  if (_pipe_comm.write_data(req_file.c_str(),
                            const_cast<NUSSD::SsdIpcRequest*>(&request),
                            sizeof(request)) !=
      static_cast<int>(sizeof(request))) {
    spdlog::critical("[SsdLegoSimLink] failed to send complete SSD request");
    std::exit(EXIT_FAILURE);
  }
  protocol_trace().emit("NPU_REQUEST_FIFO_SENT", request, issue_cycle);
  const InterChiplet::TimeType command_arrival_cycle = InterChiplet::writeSync(
      issue_cycle, _self_x, _self_y, _peer_x, _peer_y,
      static_cast<int>(request_wire_bytes), desc);
  protocol_trace().emit("NPU_COMMAND_ARRIVAL_RESOLVED", request,
                        command_arrival_cycle);

  // FIFO carries completion metadata; READ models the returned data/ack leg.
  std::string resp_file = InterChiplet::receiveSync(_peer_x, _peer_y, _self_x, _self_y);
  NUSSD::SsdIpcResponse response;
  if (_pipe_comm.read_data(resp_file.c_str(), &response, sizeof(response)) !=
      static_cast<int>(sizeof(response))) {
    spdlog::critical("[SsdLegoSimLink] failed to receive complete SSD response");
    std::exit(EXIT_FAILURE);
  }
  protocol_trace().emit("NPU_RESPONSE_FIFO_RECEIVED", request, issue_cycle,
                        response.completed_tick_ps, response.status);
  const uint64_t actual_response_wire_bytes =
      response.status == NUSSD::SsdIpcStatus::Success
          ? response_wire_bytes
          : NUSSD::kSsdReadCommandBytes;
  InterChiplet::TimeType resolved_cycle = InterChiplet::readSync(
      issue_cycle, _peer_x, _peer_y, _self_x, _self_y,
      static_cast<int>(actual_response_wire_bytes), desc);
  protocol_trace().emit("NPU_RESPONSE_ARRIVED", request, resolved_cycle,
                        response.completed_tick_ps, response.status);

  if (!NUSSD::hasValidHeader(response) ||
      response.request_id != request.request_id ||
      response.status != NUSSD::SsdIpcStatus::Success) {
    spdlog::critical(
        "[SsdLegoSimLink] invalid SSD response: request_id={} response_id={} status={}",
        request.request_id, response.request_id,
        static_cast<uint16_t>(response.status));
    std::exit(EXIT_FAILURE);
  }

  _last_cycle = resolved_cycle;
  protocol_trace().emit("NPU_REQUEST_COMPLETE", request, resolved_cycle,
                        response.completed_tick_ps, response.status);
  return {response, resolved_cycle};
}

uint64_t SsdLegoSimLink::issue_read(uint64_t offset_bytes, uint64_t nbytes,
                                    uint64_t issue_cycle) {
  if (nbytes == 0 ||
      nbytes > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    spdlog::critical("[SsdLegoSimLink] invalid SSD read length: {}", nbytes);
    std::exit(EXIT_FAILURE);
  }
  NUSSD::SsdIpcRequest request;
  request.operation = NUSSD::SsdIpcOperation::Read;
  request.request_id = allocate_request_id();
  request.offset_bytes = offset_bytes;
  request.length_bytes = nbytes;
  protocol_trace().emit("NPU_ISSUE", request, issue_cycle);

  RoundTrip result = round_trip(request, issue_cycle,
                                NUSSD::kSsdReadCommandBytes, nbytes);
  return static_cast<uint64_t>(result.resolved_cycle);
}

void SsdLegoSimLink::shutdown() {
  if (!_enabled || _shutdown_sent) return;
  _shutdown_sent = true;
  NUSSD::SsdIpcRequest request;
  request.operation = NUSSD::SsdIpcOperation::Shutdown;
  request.request_id = allocate_request_id();
  protocol_trace().emit("NPU_SHUTDOWN", request, _last_cycle);
  round_trip(request, _last_cycle, NUSSD::kSsdReadCommandBytes,
             NUSSD::kSsdReadCommandBytes);
  spdlog::info("[SsdLegoSimLink] sent shutdown request, SimpleSSD acked.");
}
