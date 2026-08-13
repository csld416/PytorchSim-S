#include "DramLegoSimLink.h"

#include <cstdlib>

#include <spdlog/spdlog.h>

namespace {
long env_long(const char* name, long fallback) {
  const char* v = std::getenv(name);
  return v ? std::atol(v) : fallback;
}
}  // namespace

DramLegoSimLink& DramLegoSimLink::instance() {
  static DramLegoSimLink link;
  return link;
}

DramLegoSimLink::DramLegoSimLink() {
  const char* on = std::getenv("TOGSIM_DRAM_LEGOSIM");
  _enabled = on && std::string(on) == "1";
  if (!_enabled) return;

  _self_x = env_long("TOGSIM_LEGOSIM_X", 0);
  _self_y = env_long("TOGSIM_LEGOSIM_Y", 0);
  _peer_x = env_long("TOGSIM_DRAM_PEER_LEGOSIM_X", 2);
  _peer_y = env_long("TOGSIM_DRAM_PEER_LEGOSIM_Y", 0);

  spdlog::info("[DramLegoSimLink] enabled: self=({},{}) dram_peer=({},{})",
               _self_x, _self_y, _peer_x, _peer_y);
}

SsdLatencyResponse DramLegoSimLink::round_trip(const SsdLatencyRequest& req) {
  // Request leg: self -> peer.
  std::string req_file = InterChiplet::sendSync(_self_x, _self_y, _peer_x, _peer_y);
  _pipe_comm.write_data(req_file.c_str(), const_cast<SsdLatencyRequest*>(&req), sizeof(req));
  InterChiplet::writeSync(0, _self_x, _self_y, _peer_x, _peer_y, sizeof(req), 0);

  // Response leg: peer -> self.
  std::string resp_file = InterChiplet::receiveSync(_peer_x, _peer_y, _self_x, _self_y);
  SsdLatencyResponse resp{};
  _pipe_comm.read_data(resp_file.c_str(), &resp, sizeof(resp));
  InterChiplet::readSync(0, _peer_x, _peer_y, _self_x, _self_y, sizeof(resp), 0);
  return resp;
}

uint64_t DramLegoSimLink::query_latency_ns(uint64_t addr, uint64_t nbytes, uint64_t inst_id,
                                           const std::string& addr_name) {
  SsdLatencyRequest req{};
  req.addr = addr;
  req.nbytes = nbytes;
  req.inst_id = inst_id;
  req.set_addr_name(addr_name);
  req.terminate = 0;
  return round_trip(req).latency_ns;
}

void DramLegoSimLink::shutdown() {
  if (!_enabled || _shutdown_sent) return;
  _shutdown_sent = true;
  SsdLatencyRequest req{};
  req.terminate = 1;
  round_trip(req);
  spdlog::info("[DramLegoSimLink] sent terminate sentinel, DRAM simlet acked.");
}
