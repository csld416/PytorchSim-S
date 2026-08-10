#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

// Wire format shared between TOGSim's SsdLegoSimLink (DMA.cc caller side) and
// ssd_simlet.cpp (the LegoSim simlet that answers it). Both are always built
// by the same toolchain on the same host, so raw struct layout is fine to
// send verbatim through the interchiplet pipe -- no portable serialization
// needed.

// Fixed-size room for Instruction::get_addr_name(), e.g.
// "model.layers.5.self_attn.q_proj.weight". Longer names are truncated
// (see set_addr_name()) rather than growing the message to a variable size.
constexpr size_t kSsdAddrNameCapacity = 96;

// TOGSim -> SSD simlet: "how long would a DMA read of `nbytes` starting at
// `addr` (tensor `addr_name`) take?" `terminate` is a sentinel: when set,
// every other field is ignored and the SSD simlet acknowledges once then
// exits its request loop (see SsdLegoSimLink::shutdown()).
struct SsdLatencyRequest {
  uint64_t addr;
  uint64_t nbytes;
  uint64_t inst_id;
  char addr_name[kSsdAddrNameCapacity];
  uint8_t terminate;

  void set_addr_name(const std::string& name) {
    std::strncpy(addr_name, name.c_str(), kSsdAddrNameCapacity - 1);
    addr_name[kSsdAddrNameCapacity - 1] = '\0';
  }
};

// SSD simlet -> TOGSim: modeled latency for the request above, in
// nanoseconds. Unused (zero) when responding to a terminate sentinel.
struct SsdLatencyResponse {
  uint64_t latency_ns;
};
