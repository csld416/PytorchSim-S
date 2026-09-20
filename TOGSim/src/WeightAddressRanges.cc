#include "WeightAddressRanges.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#include <spdlog/spdlog.h>

WeightAddressRanges& WeightAddressRanges::instance() {
  static WeightAddressRanges ranges;
  return ranges;
}

WeightAddressRanges::WeightAddressRanges() {
  const char* trace_name = std::getenv("TOGSIM_SSD_TRACE_NAME");
  if (!trace_name || std::string(trace_name).empty()) {
    return;
  }
  const char* trace_dir_env = std::getenv("TOGSIM_SSD_TRACE_DIR");
  std::string trace_dir = (trace_dir_env && std::string(trace_dir_env).size())
                              ? std::string(trace_dir_env)
                              : std::string("/workspace/PyTorchSim/ssd_traces");
  std::string path = trace_dir + "/" + trace_name + "/model_weight_placements.tsv";

  std::ifstream in(path);
  if (!in.is_open()) {
    spdlog::warn(
        "[WeightAddressRanges] Could not open {} -- live SSD path will treat no DMA as a "
        "weight access until the placement manifest exists.",
        path);
    return;
  }

  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream iss(line);
    std::string name;
    uint64_t base = 0, end_exclusive = 0, ssd_base = 0, size_bytes = 0;
    if (!(iss >> name >> base >> end_exclusive >> ssd_base >> size_bytes)) {
      continue;  // Also skips the header row.
    }
    if (end_exclusive <= base || size_bytes != end_exclusive - base ||
        ssd_base > std::numeric_limits<uint64_t>::max() - size_bytes) {
      spdlog::warn("[WeightAddressRanges] Ignoring invalid placement for {}", name);
      continue;
    }
    _ranges.push_back({name, base, end_exclusive, ssd_base});
  }
  std::sort(_ranges.begin(), _ranges.end(),
            [](const Range& a, const Range& b) { return a.base < b.base; });
  _enabled = !_ranges.empty();
  spdlog::info("[WeightAddressRanges] Loaded {} active weight placement(s) from {}",
               _ranges.size(), path);
}

bool WeightAddressRanges::translate(uint64_t host_addr,
                                    uint64_t length_bytes,
                                    uint64_t* ssd_offset) const {
  if (_ranges.empty() || ssd_offset == nullptr || length_bytes == 0 ||
      host_addr > std::numeric_limits<uint64_t>::max() - length_bytes) {
    return false;
  }
  const uint64_t host_end_exclusive = host_addr + length_bytes;
  // First range whose base is greater than host_addr; with non-overlapping
  // materialized parameters, the only candidate is immediately before it.
  auto it = std::upper_bound(_ranges.begin(), _ranges.end(), host_addr,
                             [](uint64_t value, const Range& r) { return value < r.base; });
  if (it == _ranges.begin()) return false;
  --it;
  if (host_addr < it->base || host_end_exclusive > it->end_exclusive) {
    return false;
  }
  const uint64_t displacement = host_addr - it->base;
  if (it->ssd_base > std::numeric_limits<uint64_t>::max() - displacement) {
    return false;
  }
  *ssd_offset = it->ssd_base + displacement;
  spdlog::info(
      "[NUSSD translation] tensor={} host_addr={} length_bytes={} host_base={} "
      "host_end={} ssd_base={} ssd_offset={}",
      it->name, host_addr, length_bytes, it->base, it->end_exclusive,
      it->ssd_base, *ssd_offset);
  return true;
}
