#include "WeightAddressRanges.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
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
  std::string path = trace_dir + "/" + trace_name + "/model_weight_ranges_merged.txt";

  std::ifstream in(path);
  if (!in.is_open()) {
    spdlog::warn(
        "[WeightAddressRanges] Could not open {} -- live SSD path will treat no DMA as a "
        "weight access until this file exists.",
        path);
    return;
  }

  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream iss(line);
    uint64_t base = 0, end = 0;
    if (!(iss >> base >> end)) continue;
    if (end < base) std::swap(base, end);
    _ranges.push_back({base, end});
  }
  std::sort(_ranges.begin(), _ranges.end(),
            [](const Range& a, const Range& b) { return a.base < b.base; });
  _enabled = true;
  spdlog::info("[WeightAddressRanges] Loaded {} weight address range(s) from {}", _ranges.size(),
               path);
}

bool WeightAddressRanges::is_weight_address(uint64_t addr) const {
  if (_ranges.empty()) return false;
  // First range whose base is > addr; the only candidate containing addr is the one
  // right before it (ranges are sorted and non-overlapping).
  auto it = std::upper_bound(_ranges.begin(), _ranges.end(), addr,
                             [](uint64_t value, const Range& r) { return value < r.base; });
  if (it == _ranges.begin()) return false;
  --it;
  return addr <= it->end;
}
