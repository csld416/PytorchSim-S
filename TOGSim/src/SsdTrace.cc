#include "SsdTrace.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <sstream>

#include <spdlog/spdlog.h>

namespace {

std::string trim_copy(const std::string& s) {
  const auto start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos)
    return {};
  const auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token = trim_copy(token);
    if (!token.empty())
      out.push_back(token);
  }
  return out;
}

uint64_t ns_to_cycles(uint64_t latency_ns, double period_ns) {
  if (period_ns <= 0.0)
    return 0;
  const double cycles = static_cast<double>(latency_ns) / period_ns;
  return static_cast<uint64_t>(std::ceil(cycles));
}

}  // namespace

SsdTraceManager& SsdTraceManager::instance() {
  static SsdTraceManager instance;
  return instance;
}

SsdTraceManager::SsdTraceManager() {
  const char* trace_name = std::getenv("TOGSIM_SSD_TRACE_NAME");
  if (!trace_name || std::string(trace_name).empty()) {
    return;
  }
  _trace_name = trace_name;

  const char* base_dir = std::getenv("TOGSIM_SSD_TRACE_DIR");
  _trace_base_dir = base_dir && std::string(base_dir).size()
                        ? std::string(base_dir)
                        : std::string("/workspace/PyTorchSim/ssd_traces");

  _trace_path = _trace_base_dir + "/" + _trace_name + "/ssd_trace.csv";
  _weights_path = _trace_base_dir + "/" + _trace_name + "/weights.txt";
  _latency_path = _trace_base_dir + "/" + _trace_name + "/ssd_latency_ns.txt";

  _enabled = true;
  open_trace();
  load_weight_list();
  load_latencies();

  spdlog::info("[SSD] trace enabled: {}", _trace_path);
  if (_latency_enabled)
    spdlog::info("[SSD] latency replay enabled: {}", _latency_path);
  else
    spdlog::info("[SSD] latency replay disabled (trace-only)");
}

void SsdTraceManager::open_trace() {
  namespace fs = std::filesystem;
  fs::path out_path(_trace_path);
  fs::create_directories(out_path.parent_path());
  _trace_file.open(_trace_path, std::ios::out | std::ios::trunc);
  if (!_trace_file.is_open()) {
    spdlog::error("[SSD] Failed to open trace file: {}", _trace_path);
    _enabled = false;
    return;
  }
  _trace_file << "seq,op,addr,len,timestamp_ns,core_id,inst_id,addr_name\n";
  _trace_file.flush();
}

void SsdTraceManager::load_weight_list() {
  namespace fs = std::filesystem;
  if (fs::exists(_weights_path)) {
    std::ifstream in(_weights_path);
    std::string line;
    while (std::getline(in, line)) {
      line = trim_copy(line);
      if (line.empty() || line[0] == '#')
        continue;
      _weight_entries.push_back(line);
    }
  }

  const char* substr_env = std::getenv("TOGSIM_SSD_WEIGHT_SUBSTR");
  if (substr_env && std::string(substr_env).size()) {
    _weight_substrings = split_csv(substr_env);
  }

  if (_weight_entries.empty() && _weight_substrings.empty()) {
    spdlog::warn("[SSD] No weight filters found (weights.txt or TOGSIM_SSD_WEIGHT_SUBSTR). SSD tracing will be empty.");
  }
}

void SsdTraceManager::load_latencies() {
  namespace fs = std::filesystem;
  if (!fs::exists(_latency_path)) {
    return;
  }

  std::ifstream in(_latency_path);
  if (!in.is_open()) {
    spdlog::warn("[SSD] Failed to open latency file: {}", _latency_path);
    return;
  }

  std::string line;
  while (std::getline(in, line)) {
    line = trim_copy(line);
    if (line.empty() || line[0] == '#')
      continue;
    try {
      _latency_ns.push_back(static_cast<uint64_t>(std::stoull(line)));
    } catch (const std::exception&) {
      spdlog::warn("[SSD] Invalid latency entry: {}", line);
    }
  }
  if (!_latency_ns.empty())
    _latency_enabled = true;
}

bool SsdTraceManager::match_weight(const std::string& name) const {
  if (name.empty())
    return false;
  if (_weight_entries.empty() && _weight_substrings.empty()) {
    if (_runtime_inputs.empty())
      return false;
    return _runtime_inputs.find(name) == _runtime_inputs.end();
  }
  for (const auto& entry : _weight_entries) {
    if (entry == "*")
      return true;
    if (!entry.empty() && entry.back() == '*') {
      const std::string prefix = entry.substr(0, entry.size() - 1);
      if (name.rfind(prefix, 0) == 0)
        return true;
    } else if (entry == name) {
      return true;
    }
  }
  for (const auto& sub : _weight_substrings) {
    if (name.find(sub) != std::string::npos)
      return true;
  }
  return false;
}

bool SsdTraceManager::is_weight_name(const std::string& name) const {
  return match_weight(name);
}

void SsdTraceManager::set_runtime_inputs(const std::vector<std::string>& inputs) {
  _runtime_inputs.clear();
  for (const auto& name : inputs) {
    if (!name.empty())
      _runtime_inputs.insert(name);
  }
}

void SsdTraceManager::maybe_trace_and_mark(uint32_t core_id,
                                           cycle_type core_cycle,
                                           uint32_t core_freq_mhz,
                                           Instruction& inst,
                                           mem_fetch* access) {
  if (!_enabled)
    return;
  if (!inst.is_dma_read())
    return;
  if (!is_weight_name(inst.get_addr_name()))
    return;

  const double period_ns = core_freq_mhz > 0 ? 1000.0 / static_cast<double>(core_freq_mhz) : 0.0;
  const uint64_t timestamp_ns = static_cast<uint64_t>(std::llround(static_cast<double>(core_cycle) * period_ns));

  const uint64_t seq = _trace_seq++;
  _trace_file << seq
              << ",R," << access->get_addr()
              << "," << access->get_data_size()
              << "," << timestamp_ns
              << "," << core_id
              << "," << inst.get_global_inst_id()
              << "," << inst.get_addr_name()
              << "\n";
  _trace_file.flush();

  if (_latency_enabled) {
    if (_latency_idx >= _latency_ns.size()) {
      spdlog::error("[SSD] Ran out of latency entries at seq {} (needed more)", seq);
      std::exit(EXIT_FAILURE);
    }
    const uint64_t latency_ns = _latency_ns[_latency_idx++];
    const uint64_t latency_cycles = ns_to_cycles(latency_ns, period_ns);
    access->set_ssd_seq_id(seq);
    access->set_ssd_latency_cycles(latency_cycles);
  }
}
