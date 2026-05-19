#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "Instruction.h"
#include "Memfetch.h"

class SsdTraceManager {
 public:
  static SsdTraceManager& instance();

  bool enabled() const { return _enabled; }
  bool latency_enabled() const { return _latency_enabled; }
  bool is_weight_name(const std::string& name) const;
  void set_runtime_inputs(const std::vector<std::string>& inputs);

  void maybe_trace_and_mark(uint32_t core_id,
                            cycle_type core_cycle,
                            uint32_t core_freq_mhz,
                            Instruction& inst,
                            mem_fetch* access);

 private:
  SsdTraceManager();

  void load_weight_list();
  void load_latencies();
  void open_trace();
  bool match_weight(const std::string& name) const;

  std::string _trace_base_dir;
  std::string _trace_name;
  std::string _trace_path;
  std::string _weights_path;
  std::string _latency_path;

  std::ofstream _trace_file;
  std::vector<std::string> _weight_entries;
  std::vector<std::string> _weight_substrings;
  std::vector<uint64_t> _latency_ns;
  std::unordered_set<std::string> _runtime_inputs;

  uint64_t _trace_seq = 0;
  size_t _latency_idx = 0;

  bool _enabled = false;
  bool _latency_enabled = false;
};
