#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Instruction.h"
#include "Memfetch.h"

class SsdTraceManager {
 public:
  static SsdTraceManager& instance();

  bool enabled() const { return _enabled; }
  bool is_weight_name(const std::string& name) const;
  void set_runtime_inputs(const std::vector<std::string>& inputs);

  void maybe_trace_and_mark(uint32_t core_id,
                            cycle_type core_cycle,
                            uint32_t core_freq_mhz,
                            Instruction& inst,
                            mem_fetch* access);
  bool pop_latency_for_instruction(uint32_t core_id,
                                   const Instruction& inst,
                                   uint64_t* latency_ns);
 private:
  SsdTraceManager();

  void load_weight_list();
  void open_trace();
  void load_latency_file();
  uint64_t assign_process_index();
  std::string make_latency_key(uint32_t core_id, uint64_t inst_id, const std::string& addr_name) const;
  bool match_weight(const std::string& name) const;

  std::string _trace_base_dir;
  std::string _trace_name;
  std::string _trace_path;
  std::string _weights_path;
  std::string _latency_path;
  uint64_t _process_index = 0;

  std::ofstream _trace_file;
  std::vector<std::string> _weight_entries;
  std::vector<std::string> _weight_substrings;
  std::unordered_set<std::string> _runtime_inputs;
  std::unordered_set<uint64_t> _inst_traced;

  uint64_t _trace_seq = 0;
  bool _latency_enabled = false;
  std::unordered_map<std::string, std::deque<uint64_t>> _latency_map;

  bool _enabled = false;
};
