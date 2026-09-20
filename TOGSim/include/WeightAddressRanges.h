#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Loads model_weight_placements.tsv, written by
// PyTorchSimFrontend.weight_placement, and translates a complete DMA range
// inside a currently materialized weight tensor to its stable logical SSD
// offset. Activations, KV cache, and accesses crossing a weight boundary fail
// closed and remain on the normal DRAM path.
//
// Addresses are comparable directly: PyTorchSimDevice's tensors are backed by
// real host memory (OpenRegDeviceAllocator -> orMalloc -> a page-aligned host
// allocation), so a tensor's data_ptr() on the Python side is the exact same
// address TOGSim's Instruction::get_base_dram_address() reports.
//
// Host intervals use [base, end_exclusive). Loaded lazily, once, from
// TOGSIM_SSD_TRACE_DIR/TOGSIM_SSD_TRACE_NAME/model_weight_placements.tsv. A
// fresh TOGSim process is spawned per kernel launch (see
// Simulator/simulator.py's run_standalone()), so "load once at startup" is
// enough -- there's no need to reload mid-process. If the file can't be
// found, every address is treated as "not a weight" (fail closed: the live
// SSD path should only fire when a DMA is known to be a weight access).
class WeightAddressRanges {
 public:
  static WeightAddressRanges& instance();

  bool enabled() const { return _enabled; }
  bool translate(uint64_t host_addr, uint64_t length_bytes,
                 uint64_t* ssd_offset) const;

 private:
  WeightAddressRanges();

  struct Range {
    std::string name;
    uint64_t base;
    uint64_t end_exclusive;
    uint64_t ssd_base;
  };
  std::vector<Range> _ranges;  // sorted by host base
  bool _enabled = false;
};
