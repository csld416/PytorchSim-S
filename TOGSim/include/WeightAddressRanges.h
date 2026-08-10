#pragma once

#include <cstdint>
#include <vector>

// Loads model_weight_ranges_merged.txt (written by the Python side's
// _dump_module_weight_ranges() + merge_weight_ranges.py -- see
// tests/Llama/test_tinyllama.py / test_llama2_7B.py) and answers "is this DMA
// address inside a currently-loaded weight tensor?" This is what lets the
// live LegoSim SSD path in DMA.cc route only weight DMAs to the SSD simlet,
// leaving activations/KV-cache on the normal DRAM path.
//
// Addresses are comparable directly: PyTorchSimDevice's tensors are backed by
// real host memory (OpenRegDeviceAllocator -> orMalloc -> a page-aligned host
// allocation), so a tensor's data_ptr() on the Python side is the exact same
// address TOGSim's Instruction::get_base_dram_address() reports.
//
// Loaded lazily, once, from TOGSIM_SSD_TRACE_DIR/TOGSIM_SSD_TRACE_NAME/
// model_weight_ranges_merged.txt (same env vars SsdTraceManager uses). A
// fresh TOGSim process is spawned per kernel launch (see
// Simulator/simulator.py's run_standalone()), so "load once at startup" is
// enough -- there's no need to reload mid-process. If the file can't be
// found, every address is treated as "not a weight" (fail closed: the live
// SSD path should only fire when a DMA is known to be a weight access).
class WeightAddressRanges {
 public:
  static WeightAddressRanges& instance();

  bool enabled() const { return _enabled; }
  bool is_weight_address(uint64_t addr) const;

 private:
  WeightAddressRanges();

  struct Range {
    uint64_t base;
    uint64_t end;
  };
  std::vector<Range> _ranges;  // sorted by base, non-overlapping (merge_weight_ranges.py's job)
  bool _enabled = false;
};
