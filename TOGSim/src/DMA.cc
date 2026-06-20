#include "DMA.h"
#include "SsdTrace.h"
#include "TileGraph.h"
#include "TraceLogTags.h"

#include <cmath>

DMA::DMA(uint32_t id, uint32_t dram_req_size, bool l2_datacache_enabled, uint32_t core_freq_mhz) {
  _id = id;
  _dram_req_size = dram_req_size;
  _l2_datacache_enabled = l2_datacache_enabled;
  _core_freq_mhz = core_freq_mhz;
  _current_inst = nullptr;
  _finished = true;
}

void DMA::issue_tile(std::shared_ptr<Instruction> inst) {
  _current_inst = std::move(inst);
  _ssd_pending = false;
  _ssd_finish_cycle = 0;
  std::vector<size_t>& tile_size = _current_inst->get_tile_size();
  if (tile_size.size() <= 0 || tile_size.size() > get_max_dim()) {
    spdlog::error("[DMA {}] issued tile is not supported format.. tile.size: {}, tile_size: [{}]", _id, tile_size.size(), fmt::join(tile_size, ", "));
    exit(EXIT_FAILURE);
  }
  _finished = false;
}

void DMA::update_ssd(cycle_type core_cycle) {
  if (_ssd_pending && core_cycle >= _ssd_finish_cycle) {
    _ssd_pending = false;
    _finished = true;
    _generated_once = false;
    if (_current_inst != nullptr) {
      _ssd_finished_inst = std::move(_current_inst);
      _current_inst = nullptr;
    }
  }
}

std::shared_ptr<Instruction> DMA::take_ssd_finished() {
  if (_ssd_finished_inst == nullptr)
    return nullptr;
  return std::move(_ssd_finished_inst);
}

std::shared_ptr<std::vector<mem_fetch*>> DMA::get_memory_access(cycle_type core_cycle, int nr_req) {
  auto access_vec = std::make_shared<std::vector<mem_fetch *>>();

  if (_ssd_pending)
    return access_vec;

  if (!_generated_once) {
    if (_current_inst->is_dma_read()) {
      uint64_t latency_ns = 0;
      if (SsdTraceManager::instance().pop_latency_for_instruction(_id, *_current_inst, &latency_ns)) {
        const double period_ns = _core_freq_mhz > 0 ? 1000.0 / static_cast<double>(_core_freq_mhz) : 0.0;
        uint64_t latency_cycles = 0;
        if (period_ns > 0.0) {
          latency_cycles = static_cast<uint64_t>(std::ceil(static_cast<double>(latency_ns) / period_ns));
        }
        if (latency_cycles == 0)
          latency_cycles = 1;
        _ssd_pending = true;
        _ssd_finish_cycle = core_cycle + latency_cycles;
        _finished = false;
        return access_vec;
      }
    }
    std::shared_ptr<std::set<addr_type>> addr_set =
      _current_inst->get_dram_address(_dram_req_size);

    Tile* owner = (Tile*)_current_inst->get_owner();
    std::shared_ptr<TileSubGraph> owner_subgraph = owner->get_owner();
    unsigned long long base_daddr = _current_inst->get_base_dram_address();

    bool is_cacheable =
      owner_subgraph->is_cacheable(base_daddr, base_daddr + _dram_req_size);

    if (_l2_datacache_enabled) {
      spdlog::trace(
          "[{}][Core {}][{}][INST_ID={}] dram=0x{:016x} cacheable={}",
          core_cycle,
          _id,
          TraceLogTag::pad15(TraceLogTag::kL2CacheableStatusForAddress),
          _current_inst->get_global_inst_id(),
          base_daddr,
          is_cacheable);
    }
    spdlog::trace(
        "[{}][Core {}][{}][INST_ID={}] core_id={} subgraph_id={} numa_id={} addr_name={} is_write={}",
        core_cycle,
        _id,
        TraceLogTag::pad15(TraceLogTag::kDmaNumaPlacement),
        _current_inst->get_global_inst_id(),
        owner_subgraph->get_core_id(),
        _current_inst->subgraph_id,
        _current_inst->get_numa_id(),
        _current_inst->get_addr_name(),
        _current_inst->is_dma_write());
    for (const auto& addr : *addr_set) {
      mem_access_type acc_type =
        _current_inst->is_dma_write() ? mem_access_type::GLOBAL_ACC_W
                                          : mem_access_type::GLOBAL_ACC_R;
      mf_type type =
        _current_inst->is_dma_write() ? mf_type::WRITE_REQUEST
                                          : mf_type::READ_REQUEST;

      mem_fetch* access = new mem_fetch(
          addr, acc_type, type, _dram_req_size,
          _current_inst->get_numa_id(),
          static_cast<void*>(_current_inst.get()));

      access->set_cacheable(is_cacheable);
      SsdTraceManager::instance().maybe_trace_and_mark(
          _id, core_cycle, _core_freq_mhz, *_current_inst, access);
      _current_inst->inc_waiting_request();
      _pending_accesses.push(access);
    }
    _generated_once = true;
  }

  if (nr_req == -1)
    nr_req = _pending_accesses.size();

  // Return pending accesses up to nr_req
  for (int i = 0; i < nr_req; i++) {
      if (_pending_accesses.empty())
        break;
      access_vec->push_back(_pending_accesses.front());
      _pending_accesses.pop();
  }

  if (_pending_accesses.empty()) {
    _finished = true;
    _generated_once = false;
  }

  return access_vec;
}

uint32_t DMA::generate_mem_access_id() {
  static uint32_t id_counter{0};
  return id_counter++;
}