#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
	echo "Usage: $0 TRACE_NAME [TOGSIM_LEGOSIM_SSD] [TOGSIM_LEGOSIM_DRAM] [TOGSIM_LEGOSIM_DRAM_NOC]"
	exit 1
fi

# bash cleanup_results.sh

TRACE_NAME=$1
TOGSIM_LEGOSIM_SSD=${2:-0}
TOGSIM_LEGOSIM_DRAM=${3:-0}
# Only meaningful when TOGSIM_LEGOSIM_DRAM=1 -- plugs a real popnet into
# phase2 (instead of the default no-op filler) so interchiplet's two-phase
# fixed-point loop actually runs for real testing. See
# Simulator/simulator.py's _build_legosim_yaml and
# TOGSim/legosim/dram_simlet.cpp's timeNow tracking.
TOGSIM_LEGOSIM_DRAM_NOC=${4:-0}

if [[ "${TOGSIM_LEGOSIM_SSD}" != "0" && "${TOGSIM_LEGOSIM_SSD}" != "1" ]]; then
	echo "Error: TOGSIM_LEGOSIM_SSD must be 0 or 1 (got '${TOGSIM_LEGOSIM_SSD}')"
	exit 1
fi

if [[ "${TOGSIM_LEGOSIM_DRAM}" != "0" && "${TOGSIM_LEGOSIM_DRAM}" != "1" ]]; then
	echo "Error: TOGSIM_LEGOSIM_DRAM must be 0 or 1 (got '${TOGSIM_LEGOSIM_DRAM}')"
	exit 1
fi

if [[ "${TOGSIM_LEGOSIM_DRAM_NOC}" != "0" && "${TOGSIM_LEGOSIM_DRAM_NOC}" != "1" ]]; then
	echo "Error: TOGSIM_LEGOSIM_DRAM_NOC must be 0 or 1 (got '${TOGSIM_LEGOSIM_DRAM_NOC}')"
	exit 1
fi

export TORCHSIM_DIR=/workspace/legomerged/eclab_legosim/PyTorchSim
export PYTORCHSIM_ROOT_PATH=${TORCHSIM_DIR}
export TOGSIM_DEBUG_LEVEL=info
export TOGSIM_SSD_TRACE_NAME=${TRACE_NAME}
export TOGSIM_LEGOSIM_SSD=${TOGSIM_LEGOSIM_SSD}
export TOGSIM_LEGOSIM_DRAM=${TOGSIM_LEGOSIM_DRAM}
export TOGSIM_LEGOSIM_DRAM_NOC=${TOGSIM_LEGOSIM_DRAM_NOC}

mkdir -p ${TORCHSIM_DIR}/ssd_traces
mkdir -p ${TORCHSIM_DIR}/ssd_traces/${TRACE_NAME}
mkdir -p ${TORCHSIM_DIR}/togsim_results
mkdir -p ${TORCHSIM_DIR}/togsim_results/${TRACE_NAME}
export TOGSIM_SSD_TRACE_DIR=${TORCHSIM_DIR}/ssd_traces

LOG_DIR=${TORCHSIM_DIR}/togsim_results/${TRACE_NAME}
export TORCHSIM_LOG_PATH=${LOG_DIR}
# export TOGSIM_CONFIG=${TORCHSIM_DIR}/configs/systolic_ws_128x128_c1_booksim_tpuv3.yml
export TOGSIM_CONFIG=${TORCHSIM_DIR}/configs/eclab.yml

export TORCHSIM_DEBUG_MODE=0
