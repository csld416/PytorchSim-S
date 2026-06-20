#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
	echo "Usage: $0 TRACE_NAME"
	exit 1
fi

# bash cleanup_results.sh

TRACE_NAME=$1

export TOGSIM_DEBUG_LEVEL=info
export TOGSIM_SSD_TRACE_NAME=${TRACE_NAME}
export TOGSIM_SSD_TRACE_DIR=/workspace/PyTorchSim/ssd_traces
LOG_DIR=/workspace/PyTorchSim/togsim_results/${TRACE_NAME}
if [[ ! -d "${LOG_DIR}" ]]; then
	mkdir "${LOG_DIR}"
fi
export TORCHSIM_LOG_PATH=${LOG_DIR}
export TOGSIM_CONFIG=/workspace/PyTorchSim/configs/systolic_ws_128x128_c2_simple_noc_tpuv3.yml