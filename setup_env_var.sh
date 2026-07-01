#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
	echo "Usage: $0 TRACE_NAME"
	exit 1
fi

# bash cleanup_results.sh

TRACE_NAME=$1

export TORCHSIM_DIR=/workspace/legomerged/eclab_legosim/PyTorchSim
export PYTORCHSIM_ROOT_PATH=${TORCHSIM_DIR}
export TOGSIM_DEBUG_LEVEL=info
export TOGSIM_SSD_TRACE_NAME=${TRACE_NAME}

mkdir -p ${TORCHSIM_DIR}/ssd_traces
mkdir -p ${TORCHSIM_DIR}/ssd_traces/${TRACE_NAME}
mkdir -p ${TORCHSIM_DIR}/togsim_results
mkdir -p ${TORCHSIM_DIR}/togsim_results/${TRACE_NAME}
export TOGSIM_SSD_TRACE_DIR=${TORCHSIM_DIR}/ssd_traces

LOG_DIR=${TORCHSIM_DIR}/togsim_results/${TRACE_NAME}
export TORCHSIM_LOG_PATH=${LOG_DIR}
export TOGSIM_CONFIG=${TORCHSIM_DIR}/configs/systolic_ws_128x128_c2_simple_noc_tpuv3.yml

# export TORCHSIM_DEBUG_MODE=1
