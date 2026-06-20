TRACE_NAME="${TOGSIM_SSD_TRACE_NAME:?TOGSIM_SSD_TRACE_NAME is not set}"

shopt -s dotglob
rm -rf /workspace/PyTorchSim/togsim_results/${TRACE_NAME}/*
rm -rf /workspace/PyTorchSim/outputs/*
rm -rf /workspace/PyTorchSim/ssd_traces/${TRACE_NAME}/*.csv
rm -rf /workspace/PyTorchSim/ssd_traces/${TRACE_NAME}/*.txt
shopt -u dotglob