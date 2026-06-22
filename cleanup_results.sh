TRACE_NAME="${TOGSIM_SSD_TRACE_NAME:?TOGSIM_SSD_TRACE_NAME is not set}"
TORCHSIM_DIR="${TORCHSIM_DIR:?TORCHSIM_DIR is not set}"

shopt -s dotglob
rm -rf ${TORCHSIM_DIR}/togsim_results/${TRACE_NAME}/*
rm -rf ${TORCHSIM_DIR}/outputs/*
rm -rf ${TORCHSIM_DIR}/ssd_traces/${TRACE_NAME}/*.csv
rm -rf ${TORCHSIM_DIR}/ssd_traces/${TRACE_NAME}/*.txt
shopt -u dotglob