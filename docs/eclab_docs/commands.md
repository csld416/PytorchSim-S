# 指令集
設置環境變數,輸出位置
```bash
#source setup_env.sh ${TOGSIM_SSD_TRACE_NAME} [0/1]
# No LegoSim
source setup_env.sh test1 0
# With LegoSim
source setup_env.sh test1 1
```

清除輸出
```bash
# 保留cache
bash clean_output.sh 0
# 清除cache (會重新跑gem5 建議清除)
bash clean_output.sh 1
``` 

執行模擬
```bash
# 官方範例
python3 test/Llama/test_llama.py

# sim開頭: 跑random input
# TinyLLaMA
#python3 test/Llama/test_tinyllama.py --npu --phase [decode/prefill] --num_layers [i] --seq_len [default: 500 (for prefill)] --context_len [default: 500 (for decode)]
python3 test/Llama/sim_tinyllama.py --npu --phase decode --num_layers 1

# LLaMA2-7B
python3 test/Llama/sim_llama2_7B.py --npu --phase decode --num_layers 1

# GPT_NeoX-20B
python3 test/GPT/sim_GPT_NeoX_20B.py --npu --phase decode --num_layers 1

# test開頭: 跑實際prompt
#python3 test/Llama/test_tinyllama.py --npu --prompt [ex. Machine learning is a useful tool that] --max_new_tokens [default: 1]
python3 test/Llama/test_tinyllama.py --npu 

# LLaMA2-7B
python3 test/Llama/test_llama2_7B.py --npu

# GPT_NeoX-20B
python3 test/GPT/test_GPT_NeoX_20B.py --npu
``` 

輸出總cycle數
```bash
#python3 sum_togsim_cycles.py --trace-name [ex. test1 (default: ${TOGSIM_SSD_TRACE_NAME})]
python3 sum_togsim_cycles.py
```

擷取model weights traces (如果想要手動執行)
```bash
python3 merge_weight_ranges.py
python3 extract_weight_traces.py
```

# 輸出位置
log檔案: togsim_output/${TOGSIM_SSD_TRACE_NAME}/

DMA traces: ssd_traces/${TOGSIM_SSD_TRACE_NAME}/