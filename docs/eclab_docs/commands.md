# 指令集
設置環境變數,輸出位置
```bash
#source setup_env.sh ${TOGSIM_SSD_TRACE_NAME}
source setup_env.sh test1
```

清除輸出
```bash
bash clean_output.sh
``` 

執行模擬
```bash
# single layer
python3 test/Llama/test_llama.py

# model (foward pass)
python3 test/Llama/test_tinyllama.py

# model (generate)
python3 test/Llama/test_tinyllama.py --generate
``` 

擷取model weights traces
```bash
python3 merge_weight_ranges.py
python3 extract_weight_traces.py
```

# 輸出位置
log檔案: togsim_output/${TOGSIM_SSD_TRACE_NAME}/

traces: ssd_traces/${TOGSIM_SSD_TRACE_NAME}/model_weight_traces/