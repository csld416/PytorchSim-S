# 容器環境配置  
連線到 `wombat` 主機：
```bash
ssh eclab3090_2
```

建立 `/mnt/nvme0/function92/legomerged` 目錄：
```bash
mkdir -p /mnt/nvme0/function92/legomerged
```

啟動 Container 並掛載目錄：
```bash
docker run -d -it --ipc=host --gpus '"device=1"' --mount type=bind,source=/mnt/nvme0/function92/legomerged,target=/workspace/legomerged -w /workspace/legomerged -v $SSH_AUTH_SOCK:/ssh-agent -e SSH_AUTH_SOCK=/ssh-agent --name function92-PyTorchSim ghcr.io/psal-postech/torchsim-ci:v1.1.0
```

進入容器並執行命令：
```bash
docker exec -it function92-PyTorchSim bash
```

(可選)測試官方PyTorchSim

刪除官方PyTorchSim：
```bash
rm -rf /workspace/PyTorchSim
```

克隆eclab_legosim倉庫：
```bash
cd /workspace/legomerged
git clone git@github.com:Weng20011103/eclab_legosim.git
# 後續參考eclab_docs/install_legosim.md
```

克隆PyTorchSim倉庫
```bash
cd eclab_legosim
rm -rf PyTorchSim
git clone git@github.com:whoami9203/PyTorchSim.git
```

停止容器：
```bash
docker stop ollie-PyTorchSim
```

重啟容器：
```bash
docker restart ollie-PyTorchSim
```
