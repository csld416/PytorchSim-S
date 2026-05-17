# 容器環境配置  
連線到 `wombat` 主機：
```bash
ssh eclab3090_2
```

建立 `/mnt/nvme0/ollie/workspace` 目錄：
```bash
mkdir -p /mnt/nvme0/ollie/workspace
```

啟動 Container 並掛載目錄：
```bash
docker run -d -it --ipc=host --gpus '"device=1"' --mount type=bind,source=/mnt/nvme0/ollie/workspace,target=/workspace -w /workspace/PyTorchSim -v $SSH_AUTH_SOCK:/ssh-agent -e SSH_AUTH_SOCK=/ssh-agent --name ollie-PyTorchSim ghcr.io/psal-postech/torchsim-ci:v1.1.0
```

進入容器並執行命令：
```bash
docker exec -it ollie-PyTorchSim bash
```

克隆倉庫：
```bash
# 在/workspace/PyTorchSim 底下執行
git clone git@github.com:whoami9203/PyTorchSim.git
```

把倉庫內容移到 `/workspace/PyTorchSim` 目錄下：
```bash
shopt -s dotglob
mv /workspace/PyTorchSim/PyTorchSim/* /workspace/PyTorchSim/
shopt -u dotglob
rmdir /workspace/PyTorchSim/PyTorchSim
```

停止容器：
```bash
docker stop ollie-PyTorchSim
```

重啟容器：
```bash
docker restart ollie-PyTorchSim
```
