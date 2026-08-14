# build 相關指令
如果要修改TOGSim或PyTorchSimDevice (openreg)的程式碼，改完請在container裡執行以下指令重新編譯安裝。

或是pull最新的版本後發現跑不動，也可以先嘗試執行以下指令重新編譯安裝。

TOGSim
```bash
cd TOGSim && \
mkdir -p build && \
cd build && \
conan install .. --build=missing && \
cmake .. && \
make -j4
```

> [!NOTE]
> 如果發生錯誤，把`build`資料夾刪除後再試一次。

PyTorchSimDevice (openreg)
```bash
cd PyTorchSimDevice
python3 -m pip install --no-build-isolation -e .
```