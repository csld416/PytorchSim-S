# build 相關指令
如果要修改TOGSim或PyTorchSimDevice (openreg)的程式碼，改完請在container裡執行以下指令重新編譯安裝。

TOGSim
```bash
cd TOGSim && \
mkdir -p build && \
cd build && \
conan install .. --build=missing && \
cmake .. && \
make -j4
```

PyTorchSimDevice (openreg)
```bash
cd PyTorchSimDevice
python3 -m pip install --no-build-isolation -e .
```