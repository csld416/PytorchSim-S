# build 相關指令
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