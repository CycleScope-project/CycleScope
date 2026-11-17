// Host wrappers live in flash_api.cpp (PyInit__C methods).
// Device stores and cudaMemcpy* helpers live in fa3_fwd_instances.cu.
// This TU is intentionally not compiled: a second PYBIND11_MODULE / PyInit__C
// conflicts with hopper/flash_api.cpp's dummy _C module.
#include "fa3_delivery_topo.cuh"
