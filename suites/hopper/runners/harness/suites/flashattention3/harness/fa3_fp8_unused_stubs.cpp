#include <cuda_runtime.h>

#include "flash.h"

#include <cutlass/numeric_types.h>
#include <stdexcept>

// The delivery build only instantiates the fp8 head_dim 192 forward kernel.
// flash_api still references the other fp8 head dims, with and without the
// PackGQA specialization, so every one of those needs a definition to link.
#define FA3_UNUSED_FP8_FWD(HEAD_DIM, PACK_GQA)                                 \
    template <>                                                                \
    void run_mha_fwd_<90, cutlass::float_e4m3_t, HEAD_DIM, HEAD_DIM, false,    \
                      false, false, PACK_GQA>(Flash_fwd_params &,              \
                                              cudaStream_t) {                  \
        throw std::runtime_error(                                              \
            "FA3 dual-SM build omitted fp8 hdim" #HEAD_DIM);                   \
    }

FA3_UNUSED_FP8_FWD(128, false)
FA3_UNUSED_FP8_FWD(128, true)
FA3_UNUSED_FP8_FWD(256, false)
FA3_UNUSED_FP8_FWD(256, true)
