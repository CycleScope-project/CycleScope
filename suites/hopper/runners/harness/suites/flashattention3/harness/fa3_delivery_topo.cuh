#pragma once

static constexpr unsigned int kFa3DeliveryTopoCtas = 32;
static constexpr unsigned int kFa3DeliveryTopoFields = 4;
static constexpr unsigned int kFa3DeliveryTopoSize =
    kFa3DeliveryTopoCtas * kFa3DeliveryTopoFields;

// Define the array only in the instantiation TU (FA3_DELIVERY_TOPO_DEFINE).
// Do not use extern __device__ here: without -rdc it becomes a second static copy.
#ifdef FA3_DELIVERY_TOPO_DEFINE
__device__ unsigned long long fa3_delivery_topo[kFa3DeliveryTopoSize];
#endif
