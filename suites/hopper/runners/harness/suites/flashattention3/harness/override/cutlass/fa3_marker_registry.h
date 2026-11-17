/***************************************************************************************************
 * Cross-TU host-side registry for the FA3 delivery device_kernel marker override.
 *
 * Every translation unit that compiles the device_kernel override (i.e. every nvcc TU that
 * includes <cutlass/device_kernel.h> from the override directory) owns a private set of
 * `static __device__` marker symbols.  Each TU registers its own host setter (via a static
 * registrar object) here.  The flash_api.cpp extension then walks this registry once to point
 * every TU's symbols at the same device buffers.
 *
 * This header is pure C++ (no CUDA headers, no __device__ code) so host-only TUs such as
 * flash_api.cpp can include it too.
 **************************************************************************************************/
#pragma once

#include <vector>

namespace cutlass {
namespace fa3_marker {

// Setter contract: each TU's host setter runs cudaMemcpyToSymbol on its own three
// `static __device__` symbols (cutlass_single_cta_marker_ptr / _grid_ids / _grids).
using MarkerSetterFn = void (*)(unsigned long long* markers,
                                unsigned long long* grid_ids,
                                unsigned long long* grids);

// Function-local static inside an inline function: exactly one shared instance per .so,
// visible to every TU (ODR, C++17).
inline std::vector<MarkerSetterFn>& registry() {
  static std::vector<MarkerSetterFn> instance;
  return instance;
}

// Registers one setter per TU; de-duplicates by function pointer so a TU can never
// register the same setter twice.  Returns the registry size after registration.
inline int register_marker_setter(MarkerSetterFn fn) {
  auto& setters = registry();
  for (MarkerSetterFn existing : setters) {
    if (existing == fn) {
      return static_cast<int>(setters.size());
    }
  }
  setters.push_back(fn);
  return static_cast<int>(setters.size());
}

// Static registrar token: construct one per TU (internal linkage) to enroll the TU's
// setter into the registry at static-init time.
struct RegistrarToken {
  explicit RegistrarToken(MarkerSetterFn fn) { register_marker_setter(fn); }
};

inline int registry_size() {
  return static_cast<int>(registry().size());
}

// Walks the registry and points every TU's device symbols at the shared buffers.
inline void set_all_tu_buffers(unsigned long long* markers,
                               unsigned long long* grid_ids,
                               unsigned long long* grids) {
  for (MarkerSetterFn fn : registry()) {
    fn(markers, grid_ids, grids);
  }
}

}  // namespace fa3_marker
}  // namespace cutlass
