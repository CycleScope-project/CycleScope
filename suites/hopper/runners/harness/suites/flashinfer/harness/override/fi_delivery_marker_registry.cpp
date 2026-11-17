// Host-side cross-TU registry for the FlashInfer delivery marker channel
// (T0005 修订 3-② / 修订 5; see harness/override/fi_delivery_marker.h).
//
// Built once by scripts/build.sh into build/libdelivery_fi_marker.so and linked into every
// instrumented JIT module (FLASHINFER_EXTRA_LDFLAGS), so all JIT .so files share this one
// registry instance.  Each instrumented JIT TU registers its private static setter at .so load
// time (FI_DELIVERY_MARKER_REGISTER_MODULE), and harness/marker_timing.py drives the registry
// through ctypes:
//   - flashinfer_delivery_marker_module_count / _module_names: discover the loaded instrumented
//     TUs (registry order) and allocate their marker buffers;
//   - flashinfer_delivery_marker_set_buffers: point every registered TU's device symbols at the
//     shared buffers (registry order; pointers passed as one flat array, 4 slots per module).
#include "fi_delivery_marker.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Entry {
  void (*fn)(const void* const*, int);
  std::string module_name;
};

std::vector<Entry>& entries() {
  static std::vector<Entry> instance;
  return instance;
}

constexpr int kSlotsPerModule = 4;

}  // namespace

extern "C" {

void flashinfer_delivery_marker_register(void (*setter_fn)(const void* const*, int),
                                         const char* module_name) {
  if (setter_fn == nullptr) {
    return;
  }
  auto& list = entries();
  for (const Entry& existing : list) {
    // Same-TU re-registering guard (static init runs once per process).  The key is
    // (fn, module_name): one JIT module TU includes several patched headers (page.cuh plus
    // decode.cuh/prefill.cuh/...), and the linker folds their identical static setter
    // functions to one address — an fn-only key collapsed every module to its first
    // registrar (H100 diag 2026-09-30: 10 modules -> 10 "page" entries, zero decode).
    if (existing.fn == setter_fn && existing.module_name ==
            (module_name == nullptr ? "<unnamed>" : module_name)) {
      return;
    }
  }
  list.push_back({setter_fn, module_name == nullptr ? "<unnamed>" : module_name});
}

void flashinfer_delivery_marker_set_buffers(const void* const* region_ptrs, int num_regions) {
  // region_ptrs holds kSlotsPerModule slots per registered module, registry order.
  for (size_t index = 0; index < entries().size(); ++index) {
    const int offset = static_cast<int>(index) * kSlotsPerModule;
    if (offset + kSlotsPerModule > num_regions && offset >= num_regions) {
      break;
    }
    entries()[index].fn(region_ptrs + offset,
                        num_regions - offset < kSlotsPerModule ? num_regions - offset
                                                               : kSlotsPerModule);
  }
}

int flashinfer_delivery_marker_module_count() {
  return static_cast<int>(entries().size());
}

int flashinfer_delivery_marker_module_names(char* storage, int slot_stride, int max_slots) {
  auto& list = entries();
  if (storage != nullptr && slot_stride > 0) {
    const int count = static_cast<int>(list.size()) < max_slots ? static_cast<int>(list.size())
                                                                : max_slots;
    for (int i = 0; i < count; ++i) {
      char* slot = storage + static_cast<size_t>(i) * static_cast<size_t>(slot_stride);
      std::snprintf(slot, static_cast<size_t>(slot_stride), "%s", list[i].module_name.c_str());
    }
  }
  return static_cast<int>(list.size());
}

}  // extern "C"
