// proxy_fence (schema 29): nothing extra — the leaf resolve lives in
// module.cpp::on_load, inflight tracking in on_instance_issue and
// the fence gate on descriptor_visibility op 5.  Kept as a separate
// TU per the arch-module file layout.

#include "arch/hopper/module.hpp"

namespace cyclescope {
}  // namespace cyclescope
