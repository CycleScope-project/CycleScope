#include "arch/architecture_module.hpp"

#include <cstring>

#include "arch/blackwell/module.hpp"
#include "arch/hopper/module.hpp"

namespace cyclescope {

std::unique_ptr<ArchitectureModule> make_architecture(
    const char* device) {
  if (device && std::strncmp(device, "h100", 4) == 0)
    return std::make_unique<HopperModule>();
  if (device && std::strncmp(device, "b300", 4) == 0)
    return std::make_unique<BlackwellModule>();
  return std::make_unique<NullArchitectureModule>();
}

}  // namespace cyclescope
