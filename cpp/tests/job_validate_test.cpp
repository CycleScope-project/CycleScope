// validate: clean fixture passes; corrupted index range is rejected.

#include <cstdio>
#include <fstream>
#include <string>

#include "core/job_reader.hpp"

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  const std::string dir = argv[1];
  try {
    cyclescope::Job job = cyclescope::read_container_file(dir + "/job.bin");
    if (!cyclescope::validate_job(job).empty()) {
      std::fprintf(stderr, "clean fixture failed validation\n");
      return 1;
    }
    // corrupt: warp instance range beyond INST
    job.warps[0].instance_count = job.instances.size() + 1;
    if (cyclescope::validate_job(job).empty()) {
      std::fprintf(stderr, "corrupted job accepted\n");
      return 1;
    }
    std::printf("job_validate_test ok\n");
  } catch (const cyclescope::FormatError& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
