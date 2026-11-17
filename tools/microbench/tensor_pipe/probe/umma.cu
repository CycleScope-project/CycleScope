// See ../umma_impl.cuh.  Cells: all umma_kinds with A from SMEM, and
// f16/tf32/f8f6f4 with A from TMEM, cta_group 1 and 2, plus the no-MMA
// commit baseline per cta_group.
#include "../umma_impl.cuh"

int main(int argc, char** argv) {
  std::vector<UCase> v;
  uadd<kF16, 1, 0, kXw>(v, std::integer_sequence<int, 0>{});
  uadd<kF16, 2, 0, kXw>(v, std::integer_sequence<int, 0>{});
#define CELL(KIND)            \
  uadd_cell<KIND, 1, 0>(v);   \
  uadd_cell<KIND, 2, 0>(v);
  CELL(kF16) CELL(kTF32) CELL(kF8F6F4) CELL(kMXF8F6F4) CELL(kMXF4) CELL(kMXF4NVF4)
#undef CELL
  uadd_cell<kF16, 1, 1>(v);
  uadd_cell<kF16, 2, 1>(v);
  uadd_cell<kTF32, 1, 1>(v);
  uadd_cell<kTF32, 2, 1>(v);
  uadd_cell<kF8F6F4, 1, 1>(v);
  uadd_cell<kF8F6F4, 2, 1>(v);
  uadd_cell<kMXF8F6F4, 1, 1>(v);
  uadd_cell<kMXF8F6F4, 2, 1>(v);
  uadd_cell<kMXF4, 1, 1>(v);
  uadd_cell<kMXF4, 2, 1>(v);
  uadd_cell<kMXF4NVF4, 1, 1>(v);
  uadd_cell<kMXF4NVF4, 2, 1>(v);
  return umma_main(argc, argv, "umma", v);
}
