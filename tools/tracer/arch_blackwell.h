#ifndef ARCH_BLACKWELL_H
#define ARCH_BLACKWELL_H

#include <string.h>
#include "arch_table.h"

/* Blackwell (sm_103a) table.  Coordinate bank offsets after the first
 * [URx] bracket's low register: UTMALDG/UBLKCP pack dst|mbarrier in the
 * leading pair (+2); UTMASTG/UTMAREDG carry one leading word (+1);
 * UTMAPF's two-bracket form puts the coordinates in the FIRST bracket
 * (+0), its one-bracket form keeps the post-pair convention (+2). */
static int blackwell_tma_coordinate_offset(const char *root,
                                           int bracket_count) {
  if (strcmp(root, "UTMALDG") == 0 || strcmp(root, "UBLKCP") == 0)
    return 2;
  /* stores/reductions carry a single leading word (the shared source)
   * before the coordinate bank: verified on CU3-C70-F16-C03 UTMASTG.3D
   * [UR48],[UR4] where UR49..UR51 step by box_dim[0] across tiles */
  if (strcmp(root, "UTMASTG") == 0 || strcmp(root, "UTMAREDG") == 0)
    return 1;
  if (strcmp(root, "UTMAPF") == 0)
    return bracket_count >= 2 ? 0 : 2;
  return -1;
}

static const ArchTable kArchBlackwell = {
    "blackwell",
    kOperandFamilyPrefixes,
    (int)K_OPERAND_FAMILY_COUNT,
    kMbarrierCompanionRoots,
    (int)K_MBAR_COMPANION_COUNT,
    kMbarrierSelfRoots,
    (int)K_MBAR_SELF_COUNT,
    blackwell_tma_coordinate_offset,
    1,
};

#endif /* ARCH_BLACKWELL_H */
