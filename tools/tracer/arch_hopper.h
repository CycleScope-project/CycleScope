#ifndef ARCH_HOPPER_H
#define ARCH_HOPPER_H

#include <string.h>
#include "arch_table.h"

/* Hopper (sm_90a) table.  Coordinate bank offsets after the first [URx]
 * bracket's low register: UTMALDG packs dst|mbarrier in the leading pair
 * (+2); UTMASTG/UTMAREDG carry a single leading word (+1); UTMAPF's
 * bracket is the map pair so the coordinates start at +0. */
static int hopper_tma_coordinate_offset(const char *root,
                                        int bracket_count) {
  (void)bracket_count;
  if (strcmp(root, "UTMALDG") == 0)
    return 2;
  if (strcmp(root, "UTMASTG") == 0)
    return 1;
  if (strcmp(root, "UTMAREDG") == 0)
    return 1;
  if (strcmp(root, "UTMAPF") == 0)
    return 0;
  return -1;
}

static const ArchTable kArchHopper = {
    "hopper",
    kOperandFamilyPrefixes,
    (int)K_OPERAND_FAMILY_COUNT,
    kMbarrierCompanionRoots,
    (int)K_MBAR_COMPANION_COUNT,
    kMbarrierSelfRoots,
    (int)K_MBAR_SELF_COUNT,
    hopper_tma_coordinate_offset,
    0,
};

#endif /* ARCH_HOPPER_H */
