#ifndef ARCH_TABLE_H
#define ARCH_TABLE_H

/* Per-architecture fact table (capture spec section 2.2).  This is the
 * ONLY place tracer.cu / inject.cu learn architecture differences: the
 * host picks one table at the first context init from the device
 * compute-capability major (9 -> hopper, 10 -> blackwell); no other code
 * may branch on the architecture. */

struct ArchTable {
  const char *name;
  /* Opcode prefixes whose operands are captured (the shared union:
   * prefixes that do not exist on an arch simply never match). */
  const char *const *operand_prefixes;
  int operand_prefix_count;
  /* Opcodes whose first [URx] operand packs dst|mbarrier: the mbarrier
   * companion word is URx+1. */
  const char *const *mbarrier_companion_roots;
  int mbarrier_companion_root_count;
  /* Opcodes whose [URx] operand IS the mbarrier word itself. */
  const char *const *mbarrier_self_roots;
  int mbarrier_self_root_count;
  /* UR base offset of the TMA coordinate bank, measured from the low
   * register of the first [URx] bracket; -1 = the root carries no
   * coordinate bank.  `root` is the opcode root (text before the first
   * '.'), `bracket_count` the number of [...] operand tokens. */
  int (*tma_coordinate_offset)(const char *root, int bracket_count);
  /* Whether the SASS grammar carries idesc[/gdesc[/tmem[URx] descriptor
   * operand tokens (UTC*MMA). */
  int has_descriptor_tokens;
};

/* Shared operand-capture family union (spec section 2.2, first row). */
static const char *const kOperandFamilyPrefixes[] = {
    "SYNCS", "ARRIVES",   "UTMA",   "UBLKCP",      "UBLKRED",
    "UCGABAR", "BAR",     "STAS",   "REDAS",       "MATCH",
    "NANOSLEEP", "LDGSTS", "UTC",   "UGETNEXTWORKID", "STS.64"};
#define K_OPERAND_FAMILY_COUNT \
  (sizeof(kOperandFamilyPrefixes) / sizeof(kOperandFamilyPrefixes[0]))
/* Opcodes captured by exact match (a prefix would over-match): plain
 * 32-bit STS carries a one-register mbarrier init word (converter
 * promotes it when the address is a named barrier word); STSM / STS.128
 * and sub-word STS forms never do. */
static const char *const kOperandFamilyExact[] = {"STS"};
#define K_OPERAND_EXACT_COUNT \
  (sizeof(kOperandFamilyExact) / sizeof(kOperandFamilyExact[0]))

/* Shared mbarrier-word rules (spec section 2.2, second row). */
static const char *const kMbarrierCompanionRoots[] = {
    "UTMALDG", "UTMAREDG", "UTMAPF", "UBLKCP"};
static const char *const kMbarrierSelfRoots[] = {"UTCBAR", "SYNCS",
                                                 "ARRIVES"};
#define K_MBAR_COMPANION_COUNT \
  (sizeof(kMbarrierCompanionRoots) / sizeof(kMbarrierCompanionRoots[0]))
#define K_MBAR_SELF_COUNT \
  (sizeof(kMbarrierSelfRoots) / sizeof(kMbarrierSelfRoots[0]))

#endif /* ARCH_TABLE_H */
