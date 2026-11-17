#ifndef TENSOR_MAP_OPERAND_H
#define TENSOR_MAP_OPERAND_H

/* Map-operand role classification for TMA-root instructions, decided per
 * instruction from the exact decoded SASS grammar only -- never from a
 * case/PC table, kernel identity, shape, or value numerics.
 *
 * The tensormap address rides a bracketed UR pair (the pair's low UREG is
 * reported in *map_ureg and the tracer samples that UREG and its +1
 * companion at issue, the established map_ureg/map_ureg+1 pair
 * convention).  The trailing desc[URx] operand, when present, is a
 * separate cache-policy operand captured by the TMA mbarrier sidecar
 * path (tma_descriptor_ureg); it is never the map source.
 *
 * Decoded-SASS facts (diag9b static_sass_map.csv, CU-75 GemmUniversal
 * 2x1SM; cute/arch/copy_sm100_tma.hpp: the tensormap rides [%1] of the
 * two-operand PTX form; the NVBit 3-operand lowering keeps the tensor
 * map as operand 1 while operand 0's MREF pair lowers to the shared
 * destination/mbarrier words):
 *   UTMALDG.2D [UR16], [UR26], desc[UR12] ;
 *   UTMALDG.2D.MULTICAST [UR8], [UR30], UR34, desc[UR38] ;
 *   UTMALDG.2D [UR8], [UR12] ;               (descriptor-less encoding)
 *   UTMASTG.2D [UR40], [UR54] ;              (no desc operand)
 *   UTMASTG.4D [UR8], [UR28], desc[UR30] ;
 *   UTMAPF.3D [UR44], desc[URZ] ;
 *   UTMAPF.L2.3D [UR8], [UR4] ;          (two-bracket: the SECOND bracket
 *     is the tensormap pair -- genaudit_sass_probe/utmapf.sass shows
 *     UR4 = LDCU.64 c[0x380] tensormap pointer while UR8..UR10 carry the
 *     coordinates; the first bracket is NOT a dst|mbarrier pair)
 *   UTMACCTL.IV [UR22] ;                     (descriptor-cache control)
 * Roles (inst_trace_t.tma_tensor_operand_role):
 *   1 tensor_map_address: the bracketed UR pair is the genuine tensormap
 *     generic address (r5d cross-path proof: the 16 UTMASTG issue images
 *     are byte-identical to the independent UTMACCTL.IV publish images
 *     at the same pool offsets).  *map_ureg = the pair's low UREG:
 *       - UTMALDG/UTMAREDG/UTMASTG two-bracket forms: the SECOND bracket
 *         is the tensormap pair (operand 0's first MREF is the shared
 *         destination/mbarrier pair, proven by the mbarrier companion
 *         rules in tracer_tool.cu);
 *       - UTMAPF two-bracket form: the SECOND bracket is the tensormap
 *         pair (same convention as UTMALDG/UTMASTG);
 *       - UTMAPF one-bracket form: the single bracket is the map pair.
 *     Only trailing NON-bracketed operands may follow the brackets
 *     (desc[URx|URZ], bare mask/flag registers or immediates); any
 *     further bracketed operand, URZ/RZ bracket, or bracket pair whose
 *     low UREG exceeds the T0025 sampling bound is rejected.
 *   2 encoded_descriptor: reserved; not produced by the current grammar
 *     (kept for CSV-schema stability of map_operand_role rows written by
 *     earlier tracers).
 *   0 unknown: any other form is rejected (status 2), never guessed.
 * Returns true iff the form matched one of the proven grammars.
 *
 * Host-pure by design: tracer_tool.cu and the CPU smoke binary compile
 * the same function so the grammar exercised by the smoke test is the
 * grammar shipped in the .so. */

#include <regex>
#include <string>


static bool tma_map_operand_class(const std::string &opcode,
                                  const char *sass_cstr, int *role,
                                  int *map_ureg) {
  *role = 0;
  *map_ureg = -1;
  if (sass_cstr == NULL || sass_cstr[0] == '\0')
    return false;
  std::string sass(sass_cstr);
  static const std::regex comment_re(R"(/\*.*?\*/)");
  sass = std::regex_replace(sass, comment_re, " ");
  std::vector<std::string> operands;
  if (!sass_operand_tokens(sass_cstr, opcode, &operands)) return false;
  for (const auto &operand : operands) {
    if (operand.rfind("desc[", 0) == 0 &&
        !ureg_readable(decode_ureg_text(operand, UregRole::Descriptor)))
      return false;
    if (operand.rfind("UR", 0) == 0 &&
        !ureg_readable(decode_ureg_text(operand, UregRole::Bare)))
      return false;
  }
  /* Trailing operands after the proven map brackets are non-bracketed
   * tokens only: desc[URx|URZ] cache-policy encodings, bare multicast
   * mask registers/immediates, and other bare tail operands.  A third
   * bracketed operand is never a proven tail. */
  if (opcode.rfind("UTMALDG", 0) == 0 || opcode.rfind("UTMAREDG", 0) == 0) {
    /* Two-bracket form: [URdst], [URmap]; the second bracket is the
     * tensormap generic-address pair. */
    static const std::regex ldg_re(
        R"(^\s*(?:@!?U?P[0-9]+\s+)?UTMA(?:LDG|REDG)(?:\.[A-Za-z0-9_]+)*\s+\[UR([0-9]{1,3})\],\s*\[UR([0-9]{1,3})\](?:,\s*(?:desc\[UR(?:[0-9]{1,3}|Z)\]|[^,\s\[\]]+))*\s*;\s*$)",
        std::regex::icase);
    std::smatch match;
    if (!std::regex_match(sass, match, ldg_re))
      return false;
    const int ureg = std::stoi(match[2].str());
    if (!ureg_numeric_span(std::stoi(match[1].str()), 2) ||
        !ureg_numeric_span(ureg, 2))
      return false;
    *role = 1;
    *map_ureg = ureg;
    return true;
  }
  if (opcode.rfind("UTMASTG", 0) == 0) {
    /* Two-bracket form: [URsrc], [URmap]; the second bracket is the
     * tensormap generic-address pair. */
    static const std::regex stg_re(
        R"(^\s*(?:@!?U?P[0-9]+\s+)?UTMASTG(?:\.[A-Za-z0-9_]+)*\s+\[UR([0-9]{1,3})\],\s*\[UR([0-9]{1,3})\](?:,\s*(?:desc\[UR(?:[0-9]{1,3}|Z)\]|[^,\s\[\]]+))*\s*;\s*$)",
        std::regex::icase);
    std::smatch match;
    if (!std::regex_match(sass, match, stg_re))
      return false;
    const int map_ureg_val = std::stoi(match[2].str());
    if (!ureg_numeric_span(std::stoi(match[1].str()), 2) ||
        !ureg_numeric_span(map_ureg_val, 2))
      return false;
    *role = 1;
    *map_ureg = map_ureg_val;
    return true;
  }
  if (opcode.rfind("UTMAPF", 0) == 0) {
    /* Two-bracket form: [URcoords], [URmap]; the SECOND bracket is the
     * tensormap pair (cp.async.bulk.prefetch.tensor lowers its map
     * operand to the second bracket, matching UTMALDG/UTMASTG). */
    static const std::regex utmapf2_re(
        R"(^\s*(?:@!?U?P[0-9]+\s+)?UTMAPF(?:\.[A-Za-z0-9_]+)*\s+\[UR([0-9]{1,3})\],\s*\[UR([0-9]{1,3})\](?:,\s*(?:desc\[UR(?:[0-9]{1,3}|Z)\]|[^,\s\[\]]+))*\s*;\s*$)",
        std::regex::icase);
    /* One-bracket form: [URmap]; the single bracket is the tensormap
     * generic-address pair. */
    static const std::regex utmapf_re(
        R"(^\s*(?:@!?U?P[0-9]+\s+)?UTMAPF(?:\.[A-Za-z0-9_]+)*\s+\[UR([0-9]{1,3})\](?:,\s*(?:desc\[UR(?:[0-9]{1,3}|Z)\]|[^,\s\[\]]+))*\s*;\s*$)",
        std::regex::icase);
    std::smatch match;
    if (std::regex_match(sass, match, utmapf2_re)) {
      const int map_ureg_val = std::stoi(match[2].str());
      if (!ureg_numeric_span(std::stoi(match[1].str()), 2) ||
          !ureg_numeric_span(map_ureg_val, 2))
        return false;
      *role = 1;
      *map_ureg = map_ureg_val;
      return true;
    }
    if (!std::regex_match(sass, match, utmapf_re))
      return false;
    const int ureg = std::stoi(match[1].str());
    if (!ureg_numeric_span(ureg, 2))
      return false;
    *role = 1;
    *map_ureg = ureg;
    return true;
  }
  return false;
}

#endif /* TENSOR_MAP_OPERAND_H */
