#ifndef UNIFORM_REGISTER_H
#define UNIFORM_REGISTER_H

#include <cctype>
#include <regex>
#include <string>
#include <vector>

// Inclusive T0025 sampling policy, not an architectural or NVBit SDK limit.
constexpr int kUregNumericMax = 99;
constexpr int kUregZero = -2;

enum class UregKind { Numeric, Zero, Unknown };
struct UregOperand {
  UregKind kind = UregKind::Unknown;
  int index = -1;
  const char *reason = "unknown_zero_marker";
};

inline bool ureg_numeric_span(int base, int words = 1) {
  return words > 0 && words <= kUregNumericMax + 1 && base >= 0 &&
         base <= kUregNumericMax - (words - 1);
}

inline std::string operand_trim(const std::string &text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  const auto last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

// Split only a complete decoded instruction; commas inside brackets are not
// operand boundaries. A missing terminator or mismatched opcode is unknown.
inline bool sass_operand_tokens(const char *sass, const std::string &opcode,
                                std::vector<std::string> *operands) {
  operands->clear();
  if (sass == nullptr) return false;
  static const std::regex comments(R"(/\*.*?\*/)");
  std::string text = operand_trim(std::regex_replace(sass, comments, " "));
  if (text.empty() || text.back() != ';') return false;
  text.pop_back();
  if (!text.empty() && text.front() == '@') {
    const auto end = text.find_first_of(" \t");
    if (end == std::string::npos) return false;
    text = operand_trim(text.substr(end));
  }
  const auto end = text.find_first_of(" \t");
  if (text.substr(0, end) != opcode) return false;
  if (end == std::string::npos) return true;
  text = text.substr(end + 1);
  int brackets = 0;
  size_t begin = 0;
  for (size_t i = 0; i <= text.size(); ++i) {
    if (i < text.size()) {
      if (text[i] == '[') ++brackets;
      if (text[i] == ']' && --brackets < 0) return false;
    }
    if (i == text.size() || (text[i] == ',' && brackets == 0)) {
      const auto token = operand_trim(text.substr(begin, i - begin));
      if (token.empty()) return false;
      operands->push_back(token);
      begin = i + 1;
    }
  }
  return brackets == 0;
}

enum class UregRole { Bare, Address, Descriptor };
inline UregOperand decode_ureg_text(const std::string &text, UregRole role) {
  const std::string token = operand_trim(text);
  std::smatch match;
  static const std::regex bare(R"(^UR(Z|[0-9]{1,3})$)");
  static const std::regex descriptor(R"(^desc\[UR(Z|[0-9]{1,3})\]$)");
  static const std::regex address(
      R"(^\[(?:(?:R(?:Z|[0-9]{1,3})(?:\.(?:U32|X1))?)\s*\+\s*)?UR(Z|[0-9]{1,3})(?:\.(?:U32|X1))?(?:\s*[+-]\s*(?:0x[0-9a-fA-F]+|[0-9]+))?\]$)");
  const std::regex *grammar = &bare;
  if (role == UregRole::Address) grammar = &address;
  if (role == UregRole::Descriptor) grammar = &descriptor;
  if (!std::regex_match(token, match, *grammar)) return {};
  if (match[1].str() == "Z")
    return {UregKind::Zero, kUregZero, "constant_zero"};
  const int index = std::stoi(match[1].str());
  if (!ureg_numeric_span(index))
    return {UregKind::Unknown, index, "unsupported_sampling_bound"};
  return {UregKind::Numeric, index, "decoded_numeric"};
}

inline UregOperand reconcile_ureg_operand(const std::string &operand_text,
                                          const std::string &sass_token,
                                          UregRole role, int sdk_index,
                                          bool sdk_index_present) {
  UregOperand decoded = decode_ureg_text(operand_text, role);
  const UregOperand sass = decode_ureg_text(sass_token, role);
  if (!operand_trim(operand_text).empty()) {
    if (decoded.kind == UregKind::Unknown) return decoded;
    if (!sass_token.empty() &&
        (sass.kind != decoded.kind || sass.index != decoded.index))
      return {UregKind::Unknown, -1, "operand_decode_conflict"};
  } else {
    decoded = sass;
  }
  if (decoded.kind == UregKind::Numeric && sdk_index_present &&
      sdk_index >= 0 && sdk_index != decoded.index)
    return {UregKind::Unknown, -1, "operand_decode_conflict"};
  return decoded;
}

inline bool ureg_readable(const UregOperand &operand, int words = 1) {
  return operand.kind == UregKind::Zero ||
         (operand.kind == UregKind::Numeric &&
          ureg_numeric_span(operand.index, words));
}


#include "nvbit.h"
#include <bitset>
#include <cstdio>
#include <cstdlib>

// Independently observed API reads on SM103/NVBit 1.8, not an architectural
// bound: ureg-unified-20261011T2035Z-r1/run-r8, pair bases 0..78, 32 lanes.
// Absent external evidence uses this traced set; it never enables 80..99.
inline std::bitset<kUregNumericMax + 1> ureg_backend_evidence() {
  std::bitset<kUregNumericMax + 1> registers;
  const char *cursor = std::getenv("CYCLESCOPE_TRACER_UREG_VALIDATED");
  if (cursor == nullptr) {
    for (int index = 0; index <= 79; ++index) registers.set(index);
    return registers;
  }
  while (*cursor != '\0') {
    char *end = nullptr;
    const unsigned long index = std::strtoul(cursor, &end, 10);
    if (*cursor < '0' || *cursor > '9' || end == cursor ||
        index > kUregNumericMax || (*end != '\0' && *end != ',') ||
        (*end == ',' && end[1] == '\0')) {
      std::fprintf(stderr, "ureg_backend_evidence_invalid:register_set\n");
      std::exit(2);
    }
    registers.set(index);
    cursor = *end == ',' ? end + 1 : end;
  }
  return registers;
}

inline bool ureg_backend_numeric_span(int base, int words = 1) {
  if (!ureg_numeric_span(base, words)) return false;
  static const auto registers = ureg_backend_evidence();
  for (int word = 0; word < words; ++word) {
    if (!registers.test(base + word)) return false;
  }
  return true;
}

inline bool ureg_sampling_available(const UregOperand &operand, int words = 1) {
  return operand.kind == UregKind::Zero ||
         (operand.kind == UregKind::Numeric &&
          ureg_backend_numeric_span(operand.index, words));
}


// Callers carry availability separately. A fixed-width placeholder never
// upgrades an unknown/unsupported operand to captured. Zero is a known value.
inline void add_ureg_word(const Instr *instr, int base, int word = 0,
                          int words = 1) {
  if (ureg_backend_numeric_span(base, words) && word >= 0 && word < words) {
    nvbit_add_call_arg_ureg_val(instr, base + word);
  } else {
    nvbit_add_call_arg_const_val32(instr, 0);
  }
}

#endif /* UNIFORM_REGISTER_H */
