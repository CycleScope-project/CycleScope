// Arch converter rules interface — ports of arch/<arch>/rules.py
// (3a scope: operation_class, operation_classes, memory_space,
// instruction_flags, cache_hint). One implementation per arch in
// cpp/arch/<arch>/convert_rules.cpp; no arch branches in common code.
#pragma once

#include <cstdint>
#include <string>
#include <map>
#include <set>
#include <vector>

namespace cyclescope_convert {

struct DynOps;
struct SemanticsContext;
enum class SemOp : int;

struct StaticInfo {
  std::string opcode;
  std::vector<std::string> operands;
  std::string memory_space;   // "" = none
  bool has_predicate = false;
  std::uint64_t pc = 0;
  std::vector<std::string> dst_names;   // destination register names
  std::vector<std::string> src_names;   // static source register names
};

struct ConvertRules {
  virtual ~ConvertRules() = default;
  virtual std::string operation_class(const StaticInfo &s) const = 0;
  virtual std::vector<std::string>
  operation_classes(const StaticInfo &s) const = 0;
  virtual std::string memory_space(const StaticInfo &s) const = 0;
  virtual int instruction_flags(const StaticInfo &s) const = 0;
  // opcode_facts (axes): sparse opcode->int facts
  virtual std::map<std::string, long long>
  opcode_facts(const std::string &opcode) const { return {}; }
  // named barrier opcode roots (shared-address objects the unit
  // prescans for store-as-init promotion); empty = no prescan facts.
  virtual const std::set<std::string> &named_barrier_roots() const {
    static const std::set<std::string> kEmpty;
    return kEmpty;
  }
  // per-instruction promotion over the static semantic op (dyn/context
  // facts may upgrade a plain op, e.g. blackwell STS.64 -> init).
  virtual SemOp promote_operation(const StaticInfo &,
                                  const DynOps &,
                                  const SemanticsContext &,
                                  SemOp op) const { return op; }
};

const ConvertRules *convert_rules_for(const std::string &arch);

}  // namespace cyclescope_convert
