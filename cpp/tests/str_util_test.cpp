// str_util_test — split_delim must match std::getline tokenization.

#include <cassert>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "convert/str_util.hpp"

namespace {

std::vector<std::string> via_getline(const std::string &s, char d) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, d)) out.push_back(t);
  return out;
}

}  // namespace

int main() {
  const std::string cases[] = {
      "",       "a",      "a.b",    "a.",     ".",      "..",
      "a..b",   ".a",     "a.b.c",  "a...c",  "..x..",  "abc.def.ghi.",
      ".a.b.c", "x.",     "LDGSTS.E.BYPASS.128",
      "UTOPIA..X64",    "...",    "a...................b",
  };
  for (const auto &d : {'.', ',', ':'}) {
    for (const auto &s : cases) {
      const auto want = via_getline(s, d);
      const auto got = cyclescope_convert::split_delim(s, d);
      assert(got == want);
    }
  }
  std::puts("str_util_test OK");
  return 0;
}
