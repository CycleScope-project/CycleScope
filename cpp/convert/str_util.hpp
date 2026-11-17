#pragma once
// Small string helpers shared by the converter.

#include <string>
#include <string_view>
#include <vector>

namespace cyclescope_convert {

// Same token stream as `std::getline(std::stringstream(s), tok, d)`:
// empty input -> no tokens; consecutive delimiters -> empty tokens;
// a trailing delimiter produces no empty token ("a." -> {"a"},
// "a..b" -> {"a","","b"}, "." -> {""}).
inline std::vector<std::string> split_delim(std::string_view s, char d) {
  std::vector<std::string> out;
  std::size_t pos = 0;
  while (pos < s.size()) {
    const auto e = s.find(d, pos);
    if (e == std::string_view::npos) {
      out.emplace_back(s.substr(pos));
      break;
    }
    out.emplace_back(s.substr(pos, e - pos));
    pos = e + 1;
  }
  return out;
}

}  // namespace cyclescope_convert
