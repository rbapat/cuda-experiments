#pragma once

#include <iostream>
#include <regex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace args {
struct Param {
  std::string_view type;
  std::string_view desc;
};

union ParamValue {
  int intVal;
  float floatVal;

  long span8bytes;
};

class Parser {
 public:
  Parser();

  template <class... Args>
  void registerOption(std::string_view type, Args... args) {
    auto it = options.find(type);
    if (it != options.end()) {
      throw std::runtime_error("Duplicate option registered");
    }

    std::vector<Param*> params;
    (params.push_back(std::forward<Param*>(splitParameter(args))), ...);
    options.insert(std::make_pair(type, params));
  }

  template <typename T>
  T get(size_t idx) {
    if (idx >= matched_params.size()) {
      throw std::runtime_error("Parameter doesn't exist");
    }

    return (T)matched_params[idx].span8bytes;
  }

  std::string_view parseArguments(size_t argc, char* argv[]);

 private:
  std::unordered_map<std::string_view, std::vector<Param*>> options;
  std::vector<ParamValue> matched_params;

  Param* splitParameter(const char* p);
  void printHelp();
  void parseOperation(std::vector<Param*>& options, char* argv[]);
  ParamValue getParam(const char* paramStr, Param* param);
};
}  // namespace args
