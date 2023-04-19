#include "argparse.h"

namespace args {
    Parser::Parser() {

    }

    ParamValue Parser::getParam(const char* paramStr, Param* param) {
        
        // TODO: make this faster with some hashing or store as enum
        ParamValue v;
        if (!param->type.compare("int")) {
            v.intVal = std::atoi(paramStr);
        } else if (!param->type.compare("float")) {
            v.floatVal = std::atof(paramStr);
        } else {
            throw std::runtime_error("Unknown parameter type");
        }

        return v;
    }

    void Parser::parseOperation(std::vector<Param*>& options, char* argv[]) {
        for (size_t i = 0; i < options.size(); i++) {
            matched_params.push_back(getParam(argv[i], options[i]));
        }
    }

    std::string_view Parser::parseArguments(size_t argc, char* argv[]) {
        if (argc < 2) {
            throw std::runtime_error("Not enough args");
        }

        const char* type = argv[1];
        if (!std::strcmp(type, "help")) {
            printHelp();
            return "help";
        }

        auto it = options.find(type);
        if (it == options.end()) {
            throw std::runtime_error("Param type not found");
        }

        if (argc - 2 != it->second.size()) {
            throw std::runtime_error("Too many or too little parameters");
        }

        parseOperation(it->second, argv + 2);
        return it->first;
    }

    void Parser::printHelp() {
        std::cout << "Available options: " << std::endl;

        for (auto& [opType, opParams] : options) {
            std::cout << opType << " ";

            for (auto& param  : opParams) {
                std::cout << param->type << '(' << param->desc << ')' << ' ';
            }

            std::cout << std::endl;
        }
    }

    Param* Parser::splitParameter(const char* p) {
        std::string_view tmp(p);
        
        std::size_t openPos = tmp.find('(');
        std::size_t closePos = tmp.rfind(')');

        if (openPos == std::string_view::npos || closePos == std::string_view::npos) {
            throw std::runtime_error("Invalid parameter format\n");
        }

        std::string_view typeStr = tmp.substr(0, openPos);
        std::string_view descStr = tmp.substr(openPos + 1, closePos - openPos - 1);

        return new Param{typeStr, descStr};
    }
}