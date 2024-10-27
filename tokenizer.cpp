#include <pybind11/pybind11.h>
#include <vector>
#include <string>
#include <sstream>

std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream stream(text);
    std::string token;
    while (stream >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

PYBIND11_MODULE(tokenizer, m) {
    m.def("tokenize", &tokenize, "A function that tokenizes text");
}
