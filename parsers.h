#include <boost/algorithm/string.hpp>
#include <boost/timer/timer.hpp>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include "frame.h"

namespace pandas {

namespace fs = std::experimental::filesystem;

template <typename... DType>
[[nodiscard]] DataFrame read_csv(const fs::path &filepath) {
    std::cout << "reading " << filepath << "... " << std::flush;
    boost::timer::auto_cpu_timer t{3, "%w seconds\n"};
    std::ifstream ifs{filepath};
    std::string line;
    std::getline(ifs, line);
    std::vector<std::string> columns;
    boost::split(columns, line, [](char c){return c == ',';});

    auto data = std::vector{make_series<DType>({})...};
    while (ifs.peek() != std::ifstream::traits_type::eof()) {
        for (std::size_t i = 0; i < sizeof...(DType); ++i) {
            data[i]->emplace_back(ifs);
        }
    }
    return DataFrame(std::move(data), std::move(columns));
}

}
