#pragma once

#include <execution>
#include <iostream>
#include <iterator>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "series.h"

namespace pandas {

class DataFrame {
public:
    [[nodiscard]] DataFrame(std::vector<series> &&data, std::vector<std::string> &&cols, series &&idx = make_series<std::size_t>({})) noexcept : values{data}, columns{cols}, index{idx} {}
    template <typename T>
    DataFrame loc(const T &start, const T &stop) const;
    DataFrame iloc(const std::vector<std::size_t> &ib) const;
    DataFrame iloc(const std::vector<bool> &ib) const;
    DataFrame corr() const;
    DataFrame pct_change() const;
    template <typename T>
    DataFrame resample(const T &start, const T &stop, const T &delta) const;
    series get_index() { return index; };
    DataFrame& set_index(const series &idx);
    DataFrame& set_index(const std::string &col);
    template <typename T>
    DataFrame filter(const std::string &col, const T &val) const;
    const series& operator[](const std::string &col) const;
    DataFrame operator[](std::vector<std::string> &&cols) const;
    void print(std::ostream &os) const;
private:
    std::size_t find_column_index(const std::string &col) const;
    void print_row(std::ostream &os, std::size_t i) const;
    std::vector<series> values;
    series index;
    std::vector<std::string> columns;
};

template <typename T>
DataFrame DataFrame::loc(const T &start, const T &stop) const {
    auto index_t = std::dynamic_pointer_cast<Series<T>>(index);
    std::vector<bool> index_b(index_t->size());
    std::transform(
        std::execution::par_unseq,
        std::begin(index_t->values),
        std::end(index_t->values),
        std::begin(index_b),
        [&start, &stop](const T &i) { return start <= i && i < stop; }
    );
    return iloc(index_b);
}

DataFrame DataFrame::iloc(const std::vector<std::size_t> &is) const {
    DataFrame df(*this);
    df.index = index->iloc(is);
    for (auto [itv, itr] = std::pair(std::begin(values), std::begin(df.values)); itv != std::end(values); ++itv, ++itr)
        *itr = (*itv)->iloc(is);
    return df;
}

DataFrame DataFrame::iloc(const std::vector<bool> &ib) const {
    DataFrame df(*this);
    df.index = index->iloc(ib);
    for (auto [itv, itr] = std::pair(std::begin(values), std::begin(df.values)); itv != std::end(values); ++itv, ++itr)
        *itr = (*itv)->iloc(ib);
    return df;
}

DataFrame DataFrame::corr() const {
    std::vector<series> data;
    for (const auto u : columns) {
        std::vector<double> col;
        for (const auto v : columns)
            col.push_back((*this)[u]->corr((*this)[v]));
        data.push_back(make_series(std::move(col)));
    }
    return DataFrame(
        std::move(data),
        std::vector(columns)
    );
}

DataFrame DataFrame::pct_change() const {
    DataFrame df(*this);
    for (auto [itv, itr] = std::pair(std::begin(values), std::begin(df.values)); itv != std::end(values); ++itv, ++itr)
        *itr = (*itv)->pct_change();
    return df;
}

template <typename T>
DataFrame DataFrame::resample(const T &start, const T &stop, const T &delta) const {
    auto index_ = std::dynamic_pointer_cast<Series<T>>(index);
    std::vector<T> index_t;
    index_t.reserve((stop - start) / delta);
    for (auto t = start + delta; t <= stop; t += delta)
        index_t.push_back(t);
    std::vector<std::size_t> index_s(std::size(index_t));
    for (std::size_t i = 0, j = 0, k = 0; i < index_->size() && j < std::size(index_t) && k < std::size(index_s); ) {
        if (index_->values[i] <= index_t[j]) {
            index_s[k] = i;
            ++i;
        } else {
            ++j; ++k;
        }
    }
    auto df = iloc(index_s);
    df.index = make_series(std::move(index_t));
    return df;
}

std::size_t DataFrame::find_column_index(const std::string &col) const {
    return std::distance(
        std::begin(columns),
        std::find(std::begin(columns), std::end(columns), col)
    );
}

DataFrame& DataFrame::set_index(const series &idx) {
    index = idx;
    return *this;
}

DataFrame& DataFrame::set_index(const std::string &col) {
    auto i = find_column_index(col);
    set_index(values[i]);
    values.erase(std::begin(values) + i);
    columns.erase(std::begin(columns) + i);
    return *this;
}

template <typename T>
DataFrame DataFrame::filter(const std::string &col, const T &val) const {
    auto i = find_column_index(col);
    auto s = std::dynamic_pointer_cast<Series<T>>(values[i]);
    std::vector<bool> index_b(s->size());
    std::transform(
        std::execution::par_unseq,
        std::begin(s->values),
        std::end(s->values),
        std::begin(index_b),
        [&val](const T &v) { return v == val; }
    );
    return iloc(index_b);
}

const series& DataFrame::operator[](const std::string &col) const {
    return values[find_column_index(col)];
}

DataFrame DataFrame::operator[](std::vector<std::string> &&cols) const {
    std::vector<series> data;
    for (const auto &col : cols)
        data.push_back(values[find_column_index(col)]);
    DataFrame df{std::move(data), std::move(cols)};
    df.index = index;
    return df;
}

void DataFrame::print_row(std::ostream &os, std::size_t i) const {
    index->print_value(os, i);
    for (const auto &s : values) {
        os << '\t';
        s->print_value(os, i);
    }
    os << std::endl;
}

void DataFrame::print(std::ostream &os) const {
    os << std::string(index->width(), ' ');
    for (const auto &col : columns) {
        os << '\t' << col;
    }
    os << std::endl << "index" << std::endl;
    if (index->size() > display_max_rows) {
        auto row_num = display_min_rows / 2;
        for (std::size_t i = 0; i < row_num; ++i)
            print_row(os, i);
        os << "..." << std::endl;
        for (std::size_t i = index->size() - row_num; i < index->size(); ++i)
            print_row(os, i);
    } else {
        for (std::size_t i = 0; i < index->size(); ++i)
            print_row(os, i);
    }
    os << std::endl << "[" << index->size() << " rows x " << std::size(values) << " columns]" << std::endl;
}

}

std::ostream& operator<<(std::ostream &os, pandas::DataFrame &df) {
    df.print(os);
    return os;
}
