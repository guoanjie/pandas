#pragma once

#include <cassert>
#include <cmath>
#include <execution>
#include <iostream>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace pandas {

constexpr std::size_t display_min_rows = 10;
constexpr std::size_t display_max_rows = 60;

class SeriesBase;
using series = std::shared_ptr<SeriesBase>;

class SeriesBase {
    friend class DataFrame;
public:
    virtual void emplace_back(std::istream &is) = 0;
    virtual series iloc(const std::vector<std::size_t> &ib) = 0;
    virtual series iloc(const std::vector<bool> &ib) = 0;
    virtual std::size_t size() const = 0;
    virtual double mean() const = 0;
    virtual double var() const = 0;
    virtual double cov(const series &s) const = 0;
    virtual double std() const = 0;
    virtual double corr(const series &s) const = 0;
    virtual series pct_change() const = 0;
    virtual series plus(const series &s) const = 0;
    virtual series divides(double d) const = 0;
    virtual void print_value(std::ostream &os, std::size_t i) const = 0;
    virtual void print(std::ostream &os) const = 0;
private:
    virtual int width() const = 0;
};

template <typename T>
class Series : public SeriesBase {
    friend class DataFrame;
public:
    Series(std::vector<T> &&data) noexcept : values{data} {}
    void emplace_back(std::istream &is) override;
    series iloc(const std::vector<std::size_t> &ib) override;
    series iloc(const std::vector<bool> &ib) override;
    std::size_t size() const override { return std::size(values); };
    double mean() const override;
    double var() const override;
    double cov(const series &s) const override;
    double std() const override;
    double corr(const series &s) const override;
    series pct_change() const override;
    series plus(const series &s) const override;
    series divides(double d) const override;
    void print_value(std::ostream &os, std::size_t i) const override;
    void print(std::ostream &os) const override;
private:
    static std::vector<double> centerize(const Series<T> &st);
    int width() const override { return std::to_string(values[0]).length(); };
    std::vector<T> values;
};

template <typename T>
series make_series(std::vector<T> &&v) {
    return std::make_shared<Series<T>>(std::forward<std::vector<T>>(v));
};

template <typename T>
void Series<T>::emplace_back(std::istream &is) {
    T v;
    is >> std::skipws >> v >> std::skipws;
    is.get();
    values.push_back(v);
}

template <typename T>
series Series<T>::iloc(const std::vector<std::size_t> &is) {
    std::vector<T> data;
    data.reserve(std::size(is));
    for (const auto &i : is)
        data.push_back(values[i]);
    return std::make_shared<Series<T>>(std::forward<std::vector<T>>(data));
}

template <typename T>
series Series<T>::iloc(const std::vector<bool> &ib) {
    assert(std::size(ib) == std::size(values));
    std::vector<T> data;
    data.reserve(std::reduce(
        std::execution::par_unseq,
        std::begin(ib),
        std::end(ib),
        0
    ));
    for (auto [iti, itv] = std::pair(std::begin(ib), std::begin(values)); iti != std::end(ib); ++iti, ++itv) {
        if (*iti) {
            data.push_back(*itv);
        }
    }
    return std::make_shared<Series<T>>(std::forward<std::vector<T>>(data));
}

template <typename T>
double Series<T>::mean() const {
    return std::reduce(
        std::execution::par_unseq,
        std::begin(values),
        std::end(values),
        0.
    ) / std::size(values);
}

template <typename T>
std::vector<double> Series<T>::centerize(const Series<T> &st) {
    double m = st.mean();
    std::vector<double> centered(std::size(st));
    std::transform(
        std::execution::par_unseq,
        std::begin(st.values),
        std::end(st.values),
        std::begin(centered),
        [m](const double v) { return v - m; }
    );
    return centered;
}

template <typename T>
double Series<T>::var() const {
    auto centered_this = centerize(*this);
    return std::transform_reduce(
        std::execution::par_unseq,
        std::begin(centered_this),
        std::end(centered_this),
        std::begin(centered_this),
        0.0
    ) / (std::size(values) - 1);
}

template <typename T>
double Series<T>::cov(const series &s) const {
    auto t = std::dynamic_pointer_cast<Series<T>>(s);
    auto centered_this = centerize(*this);
    auto centered_that = centerize(*t);
    return std::transform_reduce(
        std::execution::par_unseq,
        std::begin(centered_this),
        std::end(centered_this),
        std::begin(centered_that),
        0.0
    ) / (std::size(values) - 1);
}

template <typename T>
double Series<T>::std() const {
    return sqrt(var());
}

template <typename T>
double Series<T>::corr(const series &s) const {
    return cov(s) / (std() * s->std());
}

template <typename T>
series Series<T>::pct_change() const {
    std::vector<double> data(std::size(values) - 1);
    std::transform(
        std::execution::par_unseq,
        std::begin(values) + 1,
        std::end(values),
        std::begin(values),
        std::begin(data),
        [](const T &u, const T &v) { return u / v - 1; }
    );
    return make_series(std::move(data));
}

template <typename T>
series Series<T>::plus(const series &s) const {
    assert(std::size(values) == s->size());
    auto t = std::dynamic_pointer_cast<Series<T>>(s);
    std::vector<T> data(std::size(values));
    std::transform(
        std::execution::par_unseq,
        std::begin(values),
        std::end(values),
        std::begin(t->values),
        std::begin(data),
        [](const T &u, const T &v) { return u + v; }
    );
    return make_series(std::move(data));
}

template <typename T>
series Series<T>::divides(double d) const {
    std::vector<T> data(std::size(values));
    std::transform(
        std::execution::par_unseq,
        std::begin(values),
        std::end(values),
        std::begin(data),
        [d](const T &v) { return v / d; }
    );
    return make_series(std::move(data));
}

template <typename T>
void Series<T>::print_value(std::ostream &os, std::size_t i) const {
    os.precision(3);
    os.setf(std::ios::fixed, std::ios::floatfield);
    os << values[i];
}

template <typename T>
void Series<T>::print(std::ostream &os) const {
    if (std::size(values) > display_max_rows) {
        auto row_num = display_min_rows / 2;
        for (std::size_t i = 0; i < row_num; ++i) {
            print_value(os, i);
            os << std::endl;
        }
        os << "..." << std::endl;
        for (std::size_t i = std::size(values) - row_num; i < std::size(values); ++i) {
            print_value(os, i);
            os << std::endl;
        }
    } else {
        for (std::size_t i = 0; i < std::size(values); ++i) {
            print_value(os, i);
            os << std::endl;
        }
    }
    os << "Length: " << std::size(values);
}

}

pandas::series operator+(const pandas::series &s, const pandas::series &t) {
    return s->plus(t);
}

pandas::series operator/(const pandas::series &s, double d) {
    return s->divides(d);
}

std::ostream& operator<<(std::ostream &os, const pandas::series &s) {
    s->print(os);
    return os;
}
