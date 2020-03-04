#include <boost/timer/timer.hpp>
#include <chrono>
#include "pandas.h"

int main() {
    namespace pd = pandas;
    using namespace std::chrono;
    using namespace std::literals::chrono_literals;

    constexpr auto market_open  = 1'518'186'600'000'000L; // 2018-02-09 09:30
    constexpr auto market_close = 1'518'210'000'000'000L; // 2018-02-09 16:00
    constexpr auto delta = duration_cast<microseconds>( 5s ).count();

    // only supports numeric columns
    auto df = pd::read_csv<long, int, double, int, double, int>("quote.csv");
    df.set_index("recv_time");
    df = df[{"security_id", "bid_price", "ask_price"}];
    df = df.loc(market_open, market_close);

    std::vector<int> top_security_ids{3873, 3692, 1727, 3841, 1750};
    std::vector<pd::series> data;
    pd::series index;
    std::vector<std::string> columns;
    for (
        boost::timer::auto_cpu_timer t{3, "resampling mid price took %w seconds\n"};
        const auto &security_id : top_security_ids
    ) {
        auto df_sec = df.filter("security_id", security_id);
        df_sec = df_sec.resample(market_open, market_close, delta);
        data.push_back((df_sec["bid_price"] + df_sec["ask_price"]) / 2);
        index = df_sec.get_index();
        columns.push_back(std::to_string(security_id));
    }
    auto df_mid = pd::DataFrame(std::move(data), std::move(columns), std::move(index));
    std::cout << "mid price =" << std::endl << df_mid << std::endl;
    {
        boost::timer::auto_cpu_timer t{3, "calculating corr took %w seconds\n"};
        auto df_corr = df_mid.pct_change().corr();
        df_corr.set_index(pd::make_series(std::move(top_security_ids)));
        std::cout << "corr =" << std::endl << df_corr << std::endl;
    }

    return 0;
}
