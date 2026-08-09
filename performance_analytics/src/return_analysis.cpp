#include "performance_analytics/return_analysis.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iomanip>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <tuple>

#include "performance_analytics/return_ledger.h"

namespace performance_analytics {
namespace {
constexpr std::uint64_t kOffset = 1469598103934665603ULL;
constexpr std::uint64_t kPrime = 1099511628211ULL;
void hash_bytes(std::uint64_t &h, std::uint64_t v) {
  for (int s = 0; s < 64; s += 8) {
    h ^= (v >> s) & 0xffU;
    h *= kPrime;
  }
}
void hash_double(std::uint64_t &h, double v) {
  hash_bytes(h, std::bit_cast<std::uint64_t>(v == 0.0 ? 0.0 : v));
}
void hash_string(std::uint64_t &h, const std::string &v) {
  for (unsigned char c : v) {
    h ^= c;
    h *= kPrime;
  }
  hash_bytes(h, v.size());
}
bool finite(std::span<const double> x) {
  return std::all_of(x.begin(), x.end(),
                     [](double v) { return std::isfinite(v); });
}
double avg(std::span<const double> x) {
  return std::accumulate(x.begin(), x.end(), 0.0) /
         static_cast<double>(x.size());
}
double variance(std::span<const double> x, double m) {
  if (x.size() < 2)
    return 0.0;
  double s = 0.0;
  for (double v : x)
    s += (v - m) * (v - m);
  return s / static_cast<double>(x.size() - 1);
}
double qtile(std::vector<double> x, double p) {
  std::sort(x.begin(), x.end());
  if (x.empty())
    return std::numeric_limits<double>::quiet_NaN();
  double at = p * static_cast<double>(x.size() - 1);
  auto lo = static_cast<std::size_t>(std::floor(at));
  auto hi = static_cast<std::size_t>(std::ceil(at));
  return x[lo] + (x[hi] - x[lo]) * (at - lo);
}
std::uint64_t splitmix(std::uint64_t &s) {
  s += 0x9e3779b97f4a7c15ULL;
  std::uint64_t z = s;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
double uniform(std::uint64_t &s) {
  return static_cast<double>(splitmix(s) >> 11) /
         static_cast<double>(1ULL << 53);
}
std::uint32_t rotate_right(std::uint32_t value, unsigned count) {
  return (value >> count) | (value << (32U - count));
}
std::string digest(std::string_view value) {
  constexpr std::array<std::uint32_t, 64> constants = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
      0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
      0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
      0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
      0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
      0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
      0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  std::vector<std::uint8_t> bytes(value.begin(), value.end());
  const std::uint64_t bits = static_cast<std::uint64_t>(bytes.size()) * 8ULL;
  bytes.push_back(0x80U);
  while (bytes.size() % 64U != 56U)
    bytes.push_back(0U);
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>((bits >> shift) & 0xffU));
  std::array<std::uint32_t, 8> state = {0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                        0xa54ff53a, 0x510e527f, 0x9b05688c,
                                        0x1f83d9ab, 0x5be0cd19};
  for (std::size_t offset = 0; offset < bytes.size(); offset += 64U) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t i = 0; i < 16; ++i) {
      auto at = offset + 4 * i;
      words[i] = (static_cast<std::uint32_t>(bytes[at]) << 24U) |
                 (static_cast<std::uint32_t>(bytes[at + 1]) << 16U) |
                 (static_cast<std::uint32_t>(bytes[at + 2]) << 8U) |
                 bytes[at + 3];
    }
    for (std::size_t i = 16; i < 64; ++i) {
      auto a = rotate_right(words[i - 15], 7) ^
               rotate_right(words[i - 15], 18) ^ (words[i - 15] >> 3U);
      auto b = rotate_right(words[i - 2], 17) ^ rotate_right(words[i - 2], 19) ^
               (words[i - 2] >> 10U);
      words[i] = words[i - 16] + a + words[i - 7] + b;
    }
    auto a = state[0], b = state[1], c = state[2], d = state[3], e = state[4],
         f = state[5], g = state[6], h = state[7];
    for (std::size_t i = 0; i < 64; ++i) {
      auto s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
      auto choose = (e & f) ^ ((~e) & g);
      auto t1 = h + s1 + choose + constants[i] + words[i];
      auto s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
      auto majority = (a & b) ^ (a & c) ^ (b & c);
      auto t2 = s0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (auto word : state)
    out << std::setw(8) << word;
  return out.str();
}
std::string escape(const std::string &value) {
  std::string out;
  for (char c : value) {
    if (c == '\\')
      out += "\\\\";
    else if (c == '"')
      out += "\\\"";
    else if (c == '\n')
      out += "\\n";
    else
      out += c;
  }
  return out;
}
std::string optional_number(const std::optional<double> &value) {
  if (!value || !std::isfinite(*value))
    return "null";
  std::ostringstream output;
  output << std::setprecision(17) << *value;
  return output.str();
}
std::string serialize_metrics(const ReturnMetrics &metrics) {
  std::ostringstream output;
  output << std::setprecision(17)
         << "{\"observations\":" << metrics.observations
         << ",\"status\":" << static_cast<int>(metrics.status)
         << ",\"cumulative_return\":"
         << optional_number(metrics.cumulative_return)
         << ",\"annualized_return\":"
         << optional_number(metrics.annualized_return)
         << ",\"annualized_volatility\":"
         << optional_number(metrics.annualized_volatility)
         << ",\"sharpe\":" << optional_number(metrics.sharpe)
         << ",\"sortino\":" << optional_number(metrics.sortino)
         << ",\"calmar\":" << optional_number(metrics.calmar)
         << ",\"maximum_drawdown\":"
         << optional_number(metrics.maximum_drawdown)
         << ",\"maximum_drawdown_duration\":"
         << metrics.maximum_drawdown_duration
         << ",\"win_rate\":" << optional_number(metrics.win_rate)
         << ",\"profit_factor\":"
         << optional_number(metrics.profit_factor)
         << ",\"average_win_loss_ratio\":"
         << optional_number(metrics.average_win_loss_ratio)
         << ",\"cvar\":" << optional_number(metrics.cvar)
         << ",\"artifact_hash\":" << metrics.artifact_hash << '}';
  return output.str();
}
bool digest_like(const std::string &value, bool empty = false) {
  return (empty && value.empty()) ||
         (value.size() == 64 &&
          std::all_of(value.begin(), value.end(), [](char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
          }));
}
bool valid_available_at_utc(const std::string &value) {
  if (value.size() < 20 || value.back() != 'Z' || value[4] != '-' ||
      value[7] != '-' || value[10] != 'T' || value[13] != ':' ||
      value[16] != ':')
    return false;
  const std::array<std::size_t, 14> digit_positions = {
      0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18};
  for (const auto position : digit_positions)
    if (value[position] < '0' || value[position] > '9')
      return false;
  const int year = std::stoi(value.substr(0, 4));
  const int month = std::stoi(value.substr(5, 2));
  const int day = std::stoi(value.substr(8, 2));
  const int hour = std::stoi(value.substr(11, 2));
  const int minute = std::stoi(value.substr(14, 2));
  const int second = std::stoi(value.substr(17, 2));
  if (year == 0 || month < 1 || month > 12 || hour > 23 || minute > 59 ||
      second > 59)
    return false;
  constexpr std::array<int, 12> days_in_month = {
      31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  const bool leap_year = year % 4 == 0 &&
                         (year % 100 != 0 || year % 400 == 0);
  const int maximum_day = days_in_month[static_cast<std::size_t>(month - 1)] +
                          (month == 2 && leap_year ? 1 : 0);
  if (day < 1 || day > maximum_day)
    return false;
  if (value.size() == 20)
    return true;
  if (value.size() < 22 || value[19] != '.')
    return false;
  return std::all_of(value.begin() + 20, value.end() - 1,
                     [](char c) { return c >= '0' && c <= '9'; });
}
bool reference_price_ready(const std::string &quality) {
  return !quality.empty() && quality != "UNKNOWN" && quality != "UNAVAILABLE";
}
} // namespace

PairedBootstrapResult
paired_stationary_bootstrap(std::span<const double> candidate,
                            std::span<const double> baseline,
                            const StationaryBootstrapSpec &spec, double level) {
  PairedBootstrapResult r;
  r.confidence_level = level;
  if (candidate.size() != baseline.size() || candidate.empty() ||
      !finite(candidate) || !finite(baseline) || spec.replicates == 0 ||
      spec.seed == 0 || spec.config_hash == 0 ||
      !(spec.mean_block_length >= 1.0) ||
      !std::isfinite(spec.mean_block_length) || !(level > 0.0 && level < 1.0))
    return r;
  std::vector<double> diff(candidate.size());
  for (std::size_t i = 0; i < diff.size(); ++i)
    diff[i] = candidate[i] - baseline[i];
  r.observed_difference = avg(diff);
  std::vector<double> boot;
  boot.reserve(spec.replicates);
  std::uint64_t state = spec.seed;
  std::size_t extreme = 0;
  for (std::uint32_t n = 0; n < spec.replicates; ++n) {
    std::size_t at = splitmix(state) % diff.size();
    double sum = 0.0;
    for (std::size_t i = 0; i < diff.size(); ++i) {
      sum += diff[at];
      if (i + 1 < diff.size())
        at = uniform(state) < 1.0 / spec.mean_block_length
                 ? splitmix(state) % diff.size()
                 : (at + 1) % diff.size();
    }
    double value = sum / diff.size();
    boot.push_back(value);
    if (std::abs(value) >= std::abs(r.observed_difference))
      ++extreme;
  }
  r.bootstrap_mean = avg(boot);
  r.p_value = static_cast<double>(extreme + 1) / (spec.replicates + 1.0);
  r.confidence_lower = qtile(boot, (1.0 - level) / 2.0);
  r.confidence_upper = qtile(boot, 1.0 - (1.0 - level) / 2.0);
  r.artifact_hash = kOffset;
  hash_bytes(r.artifact_hash, spec.config_hash);
  hash_bytes(r.artifact_hash, spec.seed);
  for (double v : boot)
    hash_double(r.artifact_hash, v);
  r.status = AnalysisStatus::OK;
  return r;
}
std::vector<StationaryBootstrapSensitivity>
paired_stationary_bootstrap_sensitivity(std::span<const double> c,
                                        std::span<const double> b,
                                        const StationaryBootstrapSpec &base,
                                        std::span<const double> lengths,
                                        double level) {
  std::vector<StationaryBootstrapSensitivity> out;
  for (double length : lengths) {
    auto spec = base;
    spec.mean_block_length = length;
    out.push_back({length, paired_stationary_bootstrap(c, b, spec, level)});
  }
  return out;
}

HacResult newey_west_hac(std::span<const double> x, const HacSpec &spec) {
  HacResult r;
  r.lag = spec.lag;
  if (x.size() < 2 || spec.config_hash == 0 || !finite(x))
    return r;
  r.mean = avg(x);
  double v = variance(x, r.mean);
  std::size_t lag = std::min<std::size_t>(spec.lag, x.size() - 1);
  for (std::size_t k = 1; k <= lag; ++k) {
    double cov = 0.0;
    for (std::size_t i = k; i < x.size(); ++i)
      cov += (x[i] - r.mean) * (x[i - k] - r.mean);
    v += 2.0 * (1.0 - static_cast<double>(k) / (lag + 1.0)) * cov /
         static_cast<double>(x.size());
  }
  if (v < 0.0 && v > -1e-14)
    v = 0.0;
  if (v < 0.0 || !std::isfinite(v))
    return r;
  r.long_run_variance = v;
  r.standard_error = std::sqrt(v / x.size());
  r.t_statistic = r.standard_error > 0.0 ? r.mean / r.standard_error : 0.0;
  r.artifact_hash = kOffset;
  hash_bytes(r.artifact_hash, spec.config_hash);
  r.status = r.standard_error > 0.0 ? AnalysisStatus::OK
                                    : AnalysisStatus::ZERO_VARIANCE;
  return r;
}

EffectiveTrialResult
estimate_effective_trial_count(std::span<const double> flat, std::size_t trials,
                               std::size_t observations,
                               const EffectiveTrialSpec &spec) {
  EffectiveTrialResult r;
  r.trial_count = trials;
  r.observations = observations;
  if (trials == 0 || observations < spec.minimum_observations ||
      spec.config_hash == 0 || flat.size() != trials * observations ||
      !finite(flat))
    return r;
  std::vector<double> means(trials), sd(trials);
  for (std::size_t t = 0; t < trials; ++t) {
    auto row = flat.subspan(t * observations, observations);
    means[t] = avg(row);
    sd[t] = std::sqrt(variance(row, means[t]));
    if (!(sd[t] > 0.0)) {
      r.status = AnalysisStatus::ZERO_VARIANCE;
      return r;
    }
  }
  double sum = 0.0;
  std::size_t pairs = 0;
  for (std::size_t a = 0; a < trials; ++a)
    for (std::size_t b = a + 1; b < trials; ++b) {
      double cov = 0.0;
      for (std::size_t i = 0; i < observations; ++i)
        cov += (flat[a * observations + i] - means[a]) *
               (flat[b * observations + i] - means[b]);
      sum += std::abs(
          std::clamp(cov / ((observations - 1.0) * sd[a] * sd[b]), -1.0, 1.0));
      ++pairs;
    }
  r.mean_absolute_correlation = pairs ? sum / pairs : 0.0;
  r.effective_trials =
      std::clamp(static_cast<double>(trials) /
                     (1.0 + (trials - 1.0) * r.mean_absolute_correlation),
                 1.0, static_cast<double>(trials));
  r.artifact_hash = kOffset;
  hash_bytes(r.artifact_hash, spec.config_hash);
  hash_double(r.artifact_hash, r.effective_trials);
  r.status = AnalysisStatus::OK;
  return r;
}

DsrResult compute_deflated_sharpe_ratio(std::span<const double> x,
                                        const PerformanceSpecV1 &spec,
                                        double effective,
                                        std::uint64_t config_hash,
                                        double benchmark) {
  DsrResult r;
  r.observations = x.size();
  r.effective_trials = effective;
  r.benchmark_sharpe = benchmark;
  if (!valid_performance_spec(spec) || config_hash == 0 ||
      x.size() < std::max<std::size_t>(3, spec.minimum_return_observations) ||
      !finite(x) || effective < 1.0 || !std::isfinite(effective) ||
      !std::all_of(x.begin(), x.end(), [](double v) { return v > -1.0; }))
    return r;
  const double m = avg(x), sd = std::sqrt(variance(x, m));
  if (!(sd > 0.0)) {
    r.status = AnalysisStatus::ZERO_VARIANCE;
    return r;
  }
  const double sr = m / sd, scale = std::sqrt(spec.calendar_periods_per_year);
  r.raw_sharpe = sr * scale;
  double m3 = 0.0, m4 = 0.0;
  for (double v : x) {
    double z = (v - m) / sd;
    m3 += z * z * z;
    m4 += z * z * z * z;
  }
  r.skewness = m3 / x.size();
  r.excess_kurtosis = m4 / x.size() - 3.0;
  const double threshold = effective > 1.0
                               ? std::sqrt(2.0 * std::log(effective) / x.size())
                               : benchmark / scale;
  r.expected_max_sharpe = threshold * scale;
  const double correction =
      1.0 - r.skewness * sr + 0.25 * r.excess_kurtosis * sr * sr;
  if (!(correction > 0.0)) {
    r.status = AnalysisStatus::NUMERICAL_FAILURE;
    return r;
  }
  r.deflated_sharpe =
      (sr - threshold) / std::sqrt(correction / (x.size() - 1.0));
  r.one_sided_p_value = 0.5 * std::erfc(r.deflated_sharpe / std::sqrt(2.0));
  r.artifact_hash = kOffset;
  hash_bytes(r.artifact_hash, config_hash);
  hash_double(r.artifact_hash, r.deflated_sharpe);
  r.status = AnalysisStatus::OK;
  return r;
}

ReturnMetrics compute_return_metrics(std::span<const double> x,
                                     const PerformanceSpecV1 &spec) {
  ReturnMetrics r;
  r.observations = x.size();
  if (!valid_performance_spec(spec) ||
      x.size() < spec.minimum_return_observations || !finite(x))
    return r;
  double wealth = 1.0, peak = 1.0, losses = 0.0, wins = 0.0, win_count = 0.0,
         loss_count = 0.0, max_dd = 0.0;
  std::uint32_t duration = 0;
  std::vector<double> downside, sorted(x.begin(), x.end());
  for (double v : x) {
    if (v <= -1.0)
      return r;
    wealth *= 1.0 + v;
    peak = std::max(peak, wealth);
    double dd = wealth / peak - 1.0;
    max_dd = std::min(max_dd, dd);
    duration = dd < 0.0 ? duration + 1 : 0;
    r.maximum_drawdown_duration =
        std::max(r.maximum_drawdown_duration, duration);
    if (v > 0.0) {
      wins += v;
      ++win_count;
    } else if (v < 0.0) {
      losses -= v;
      ++loss_count;
    }
    double e = v - spec.annual_risk_free_rate / spec.calendar_periods_per_year;
    if (e < 0.0)
      downside.push_back(e * e);
  }
  r.cumulative_return = wealth - 1.0;
  r.annualized_return =
      std::pow(wealth, spec.calendar_periods_per_year / x.size()) - 1.0;
  double m = avg(x), v = variance(x, m),
         scale = std::sqrt(spec.calendar_periods_per_year),
         rf = spec.annual_risk_free_rate / spec.calendar_periods_per_year;
  r.annualized_volatility = std::sqrt(std::max(0.0, v)) * scale;
  r.sharpe =
      r.annualized_volatility > 0.0
          ? (m - rf) * spec.calendar_periods_per_year / r.annualized_volatility
          : 0.0;
  double down =
      downside.empty()
          ? 0.0
          : std::sqrt(std::accumulate(downside.begin(), downside.end(), 0.0) /
                      x.size()) *
                scale;
  r.sortino =
      down > 0.0 ? (m - rf) * spec.calendar_periods_per_year / down : 0.0;
  r.maximum_drawdown = max_dd;
  r.calmar = max_dd < 0.0 ? r.annualized_return / -max_dd : 0.0;
  r.win_rate = win_count / x.size();
  r.profit_factor = losses > 0.0 ? wins / losses : 0.0;
  r.average_win_loss_ratio = win_count > 0.0 && loss_count > 0.0
                                 ? (wins / win_count) / (losses / loss_count)
                                 : 0.0;
  if (x.size() >= spec.minimum_tail_observations) {
    std::sort(sorted.begin(), sorted.end());
    auto n = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil(sorted.size() * 0.05)));
    r.cvar = std::accumulate(sorted.begin(), sorted.begin() + n, 0.0) / n;
  }
  r.artifact_hash = kOffset;
  hash_bytes(r.artifact_hash, spec.config_hash);
  r.status = v > 0.0 ? AnalysisStatus::OK : AnalysisStatus::ZERO_VARIANCE;
  return r;
}

ActiveMetrics compute_active_metrics(std::span<const double> p,
                                     std::span<const double> b,
                                     const PerformanceSpecV1 &spec) {
  ActiveMetrics r;
  if (b.empty()) {
    r.status = AnalysisStatus::UNAVAILABLE;
    return r;
  }
  if (!valid_performance_spec(spec) || p.size() != b.size() ||
      p.size() < spec.minimum_return_observations || !finite(p) || !finite(b))
    return r;
  double pg = 1.0, bg = 1.0;
  std::vector<double> a(p.size());
  for (std::size_t i = 0; i < p.size(); ++i) {
    if (p[i] <= -1.0 || b[i] <= -1.0)
      return r;
    pg *= 1.0 + p[i];
    bg *= 1.0 + b[i];
    a[i] = p[i] - b[i];
  }
  r.active_growth = pg / bg - 1.0;
  r.mean_active_return = avg(a);
  r.tracking_error =
      std::sqrt(std::max(0.0, variance(a, r.mean_active_return))) *
      std::sqrt(spec.calendar_periods_per_year);
  r.information_ratio = r.tracking_error > 0.0
                            ? r.mean_active_return *
                                  spec.calendar_periods_per_year /
                                  r.tracking_error
                            : 0.0;
  double pm = avg(p), bm = avg(b), bv = variance(b, bm), cov = 0.0;
  for (std::size_t i = 0; i < p.size(); ++i)
    cov += (p[i] - pm) * (b[i] - bm);
  cov /= p.size() - 1.0;
  r.capm_beta = bv > 0.0 ? cov / bv : 0.0;
  r.capm_alpha =
      ((pm - spec.annual_risk_free_rate / spec.calendar_periods_per_year) -
       r.capm_beta *
           (bm - spec.annual_risk_free_rate / spec.calendar_periods_per_year)) *
      spec.calendar_periods_per_year;
  double up_p = 1.0, up_b = 1.0, down_p = 1.0, down_b = 1.0;
  for (std::size_t i = 0; i < p.size(); ++i) {
    if (b[i] > 0.0) {
      up_p *= 1.0 + p[i];
      up_b *= 1.0 + b[i];
    }
    if (b[i] < 0.0) {
      down_p *= 1.0 + p[i];
      down_b *= 1.0 + b[i];
    }
  }
  r.up_capture = up_b > 0.0 ? up_p / up_b : 0.0;
  r.down_capture = down_b > 0.0 ? down_p / down_b : 0.0;
  r.status = bv > 0.0 ? AnalysisStatus::OK : AnalysisStatus::ZERO_VARIANCE;
  return r;
}
ActiveMetrics
compute_aligned_active_metrics(std::span<const AlignedReturnPair> rows,
                               const PerformanceSpecV1 &spec) {
  ActiveMetrics r;
  if (rows.empty()) {
    r.status = AnalysisStatus::UNAVAILABLE;
    return r;
  }
  std::vector<double> p, b;
  std::uint64_t previous = 0;
  for (const auto &row : rows) {
    if (row.period_id == 0 || row.period_id <= previous)
      return r;
    previous = row.period_id;
    p.push_back(row.portfolio_return);
    b.push_back(row.benchmark_return);
  }
  return compute_active_metrics(p, b, spec);
}

ScoreBucketResult analyze_score_buckets(std::span<const ScoreObservation> rows,
                                        const ScoreBucketSpec &spec) {
  ScoreBucketResult result;
  if (rows.empty() || spec.bucket_count < 2 || spec.minimum_observations == 0 ||
      spec.config_hash == 0)
    return result;
  std::map<std::uint64_t, std::vector<ScoreObservation>> dates;
  for (const auto &row : rows) {
    if (!row.date_key || !row.symbol_id || !row.horizon ||
        !std::isfinite(row.score) || !std::isfinite(row.realized_return) ||
        row.realized_return <= -1.0)
      return result;
    dates[row.date_key].push_back(row);
  }
  result.buckets.resize(spec.bucket_count);
  std::vector<std::vector<double>> bucket_returns(spec.bucket_count);
  std::map<std::uint32_t, std::vector<double>> horizon_returns;
  std::vector<double> daily_ic;
  std::set<std::uint64_t> symbols, previous_top;
  double turnover = 0.0;
  std::size_t turnover_count = 0;
  for (auto &[date, group] : dates) {
    (void)date;
    if (group.size() < spec.minimum_observations)
      continue;
    std::sort(group.begin(), group.end(),
              [](const auto &left, const auto &right) {
                return std::tie(left.score, left.symbol_id) <
                       std::tie(right.score, right.symbol_id);
              });
    double score_mean = 0.0, return_mean = 0.0;
    for (const auto &row : group) {
      score_mean += row.score;
      return_mean += row.realized_return;
    }
    score_mean /= group.size();
    return_mean /= group.size();
    double covariance = 0.0, score_variance = 0.0, return_variance = 0.0;
    std::vector<bool> used(spec.bucket_count, false);
    std::set<std::uint64_t> top;
    for (std::size_t rank = 0; rank < group.size(); ++rank) {
      const std::size_t bucket = rank * spec.bucket_count / group.size();
      result.buckets[bucket].bucket = static_cast<std::uint32_t>(bucket);
      ++result.buckets[bucket].observations;
      used[bucket] = true;
      bucket_returns[bucket].push_back(group[rank].realized_return);
      horizon_returns[group[rank].horizon].push_back(
          group[rank].realized_return);
      symbols.insert(group[rank].symbol_id);
      if (bucket + 1 == spec.bucket_count)
        top.insert(group[rank].symbol_id);
      const double dx = group[rank].score - score_mean;
      const double dy = group[rank].realized_return - return_mean;
      covariance += dx * dy;
      score_variance += dx * dx;
      return_variance += dy * dy;
    }
    for (std::size_t bucket = 0; bucket < used.size(); ++bucket)
      if (used[bucket])
        ++result.buckets[bucket].dates;
    if (score_variance > 0.0 && return_variance > 0.0)
      daily_ic.push_back(covariance /
                         std::sqrt(score_variance * return_variance));
    if (!previous_top.empty() && !top.empty()) {
      std::size_t changed = 0;
      for (auto id : previous_top)
        if (!top.contains(id))
          ++changed;
      turnover += static_cast<double>(changed) / previous_top.size();
      ++turnover_count;
    }
    previous_top = std::move(top);
  }
  std::vector<double> means;
  for (std::size_t bucket = 0; bucket < bucket_returns.size(); ++bucket) {
    if (!bucket_returns[bucket].empty()) {
      result.buckets[bucket].mean_return = avg(bucket_returns[bucket]);
      means.push_back(result.buckets[bucket].mean_return);
    }
  }
  if (means.size() < 2)
    return result;
  result.top_bottom_spread =
      result.buckets.back().mean_return - result.buckets.front().mean_return;
  const double x_mean = (spec.bucket_count - 1.0) / 2.0;
  double numerator = 0.0, denominator = 0.0;
  const double mean_return = avg(means);
  for (std::size_t bucket = 0; bucket < result.buckets.size(); ++bucket) {
    if (!std::isfinite(result.buckets[bucket].mean_return))
      continue;
    const double x = static_cast<double>(bucket) - x_mean;
    numerator += x * (result.buckets[bucket].mean_return - mean_return);
    denominator += x * x;
  }
  result.monotonicity_slope = denominator > 0.0 ? numerator / denominator : 0.0;
  result.covered_symbols = symbols.size();
  result.top_bucket_turnover = turnover_count ? turnover / turnover_count : 0.0;
  if (daily_ic.size() >= 2) {
    const double ic_mean = avg(daily_ic),
                 ic_variance = variance(daily_ic, ic_mean);
    result.icir = ic_variance > 0.0 ? ic_mean / std::sqrt(ic_variance) : 0.0;
  }
  for (const auto &[horizon, values] : horizon_returns) {
    (void)horizon;
    result.horizon_decay.push_back(avg(values));
  }
  result.artifact_hash = kOffset;
  hash_bytes(result.artifact_hash, spec.config_hash);
  result.status = AnalysisStatus::OK;
  return result;
}

std::string
serialize_return_analysis_manifest(const ReturnAnalysisManifest &m) {
  std::ostringstream out;
  out << std::setprecision(17);
  const bool proxy = m.reference_price_quality == "PROXY" ||
                     m.reference_price_quality == "ARRIVAL_PROXY";
  out << "{\"schema_version\":" << m.schema_version
      << ",\"metric_spec_version\":\"" << escape(m.metric_spec_version)
      << "\",\"source_replay_sha256\":\"" << escape(m.source_replay_sha256)
      << "\",\"dataset_fingerprint\":\"" << escape(m.dataset_fingerprint)
      << "\",\"portfolio_return_ledger_sha256\":\""
      << escape(m.portfolio_return_ledger_sha256)
      << "\",\"benchmark_artifact_sha256\":\""
      << escape(m.benchmark_artifact_sha256)
      << "\",\"implementation_shortfall_ledger_sha256\":\""
      << escape(m.implementation_shortfall_ledger_sha256)
      << "\",\"stationary_bootstrap_spec_hash\":\""
      << escape(m.stationary_bootstrap_spec_hash)
      << "\",\"hac_sensitivity_spec_hash\":\""
      << escape(m.hac_sensitivity_spec_hash) << "\",\"benchmark_id\":\""
      << escape(m.benchmark_id) << "\",\"calendar_id\":\""
      << escape(m.calendar_id) << "\",\"benchmark_available\":"
      << (m.benchmark_available ? "true" : "false")
      << ",\"promotion_eligible\":"
      << (m.promotion_eligible && m.benchmark_available &&
                  reference_price_ready(m.reference_price_quality) && !proxy
              ? "true"
              : "false")
      << ",\"reference_price_quality\":\"" << escape(m.reference_price_quality)
      << "\",\"raw_trials\":" << m.raw_trials;
  auto optional = [&out](const char *key, const std::optional<double> &value) {
    out << ",\"" << key << "\":";
    if (value && std::isfinite(*value))
      out << *value;
    else
      out << "null";
  };
  optional("effective_trials", m.effective_trials);
  optional("deflated_sharpe", m.deflated_sharpe);
  optional("execution_price_cost", m.execution_price_cost);
  optional("explicit_fees", m.explicit_fees);
  optional("opportunity_cost", m.opportunity_cost);
  optional("implementation_shortfall", m.implementation_shortfall);
  optional("gross_pnl", m.gross_pnl);
  optional("net_pnl", m.net_pnl);
  optional("accounting_residual", m.accounting_residual);
  optional("var_loss", m.var_loss);
  optional("expected_shortfall_loss", m.expected_shortfall_loss);
  optional("return_cvar", m.return_cvar);
  out << ",\"limitations\":[";
  for (std::size_t i = 0; i < m.limitations.size(); ++i) {
    if (i)
      out << ',';
    out << '"' << escape(m.limitations[i]) << '"';
  }
  out << "]}";
  return out.str();
}
std::uint64_t hash_return_analysis_manifest(const ReturnAnalysisManifest &m) {
  std::uint64_t h = kOffset;
  for (unsigned char c : serialize_return_analysis_manifest(m)) {
    h ^= c;
    h *= kPrime;
  }
  return h;
}
std::string sha256_return_analysis_manifest(const ReturnAnalysisManifest &m) {
  return digest(serialize_return_analysis_manifest(m));
}
std::string sha256_text(std::string_view value) { return digest(value); }

ReturnAnalysisReport build_return_analysis_report(
    const ReturnLedger &ledger, ReturnAnalysisManifest manifest) {
  ReturnAnalysisReport report;
  report.ledger_hash = ledger.ledger_hash();
  std::vector<double> period_returns;
  period_returns.reserve(ledger.records().size());
  double gross_pnl = 0.0;
  double net_pnl = 0.0;
  double accounting_residual = 0.0;
  for (const auto &record : ledger.records()) {
    period_returns.push_back(record.period_return);
    gross_pnl += record.executed_gross_pnl;
    net_pnl += record.net_pnl;
    accounting_residual += record.accounting_residual;
  }
  report.metrics = compute_return_metrics(period_returns, ledger.spec());
  report.status = report.metrics.status;
  if (manifest.portfolio_return_ledger_sha256.empty())
    manifest.portfolio_return_ledger_sha256 = ledger.ledger_sha256();
  if (manifest.calendar_id.empty())
    manifest.calendar_id = ledger.spec().calendar_id;
  if (manifest.benchmark_id.empty())
    manifest.benchmark_id = ledger.spec().benchmark_id;
  if (!manifest.gross_pnl)
    manifest.gross_pnl = gross_pnl;
  if (!manifest.net_pnl)
    manifest.net_pnl = net_pnl;
  if (!manifest.accounting_residual)
    manifest.accounting_residual = accounting_residual;
  report.manifest_json = serialize_return_analysis_manifest(manifest);
  report.manifest_hash = hash_return_analysis_manifest(manifest);
  const std::string unsigned_artifact =
      "{\"schema_version\":1,\"role\":\"return_analysis_v1\",\"manifest\":" +
      report.manifest_json + ",\"metrics\":" +
      serialize_metrics(report.metrics) + ",\"ledger_hash\":" +
      std::to_string(report.ledger_hash) + ",\"manifest_hash\":" +
      std::to_string(report.manifest_hash) + '}';
  report.report_sha256 = sha256_text(unsigned_artifact);
  report.artifact_json = unsigned_artifact.substr(0, unsigned_artifact.size() - 1) +
      ",\"report_sha256\":\"" + report.report_sha256 + "\"}";
  return report;
}

namespace {
bool valid_drift_snapshot_payload(const DriftSnapshotContractV0 &c) noexcept {
  return c.schema_version == 0 && digest_like(c.model_manifest_sha256) &&
         digest_like(c.raw_schema_hash) &&
         digest_like(c.preprocessing_spec_sha256) &&
         digest_like(c.feature_schema_hash) &&
         digest_like(c.prediction_schema_hash) &&
         digest_like(c.raw_fields_sha256) &&
         digest_like(c.preprocessed_features_sha256) &&
         digest_like(c.prediction_values_sha256) &&
         digest_like(c.embedding_values_sha256) &&
         digest_like(c.matured_labels_sha256, !c.labels_mature) &&
         digest_like(c.source_snapshot_set_sha256) &&
         digest_like(c.ledger_schema_hash) && !c.available_at_utc.empty() &&
         valid_available_at_utc(c.available_at_utc) &&
         digest_like(c.label_spec_sha256, !c.labels_mature);
}
} // namespace
bool valid_drift_snapshot_contract(const DriftSnapshotContractV0 &c) noexcept {
  return valid_drift_snapshot_payload(c) && digest_like(c.report_sha256);
}
std::uint64_t hash_drift_snapshot_contract(const DriftSnapshotContractV0 &c) {
  std::uint64_t h = kOffset;
  hash_bytes(h, c.schema_version);
  for (const auto &v :
       {c.model_manifest_sha256, c.raw_schema_hash, c.preprocessing_spec_sha256,
        c.feature_schema_hash, c.prediction_schema_hash, c.label_spec_sha256,
        c.raw_fields_sha256, c.preprocessed_features_sha256,
        c.prediction_values_sha256, c.embedding_values_sha256,
        c.matured_labels_sha256, c.source_snapshot_set_sha256,
        c.ledger_schema_hash, c.available_at_utc, c.report_sha256})
    hash_string(h, v);
  hash_bytes(h, c.labels_mature);
  return h;
}

std::string
serialize_drift_snapshot_artifact(const DriftSnapshotContractV0 &c) {
  if (!valid_drift_snapshot_payload(c))
    return {};
  std::ostringstream output;
  output << "{\"schema_version\":0,\"role\":\"drift_snapshot_v0\""
         << ",\"labels_mature\":"
         << (c.labels_mature ? "true" : "false")
         << ",\"model_manifest_sha256\":\""
         << escape(c.model_manifest_sha256)
         << "\",\"raw_schema_hash\":\"" << escape(c.raw_schema_hash)
         << "\",\"preprocessing_spec_sha256\":\""
         << escape(c.preprocessing_spec_sha256)
         << "\",\"feature_schema_hash\":\""
         << escape(c.feature_schema_hash)
         << "\",\"prediction_schema_hash\":\""
         << escape(c.prediction_schema_hash)
         << "\",\"label_spec_sha256\":\""
         << escape(c.label_spec_sha256)
         << "\",\"raw_fields_sha256\":\"" << escape(c.raw_fields_sha256)
         << "\",\"preprocessed_features_sha256\":\""
         << escape(c.preprocessed_features_sha256)
         << "\",\"prediction_values_sha256\":\""
         << escape(c.prediction_values_sha256)
         << "\",\"embedding_values_sha256\":\""
         << escape(c.embedding_values_sha256)
         << "\",\"matured_labels_sha256\":\""
         << escape(c.matured_labels_sha256)
         << "\",\"source_snapshot_set_sha256\":\""
         << escape(c.source_snapshot_set_sha256)
         << "\",\"ledger_schema_hash\":\""
         << escape(c.ledger_schema_hash)
         << "\",\"available_at_utc\":\"" << escape(c.available_at_utc)
         << "\"}";
  const std::string unsigned_artifact = output.str();
  const std::string report_sha256 = sha256_text(unsigned_artifact);
  if (!c.report_sha256.empty() && c.report_sha256 != report_sha256)
    return {};
  return unsigned_artifact.substr(0, unsigned_artifact.size() - 1) +
         ",\"report_sha256\":\"" + report_sha256 + "\"}";
}
} // namespace performance_analytics
