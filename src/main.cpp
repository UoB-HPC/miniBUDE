#include <cmath>
#include <iomanip>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "bude.h"
#if __has_include("meta_build.h")
  #include "meta_build.h"
#endif
#if __has_include("meta_vcs.h")
  #include "meta_vcs.h"
#endif

#define MINIBUDE_WARN_NOT_CMAKE "unknown (built with deprecated GNU Make, please migrate to CMake)"

#ifndef MINIBUDE_VERSION
  #define MINIBUDE_VERSION MINIBUDE_WARN_NOT_CMAKE
#endif

#ifndef MINIBUDE_COMPILE_COMMANDS
  #define MINIBUDE_COMPILE_COMMANDS {MINIBUDE_WARN_NOT_CMAKE}
#endif

#ifdef USE_CPU_FEATURES
  #include "cpu_features_macros.h"
  #if defined(CPU_FEATURES_ARCH_X86)
    #include "cpuinfo_x86.h"
  #elif defined(CPU_FEATURES_ARCH_ARM)
    #include "cpuinfo_arm.h"
  #elif defined(CPU_FEATURES_ARCH_AARCH64)
    #include "cpuinfo_aarch64.h"
  #elif defined(CPU_FEATURES_ARCH_MIPS)
    #include "cpuinfo_mips.h"
  #elif defined(CPU_FEATURES_ARCH_PPC)
    #include "cpuinfo_ppc.h"
  #endif
#endif

#define DEFAULT_PPWI 1, 2, 4, 8, 16, 32, 64, 128

#ifndef USE_PPWI
  // #warning no ppwi list defined, defaulting to DEFAULT_PPWI
  #define USE_PPWI DEFAULT_PPWI
#endif

#if defined(CUDA)
  #include "cuda/fasten.hpp"
#elif defined(STD_INDICES)
  #include "std-indices/fasten.hpp"
#elif defined(STD_RANGES)
  #include "std-ranges/fasten.hpp"
#elif defined(TBB)
  #include "tbb/fasten.hpp"
#elif defined(HIP)
  #include "hip/fasten.hpp"
#elif defined(HC)
  #include "hc/fasten.hpp"
#elif defined(OCL)
  #include "ocl/fasten.hpp"
#elif defined(USE_RAJA)
  #include "raja/fasten.hpp"
#elif defined(KOKKOS)
  #include "kokkos/fasten.hpp"
#elif defined(ACC)
  #include "acc/fasten.hpp"
#elif defined(SYCL)
  #include "sycl/fasten.hpp"
#elif defined(OMP)
  #include "omp/fasten.hpp"
#elif defined(SERIAL)
  #include "serial/fasten.hpp"
#elif defined(THRUST)
  #include "thrust/fasten.hpp"
#else
  #error "No model defined"
#endif

#ifndef IMPL_CLS
  #error "Model did not define IMPL_CLS!"
#endif

template <typename V, typename... T> constexpr std::array<V, sizeof...(T)> make_array(T &&...ts) {
  return {{static_cast<V>(std::forward<T>(ts))...}};
}
constexpr static auto PPWIs = make_array<size_t>(USE_PPWI);

template <typename N> struct SummaryStats {
  N min, max, sum, mean, variance, stdDev;

  explicit SummaryStats(const std::vector<N> &ys) {
    auto minmax = std::minmax_element(ys.begin(), ys.end());
    min = *minmax.first;
    max = *minmax.second;
    sum = std::accumulate(ys.begin(), ys.end(), N(0));
    mean = sum / ys.size();
    variance =
        std::accumulate(ys.begin(), ys.end(), N(0), [&](auto acc, auto t) { return acc + std::pow(t - mean, 2.0); }) /
        ys.size();
    stdDev = std::sqrt(variance);
  }
};

template <typename T>
std::vector<T> split(const std::string &s, char delimiter, const std::function<T(const std::string &)> &f) {
  std::vector<T> xs;
  std::string token;
  std::istringstream tokenStream(s);
  while (std::getline(tokenStream, token, delimiter))
    xs.push_back(f(token));
  return xs;
}

template <typename C> std::string mk_string(const C &xs, const std::string &delim = ",") {
  std::ostringstream imploded;
  for (size_t i = 0; i < xs.size(); ++i) {
    imploded << xs[i] << (i == xs.size() - 1 ? "" : delim);
  }
  return imploded.str();
}

struct Result {
  bool valid;
  double maxDiffPct;
  Sample sample;
  SummaryStats<double> ms;
  double gflops, ginsts, interactionsPerSec;
};

template <typename T> std::vector<T> readNStruct(const std::string &path) {
  std::fstream s(path, std::ios::binary | std::ios::in);
  if (!s.good()) {
    throw std::invalid_argument("Bad file: " + path);
  }
  s.ignore(std::numeric_limits<std::streamsize>::max());
  auto len = s.gcount();
  s.clear();
  s.seekg(0, std::ios::beg);
  std::vector<T> xs(len / sizeof(T));
  s.read(reinterpret_cast<char *>(xs.data()), len);
  s.close();
  return xs;
}

[[nodiscard]] std::tuple<Params, std::vector<size_t>, std::vector<size_t>>
parseParams(const std::vector<std::string> &args) {

  Params params = {};

  // Defaults
  params.iterations = DEFAULT_ITERS;
  params.warmupIterations = 2;
  params.deckDir = DATA_DIR;
  params.outRows = DEFAULT_ENERGY_ENTRIES;

  //	params.posesPerWI = DEFAULT_PPWI;

  const auto read = [&args](size_t &current, const std::string &arg, const std::initializer_list<std::string> &matches,
                            const std::function<void(std::string)> &handle) {
    if (matches.size() == 0) return false;
    if (std::find(matches.begin(), matches.end(), arg) != matches.end()) {
      if (current + 1 < args.size()) {
        current++;
        handle(args[current]);
      } else {
        std::cerr << "[";
        for (const auto &m : matches)
          std::cerr << m;
        std::cerr << "] specified but no value was given" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      return true;
    }
    return false;
  };

  const auto bindInt = [](const std::string &param, size_t &dest, const std::string &name) {
    try {
      auto parsed = std::stol(param);
      if (parsed < 0) {
        std::cerr << "positive integer required for <" << name << ">: `" << parsed << "`" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      dest = parsed;
    } catch (...) {
      std::cerr << "malformed value, integer required for <" << name << ">: `" << param << "`" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  };

  const auto bindInts = [&](const std::string &param, std::vector<size_t> &dest, const std::string &name) {
    dest = split<size_t>(param, ',', [&](auto &&r) {
      size_t v;
      bindInt(r, v, name);
      return v;
    });
  };

  size_t nposes = 0;
  std::vector<size_t> wgsizes;
  std::vector<size_t> ppwis;

  for (size_t i = 0; i < args.size(); ++i) {
    using namespace std::placeholders;
    const auto arg = args[i];
    if (read(i, arg, {"--iter", "-i"}, [&](auto &&s) { return bindInt(s, params.iterations, "iter"); })) continue;
    if (read(i, arg, {"--poses", "-n"}, [&](auto &&s) { bindInt(s, nposes, "poses"); })) continue;
    if (read(i, arg, {"--device", "-d"}, [&](auto &&s) { params.deviceSelector = s; })) continue;
    if (read(i, arg, {"--deck"}, [&](auto &&s) { params.deckDir = s; })) continue;
    if (read(i, arg, {"--out", "-o"}, [&](auto &&s) { params.output = s; })) continue;
    if (read(i, arg, {"--rows", "-r"}, [&](auto &&s) { return bindInt(s, params.outRows, "rows"); })) continue;
    if (read(i, arg, {"--wgsize", "-w"}, [&](auto &&s) { bindInts(s, wgsizes, "wgsize"); })) continue;
    if (read(i, arg, {"--ppwi", "-p"}, [&](auto &&s) {
          if (s == "all") ppwis = std::vector<size_t>(PPWIs.begin(), PPWIs.end());
          else {
            bindInts(s, ppwis, "ppwi");
            for (auto &p : ppwis)
              if (std::count(PPWIs.begin(), PPWIs.end(), p) == 0) {
                std::cerr << "PPWI " << p << " is not a supported value, should be one of `" << mk_string(PPWIs) << "`"
                          << std::endl;
                std::exit(EXIT_FAILURE);
              }
          }
        }))
      continue;

    if (arg == "--csv") {
      params.csv = true;
      continue;
    }

    if (arg == "list" || arg == "--list" || arg == "-l") {
      params.list = true;
      continue;
    }

    if (arg == "help" || arg == "--help" || arg == "-h") {
      std::cout << "\n";
      // clang-format off
      std::cout
          << "Usage: ./bude [COMMAND|OPTIONS]\n\n"
          << "Commands:\n"
          << "  help -h --help       Print this message\n"
          << "  list -l --list       List available devices\n"
          << "Options:\n"
          << "  -d --device  INDEX   Select device at INDEX from output of --list, performs a substring match of device names if INDEX is not an integer\n"
             "                       [optional] default=0\n"
          << "  -i --iter    I       Repeat kernel I times\n"
             "                       [optional] default=" << DEFAULT_ITERS << "\n"
          << "  -n --poses   N       Compute energies for only N poses, use 0 for deck max\n"
             "                       [optional] default=0 \n"
          << "  -p --ppwi    PPWI    A CSV list of poses per work-item for the kernel, use `all` for everything\n"
             "                       [optional] default=" << PPWIs[0] << "; available=" << mk_string(PPWIs) << "\n"
          << "  -w --wgsize  WGSIZE  A CSV list of work-group sizes, not all implementations support this parameter\n"
             "                       [optional] default=" << 1 << "\n"
          << "     --deck    DIR     Use the DIR directory as input deck\n"
             "                       [optional] default=`" << DATA_DIR << "`\n"
          << "  -o --out     PATH    Save resulting energies to PATH (no-op if more than one PPWI/WGSIZE specified)\n"
             "                       [optional]\n"
          << "  -r --rows    N       Output first N row(s) of energy values as part of the on-screen result\n"
             "                       [optional] default=" << DEFAULT_ENERGY_ENTRIES << "\n"
          << "     --csv             Output results in CSV format\n"
             "                       [optional] default=false"

          << std::endl;
      // clang-format on
      std::exit(EXIT_SUCCESS);
    }

    std::cout << "Unrecognized argument '" << arg << "' (try '--help')" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (ppwis.empty()) ppwis = {PPWIs[0]};
  if (wgsizes.empty()) wgsizes = {1};

  if (params.list) {
    // Don't read any decks if we're just listing devices
    return {params, wgsizes, ppwis};
  }

  params.ligand = readNStruct<Atom>(params.deckDir + FILE_LIGAND);
  params.protein = readNStruct<Atom>(params.deckDir + FILE_PROTEIN);
  params.forcefield = readNStruct<FFParams>(params.deckDir + FILE_FORCEFIELD);

  // poses is stored in the following format:
  // std::array<std::array<float, N>, 6> for the six degrees of freedom
  auto poses = readNStruct<float>(params.deckDir + FILE_POSES);
  if (poses.size() % 6 != 0) {
    throw std::invalid_argument("Pose size (" + std::to_string(poses.size()) + ") not divisible my 6!");
  }
  auto maxPoses = poses.size() / 6;
  nposes = nposes == 0 ? maxPoses : nposes;
  if (nposes > maxPoses) {
    throw std::invalid_argument("Requested poses (" + std::to_string(nposes) + ") exceeded max poses (" +
                                std::to_string(maxPoses) + ") for deck");
  }

  for (size_t i = 0; i < 6; ++i) {
    params.poses[i].resize(nposes);
    std::copy(std::next(poses.cbegin(), int(i * maxPoses)), std::next(poses.cbegin(), int(i * maxPoses + nposes)),
              params.poses[i].begin());
  }

  // Validate energies
  std::ifstream input(params.deckDir + FILE_REF_ENERGIES);
  std::string line;
  while (std::getline(input, line))
    params.refEnergies.push_back(std::stof(line));
  input.close();
  if (params.nposes() > params.refEnergies.size()) {
    throw std::invalid_argument("Size of reference energies (" + std::to_string(params.refEnergies.size()) +
                                ") is less than poses (" + std::to_string(params.nposes()) + ")");
  }

  params.maxPoses = maxPoses;

  return {params, wgsizes, ppwis};
}

double difference(float a, float b) { return std::fabs(double(a) - b) / a; }

std::pair<double, std::vector<size_t>> validate(const Sample &sample, const Params &p) {
  double maxdiff = 0.0f;
  std::vector<size_t> failedEntries;
  for (size_t i = 0; i < sample.energies.size(); i++) {
    // don't verify anything less than one
    if (std::fabs(p.refEnergies[i]) < 1.f && std::fabs(sample.energies[i]) < 1.f) continue;
    double diff = difference(p.refEnergies[i], sample.energies[i]);
    if (diff > maxdiff) maxdiff = diff;
    if ((100.0 * diff) > DIFF_TOLERANCE_PCT) failedEntries.emplace_back(i);
  }
  return {(100.0 * maxdiff), failedEntries};
}

[[nodiscard]] Result evaluate(const Params &p, const Sample &s, bool verbose) {
  std::vector<double> msWithoutWarmup;
  if (s.kernelTimes.empty()) {
    throw std::logic_error("Sample size is 0, this implementation did not measure kernel runtime!");
  }
  std::transform(s.kernelTimes.begin() + int(p.warmupIterations), //
                 s.kernelTimes.end(), std::back_inserter(msWithoutWarmup),
                 [](auto &&p) { return elapsedMillis(p.first, p.second); });

  auto elapsed = std::accumulate(msWithoutWarmup.begin(), msWithoutWarmup.end(), 0.0);
  // Average time per iteration
  double ms = (elapsed / double(p.iterations));
  double runtime = ms * 1e-3;

  // Compute FLOP/s
  size_t ops_per_wg = s.ppwi * 27 + p.natlig() * (2 + s.ppwi * 18 + p.natpro() * (10 + s.ppwi * 30)) + s.ppwi;
  double total_ops = double(ops_per_wg) * (double(p.nposes()) / double(s.ppwi));
  double flops = total_ops / runtime;
  double gflops = flops / 1e9;

  size_t total_finsts = 25 * p.natpro() * p.natlig() * p.nposes();
  double finsts = double(total_finsts) / runtime;
  double gfinsts = finsts / 1e9;

  size_t interactions = p.nposes() * p.natlig() * p.natpro();
  double interactions_per_sec = double(interactions) / runtime;
  auto [maxDiffPct, failedEntries] = validate(s, p);
  bool valid = maxDiffPct < DIFF_TOLERANCE_PCT;

  // expect numbers to be accurate to 2 decimal places
  // verify tolerance
  if (!valid && verbose) {
    std::cerr << "# Verification failed for ppwi=" << s.ppwi << ", wgsize=" << s.wgsize
              << "; difference exceeded tolerance (" << DIFF_TOLERANCE_PCT << "%)"
              << "\n";
    std::cerr << "# Bad energies (failed/total=" << failedEntries.size() << "/" << s.energies.size()
              << ", showing first " << std::min(failedEntries.size(), p.outRows) << "): \n"
              << "# index,actual,expected,difference_%"
              << "\n";
    for (size_t i = 0; i < std::min(failedEntries.size(), p.outRows); ++i) {
      auto idx = failedEntries[i];
      std::cerr << "# " << idx << "," << s.energies[idx] << "," << p.refEnergies[idx] << ","
                << (difference(p.refEnergies[i], s.energies[i]) * 100.f) << "\n";
    }
    // flush at the end to make sure errors are clumped together
    std::cerr << std::flush;
    return {valid, maxDiffPct, s, SummaryStats<double>({std::numeric_limits<double>::max()}), 0, 0, 0};
  } else if (!valid)
    return {valid, maxDiffPct, s, SummaryStats<double>({std::numeric_limits<double>::max()}), 0, 0, 0};
  else {
    return {valid, maxDiffPct, s, SummaryStats<double>(msWithoutWarmup), gflops, gfinsts, interactions_per_sec};
  }
}

[[nodiscard]] std::pair<int, std::string> selectDevice(const std::string &needle,
                                                       const std::vector<std::pair<size_t, std::string>> &haystack) {
  if (needle.empty()) return haystack.at(0);
  try {
    return haystack.at(std::stoul(needle));
  } catch (const std::exception &e) {
    std::cerr << "# Unable to parse/select device index `" << needle << "`:" << e.what() << std::endl;
    std::cerr << "# Attempting to match device with substring  `" << needle << "`" << std::endl;
    auto matching = std::find_if(haystack.begin(), haystack.end(), [needle](const auto &device) {
      return device.second.find(needle) != std::string::npos;
    });
    if (matching != haystack.end()) {
      return *matching;
    } else if (haystack.size() == 1) {
      std::cerr << "# No matching device but there's only one device, using it anyway" << std::endl;
      return haystack[0];
    } else {
      std::cerr << "# No matching devices" << std::endl;
      return {-1, ""};
    }
  }
}

void dumpResults(const Params &p, const Result &r) {
  // write energies to output if requested
  if (!p.output.empty()) {
    std::fstream out(p.output, std::ios::out | std::ios::trunc);
    for (auto e : r.sample.energies)
      out << std::setw(7) << std::setprecision(2) << e << "\n";
    out.close();
  }
}

void showHumanReadable(const Params &p, const Result &r, int indent = 1) {
  std::string prefix(indent, ' ');
  std::cout.precision(3);

  auto contextMs = r.sample.contextTime
                       ? std::to_string(elapsedMillis(r.sample.contextTime->first, r.sample.contextTime->second))
                       : "~";

  std::vector<double> iterationTimesMs;
  std::transform(r.sample.kernelTimes.begin(), //
                 r.sample.kernelTimes.end(), std::back_inserter(iterationTimesMs),
                 [](auto &&p) { return elapsedMillis(p.first, p.second); });

  std::cout << std::fixed //
            << prefix << " - outcome:             { "
            << "valid: " << (r.valid ? "true" : "false") << ", "
            << "max_diff_%: " << r.maxDiffPct << " }\n"
            << prefix << "   param:               { "
            << "ppwi: " << r.sample.ppwi << ", "
            << "wgsize: " << r.sample.wgsize << " }\n"
            << prefix << "   raw_iterations:      [" << mk_string(iterationTimesMs) << "]\n"
            << prefix << "   context_ms:          " << contextMs << "\n"
            << prefix << "   sum_ms:              " << r.ms.sum << "\n"
            << prefix << "   avg_ms:              " << r.ms.mean << "\n"
            << prefix << "   min_ms:              " << r.ms.min << "\n"
            << prefix << "   max_ms:              " << r.ms.max << "\n"
            << prefix << "   stddev_ms:           " << r.ms.stdDev << "\n"
            << prefix << "   giga_interactions/s: " << (r.interactionsPerSec / 1e9) << "\n"
            << prefix << "   gflop/s:             " << r.gflops << "\n"
            << prefix << "   gfinst/s:            " << r.ginsts << "\n"
            << prefix << "   energies:            "
            << "\n";
  // print out the energies
  for (size_t i = 0; i < std::min(size_t(p.outRows), r.sample.energies.size()); i++)
    std::cout << prefix << std::setprecision(2) << "     - " << r.sample.energies[i] << "\n";
  std::cout << std::flush;
}

void showCsv(const Params &p, const Result &r, bool header) {
  if (header) std::cout << "ppwi,wgsize,sum_ms,avg_ms,min_ms,max_ms,stddev_ms,interactions/s,gflops/s,gfinst/s\n";
  std::cout.precision(3);
  std::cout << std::fixed;
  std::cout << r.sample.ppwi << "," << r.sample.wgsize                                                         //
            << "," << r.ms.sum << "," << r.ms.mean << "," << r.ms.min << "," << r.ms.max << "," << r.ms.stdDev //
            << "," << (r.interactionsPerSec) << "," << r.gflops << "," << r.ginsts << std::endl;
}

template <size_t... Ns>
bool run(const Params &p, const std::vector<size_t> &wgsizes, const std::vector<size_t> &ppwis) {
  static_assert(sizeof...(Ns) > 0, "compile-time PPWI args must be non-empty");

  std::unordered_map<size_t, std::function<const std::vector<Device>()>> enumerate = {{Ns, []() {
                                                                                         auto bude = IMPL_CLS<Ns>();
                                                                                         return bude.enumerateDevices();
                                                                                       }}...};

  std::unordered_map<size_t, std::function<const Sample(size_t, size_t)>> kernel = {
      //
      {Ns, [&p](size_t wgsize, size_t device) {
         auto bude = IMPL_CLS<Ns>();
         auto hp = std::make_unique<Params>(p);
         if (!bude.compatible(*hp, wgsize, device)) {
           std::cerr << "Selected device is not compatible with this implementation, results may not be correct!"
                     << std::endl;
         }
         return bude.fasten(*hp, wgsize, device);
       }}...};

  auto devices = enumerate[ppwis[0]]();
  if (devices.empty()) std::cerr << " # (no devices available)" << std::endl;
  if (p.list) {
    std::cout << (p.csv ? "index,name" : "devices:") << std::endl;
    for (size_t j = 0; j < devices.size(); ++j)
      if (p.csv)                                                                    //
        std::cout << j << "," << devices[j].second << std::endl;                    //
      else                                                                          //
        std::cout << "  " << j << ": \"" << devices[j].second << "\"" << std::endl; //
    return true;
  } else {
    auto dev = selectDevice(p.deviceSelector, devices);
    if (dev.first >= 0) {
      if (!p.csv)
        std::cout << "device: { index: " << dev.first << ", "
                  << " name: \"" << dev.second << "\" }" << std::endl;
      bool dump = true;
      std::vector<Result> results;
      for (auto &ppwi : ppwis) {
        for (auto &wgsize : wgsizes) {

          if (p.nposes() < ppwi * wgsize) {
            std::cout << " # WARNING: pose count " << p.nposes() << " <= (" << wgsize << " (wgsize) * " << ppwi
                      << " (ppwi)), skipping" << std::endl;
            continue;
          }
          if (p.nposes() % (ppwi * wgsize) != 0) {

            std::cout << " # WARNING: pose count " << p.nposes() << " % (" << wgsize << " (wgsize) * " << ppwi
                      << " (ppwi)) != 0, skipping" << std::endl;
            continue;
          }

          auto result = evaluate(p, kernel[ppwi](wgsize, size_t(dev.first)), true);
          results.push_back(result);
          std::cout << "# (ppwi=" << ppwi << ",wgsize=" << wgsize << ",valid=" << result.valid << ")" << std::endl;
          // validate
          if (dump || !result.valid) { // dump failures too
            dump = false;
            dumpResults(p, result); // only write when failure occurs
          }
        }
      }

      if (!p.csv) std::cout << "results:" << std::endl;
      for (size_t i = 0; i < results.size(); ++i) {
        if (p.csv) showCsv(p, results[i], i == 0);
        else
          showHumanReadable(p, results[i]);
      }

      auto min = std::min_element(results.begin(), results.end(),
                                  [](const Result &l, const Result &r) { return l.ms.sum < r.ms.sum; });

      std::cout << (p.csv ? "# " : "") << "best: { "
                << "min_ms: " << min->ms.min << ", "
                << "max_ms: " << min->ms.max << ", "
                << "sum_ms: " << min->ms.sum << ", "
                << "avg_ms: " << min->ms.mean << ", "
                << "ppwi: " << min->sample.ppwi << ", "
                << "wgsize: " << min->sample.wgsize << " }\n";

      return std::all_of(results.begin(), results.end(), [](auto &r) { return r.valid; });
    }
    return false;
  }
}

int main(int argc, char *argv[]) {

  auto args = std::vector<std::string>(argv + 1, argv + argc);
  auto [params, wgsizes, ppwis] = parseParams(args);
  if (!params.csv) {
    std::vector<std::string> compileCmds = MINIBUDE_COMPILE_COMMANDS;
    std::vector<std::string> quotedCmds;
    std::transform(compileCmds.begin(), compileCmds.end(), std::back_inserter(quotedCmds),
                   [](auto &s) { return "\"" + s + "\""; });

    std::cout << "miniBUDE:  " << MINIBUDE_VERSION << "\n"
              << "compile_commands:\n   - " << mk_string(quotedCmds, "\n   - ") << "\n"
              << "vcs:\n";
#ifdef MINIBUDE_VCS_RETRIEVED_STATE
    std::cout << "  commit:  " << MINIBUDE_VCS_HEAD_SHA1 << (MINIBUDE_VCS_IS_DIRTY ? "*" : "") << "\n"
              << "  author:  \"" << MINIBUDE_VCS_AUTHOR_NAME << " (" << MINIBUDE_VCS_AUTHOR_EMAIL << ")\"\n"
              << "  date:    \"" << MINIBUDE_VCS_COMMIT_DATE_ISO8601 << "\"\n"
              << "  subject: \"" << MINIBUDE_VCS_COMMIT_SUBJECT << "\"\n";
#else
    std::cout << "   # " MINIBUDE_WARN_NOT_CMAKE "\n";
#endif

    std::cout << "host_cpu:" << std::endl;

#if defined(CPU_FEATURES_ARCH_X86)
    const auto info = cpu_features::GetX86Info();
    char brand_string[49];
    cpu_features::FillX86BrandString(brand_string);
    std::cout << "  arch:     \""
              << "x86 "
              << "(" << GetX86MicroarchitectureName(GetX86Microarchitecture(&info)) << ")\"\n"
              << "  brand:    \"" << brand_string << "\"\n"
              << "  family:   \"" << info.family << "\"\n"
              << "  model:    \"" << info.model << "\"\n"
              << "  stepping: \"" << info.stepping << "\"" << std::endl;
#elif defined(CPU_FEATURES_ARCH_ARM)
    const auto info = cpu_features::GetArmInfo();
    std::cout << "  arch:        \""
              << "arm "
              << "(" << info.architecture << ")\"\n"
              << "  implementer: \"" << info.implementer << "\"\n"
              << "  variant:     \"" << info.variant << "\"\n"
              << "  part:        \"" << info.part << "\"\n"
              << "  revision:    \"" << info.revision << "\"" << std::endl;
#elif defined(CPU_FEATURES_ARCH_AARCH64)
    const auto info = cpu_features::GetAarch64Info();
    std::cout << "  arch:        \"aarch64\"\n"
              << "  implementer: \"" << info.implementer << "\"\n"
              << "  variant:     \"" << info.variant << "\"\n"
              << "  part:        \"" << info.part << "\"\n"
              << "  revision:    \"" << info.revision << "\"" << std::endl;
#elif defined(CPU_FEATURES_ARCH_MIPS)
    const auto info = cpu_features::GetMipsInfo();
    std::cout << "  arch :\"mips\"" << std::endl;
#elif defined(CPU_FEATURES_ARCH_PPC)
    const auto strings = cpu_features::GetPPCPlatformStrings();
    std::cout << "  arch :    \""
              << "ppc "
              << "(" << strings.type.base_platform << ")\"\n"
              << "  platform: \"" << strings.platform << "\"\n"
              << "  model:    \"" << strings.model << "\"\n"
              << "  machine:  \"" << strings.machine << "\"\n"
              << "  cpu:      \"" << strings.cpu << "\"" << std::endl;
#else
    std::cout << "  ~" << std::endl;
#endif

    auto now = std::time(nullptr);
    std::cout << "time: { epoch_s:" << now << ", formatted: \"" << std::put_time(std::gmtime(&now), "%c %Z") << "\" }"
              << std::endl;

    if (!params.list) {
      std::cout << "deck:\n"
                << "  path:         \"" << params.deckDir << "\"\n"
                << "  poses:        " << params.maxPoses << "\n"
                << "  proteins:     " << params.natpro() << "\n"
                << "  ligands:      " << params.natlig() << "\n"
                << "  forcefields:  " << params.ntypes() << "\n"
                << "config:\n"
                << "  iterations:   " << params.iterations << "\n"
                << "  poses:        " << params.nposes() << "\n"
                << "  ppwi:\n"
                << "    available:  [" << mk_string(PPWIs) << "]\n"
                << "    selected:   [" << mk_string(ppwis) << "]\n"
                << "  wgsize:       [" << mk_string(wgsizes) << "]" << std::endl;
    }
  }
  return run<USE_PPWI>(params, wgsizes, ppwis) ? EXIT_SUCCESS : EXIT_FAILURE;
}
