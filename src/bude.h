#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#define DIFF_TOLERANCE_PCT 0.025f
#define DEFAULT_ITERS 8
#define DEFAULT_ENERGY_ENTRIES 8

#define DATA_DIR "../data/bm1"
#define FILE_LIGAND "/ligand.in"
#define FILE_PROTEIN "/protein.in"
#define FILE_FORCEFIELD "/forcefield.in"
#define FILE_POSES "/poses.in"
#define FILE_REF_ENERGIES "/ref_energies.out"

#define ZERO 0.0f
#define QUARTER 0.25f
#define HALF 0.5f
#define ONE 1.0f
#define TWO 2.0f
#define FOUR 4.0f
#define CNSTNT 45.0f

// Energy evaluation parameters
#define HBTYPE_F 70
#define HBTYPE_E 69
#define HARDNESS 38.0f
#define NPNPDIST 5.5f
#define NPPDIST 1.0f

static constexpr auto FloatMax = std::numeric_limits<float>::max();

struct __attribute__((__packed__)) Atom {
  float x, y, z;
  int32_t type;
};

struct __attribute__((__packed__)) FFParams {
  int32_t hbtype;
  float radius;
  float hphb;
  float elsc;
};

template <typename T> struct Vec3 { T x, y, z; };
template <typename T> struct Vec4 { T x, y, z, w; };

struct Params {
  std::vector<Atom> protein;
  std::vector<Atom> ligand;
  std::vector<FFParams> forcefield;
  std::array<std::vector<float>, 6> poses;

  std::vector<float> refEnergies;
  size_t maxPoses, iterations, warmupIterations, outRows;
  std::string deckDir, output, deviceSelector;
  bool csv;

  bool list;

  [[nodiscard]] size_t totalIterations() const { return iterations + warmupIterations; }

  [[nodiscard]] size_t natpro() const { return protein.size(); }
  [[nodiscard]] size_t natlig() const { return ligand.size(); }
  [[nodiscard]] size_t ntypes() const { return forcefield.size(); }
  [[nodiscard]] size_t nposes() const { return poses[0].size(); }
};

using TimePoint = std::chrono::high_resolution_clock::time_point;

[[nodiscard]] static inline double elapsedMillis(const TimePoint &start, const TimePoint &end) {
  auto elapsedNs = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
  return elapsedNs * 1e-6;
}

[[nodiscard]] static inline TimePoint now() { return std::chrono::high_resolution_clock::now(); }

struct Sample {
  size_t ppwi, wgsize;
  std::vector<float> energies;
  std::vector<std::pair<TimePoint, TimePoint>> kernelTimes;
  std::optional<std::pair<TimePoint, TimePoint>> contextTime;
  Sample(size_t ppwi, size_t wgsize, size_t nposes)
      : ppwi(ppwi), wgsize(wgsize), energies(nposes), kernelTimes(), contextTime() {}
};

using Device = std::pair<size_t, std::string>;

template <size_t PPWI> class Bude {
public:
  [[nodiscard]] virtual std::string name() = 0;
  [[nodiscard]] virtual std::vector<Device> enumerateDevices() = 0;
  [[nodiscard]] virtual bool compatible(const Params &p, size_t wgsize, size_t device) const { return true; };
  [[nodiscard]] virtual Sample fasten(const Params &p, size_t wgsize, size_t device) const = 0;
};
