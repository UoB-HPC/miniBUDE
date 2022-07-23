#pragma once

#include "../bude.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <execution>
#include <string>

#ifdef IMPL_CLS
  #error IMPL_CLS was already defined
#endif
#define IMPL_CLS StdIndicesBude

template <typename Z = size_t> class ranged {
  Z from, to;

public:
  ranged(Z from, Z to) : from(from), to(to) {}
  class iterator {
    Z num;

  public:
    using difference_type = Z;
    using value_type = Z;
    using pointer = const Z *;
    using reference = Z &;
    using iterator_category = std::random_access_iterator_tag;
    explicit iterator(Z _num = 0) : num(_num) {}
    iterator &operator++() {
      num++;
      return *this;
    }
    iterator operator++(int) {
      iterator retval = *this;
      ++(*this);
      return retval;
    }
    iterator operator+(const value_type v) const { return iterator(num + v); }

    bool operator==(iterator other) const { return num == other.num; }
    bool operator!=(iterator other) const { return *this != other; }
    bool operator<(iterator other) const { return num < other.num; }
    reference operator*() const { return num; }
    difference_type operator-(const iterator &it) const { return num - it.num; }

    value_type operator[](const difference_type &i) const { return num + i; }
  };
  iterator begin() { return iterator(from); }
  iterator end() { return iterator(to >= from ? to + 1 : to - 1); }
};

template <size_t PPWI> class IMPL_CLS final : public Bude<PPWI> {

  static void fasten_main(const Params &p, std::vector<float> &results) {

    auto groups = ranged<int>(0, p.nposes() / PPWI);

    //    const auto natpro = p.natpro();
    //    const auto natlig = p.natlig();
    //    const auto   proteins = p.protein.data();
    //    const auto   ligands = p.ligand.data();
    //    const auto   forcefield = p.forcefield.data();
    //    const auto   transforms_0 = p.poses[0].data();
    //    const auto   transforms_1 = p.poses[1].data();
    //    const auto   transforms_2 = p.poses[2].data();
    //    const auto   transforms_3 = p.poses[3].data();
    //    const auto   transforms_4 = p.poses[4].data();
    //    const auto   transforms_5 = p.poses[5].data();
    //    const auto   energies = results.data();

    std::for_each(std::execution::par_unseq, groups.begin(), groups.end(), [&, energies = results.data()](int group) {
      std::array<std::array<Vec4<float>, 3>, PPWI> transform = {};
      std::array<float, PPWI> etot = {};

#pragma omp simd
      for (int l = 0; l < PPWI; l++) {
        int ix = group * PPWI + l;

        const float sx = std::sin(p.poses[0][ix]);
        const float cx = std::cos(p.poses[0][ix]);
        const float sy = std::sin(p.poses[1][ix]);
        const float cy = std::cos(p.poses[1][ix]);
        const float sz = std::sin(p.poses[2][ix]);
        const float cz = std::cos(p.poses[2][ix]);

        transform[l][0].x = cy * cz;
        transform[l][0].y = sx * sy * cz - cx * sz;
        transform[l][0].z = cx * sy * cz + sx * sz;
        transform[l][0].w = p.poses[3][ix];
        transform[l][1].x = cy * sz;
        transform[l][1].y = sx * sy * sz + cx * cz;
        transform[l][1].z = cx * sy * sz - sx * cz;
        transform[l][1].w = p.poses[4][ix];
        transform[l][2].x = -sy;
        transform[l][2].y = sx * cy;
        transform[l][2].z = cx * cy;
        transform[l][2].w = p.poses[5][ix];
      }

      // Loop over ligand atoms
      for (const Atom &l_atom : p.ligand) {
        const FFParams l_params = p.forcefield[l_atom.type];
        const int lhphb_ltz = l_params.hphb < ZERO;
        const int lhphb_gtz = l_params.hphb > ZERO;

        // Transform ligand atom
        std::array<Vec3<float>, PPWI> lpos = {};
#pragma omp simd
        for (int l = 0; l < PPWI; l++) {
          lpos[l].x = transform[l][0].w + l_atom.x * transform[l][0].x + l_atom.y * transform[l][0].y + l_atom.z * transform[l][0].z;
          lpos[l].y = transform[l][1].w + l_atom.x * transform[l][1].x + l_atom.y * transform[l][1].y + l_atom.z * transform[l][1].z;
          lpos[l].z = transform[l][2].w + l_atom.x * transform[l][2].x + l_atom.y * transform[l][2].y + l_atom.z * transform[l][2].z;
        }

        // Loop over protein atoms
        for (const Atom &p_atom : p.protein) {
          //          // Load protein atom data
          const FFParams p_params = p.forcefield[p_atom.type];

          const float radij = p_params.radius + l_params.radius;
          const float r_radij = ONE / radij;

          const float elcdst = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? FOUR : TWO;
          const float elcdst1 = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? QUARTER : HALF;
          const int type_E = ((p_params.hbtype == HBTYPE_E || l_params.hbtype == HBTYPE_E));

          const int phphb_ltz = p_params.hphb < ZERO;
          const int phphb_gtz = p_params.hphb > ZERO;
          const int phphb_nz = p_params.hphb != ZERO;
          const float p_hphb = p_params.hphb * (phphb_ltz && lhphb_gtz ? -ONE : ONE);
          const float l_hphb = l_params.hphb * (phphb_gtz && lhphb_ltz ? -ONE : ONE);
          const float distdslv = (phphb_ltz ? (lhphb_ltz ? NPNPDIST : NPPDIST) : (lhphb_ltz ? NPPDIST : -FloatMax));
          const float r_distdslv = ONE / distdslv;

          const float chrg_init = l_params.elsc * p_params.elsc;
          const float dslv_init = p_hphb + l_hphb;

#pragma omp simd
          for (int l = 0; l < PPWI; l++) {

            // Calculate distance between atoms
            const float x = lpos[l].x - p_atom.x;
            const float y = lpos[l].y - p_atom.y;
            const float z = lpos[l].z - p_atom.z;
            const float distij = std::sqrt(x * x + y * y + z * z);

            // Calculate the sum of the sphere radii
            const float distbb = distij - radij;

            const int zone1 = (distbb < ZERO);

            //  Calculate steric energy
            etot[l] += (ONE - (distij * r_radij)) * (zone1 ? TWO * HARDNESS : ZERO);

            // Calculate formal and dipole charge interactions
            float chrg_e = chrg_init * ((zone1 ? ONE : (ONE - distbb * elcdst1)) * (distbb < elcdst ? ONE : ZERO));
            float neg_chrg_e = -std::abs(chrg_e);
            chrg_e = type_E ? neg_chrg_e : chrg_e;
            etot[l] += chrg_e * CNSTNT;

            // Calculate the two cases for Nonpolar-Polar repulsive interactions
            float coeff = (ONE - (distbb * r_distdslv));
            float dslv_e = dslv_init * ((distbb < distdslv && phphb_nz) ? ONE : ZERO);
            dslv_e *= (zone1 ? ONE : coeff);
            etot[l] += dslv_e;
          }
        }
      }

      ////#pragma omp simd
      //          for (int l = 0; l < PPWI; l++) {
      //            etot[l] *= HALF;
      //          }
      //
      //          return std::make_pair(group, etot);

// Write result
#pragma omp simd
      for (int l = 0; l < PPWI; l++) {
        energies[group * PPWI + l] = etot[l] * HALF;
      }
    });

    //    energies.clear();
    //    for (auto &[g, xs] : buffer) {
    //      energies.insert(energies.begin() + (PPWI * g), xs.begin(), xs.end());
    //    }

    //    std::transform(buffer.begin(),  buffer.end(), energies.begin(), )
  }

public:
  IMPL_CLS() = default;

  [[nodiscard]] std::string name() override { return "std"; };

  [[nodiscard]] std::vector<Device> enumerateDevices() override { return {{0, "(not exposed)"}}; };

  [[nodiscard]] Sample fasten(const Params &p, size_t wgsize, size_t) const override {

    if (wgsize != 1 && wgsize != 0) {
      throw std::invalid_argument("Only wgsize = {1|0} (i.e no workgroup) are supported for OpenMP, got " +
                                  std::to_string((wgsize)));
    }

    Sample sample(PPWI, wgsize, p.nposes());

    //    std::vector<size_t> groups(p.nposes() / PPWI);

    //    std::generate(groups.begin(), groups.end(), [n = 0]() mutable { return n++; });

    //    std::vector<std::array<float, PPWI>> buffer(groups.size());

    for (size_t i = 0; i < p.totalIterations(); ++i) {
      auto kernelStart = now();
      fasten_main(p, sample.energies);
      auto kernelEnd = now();
      sample.kernelTimes.emplace_back(kernelStart, kernelEnd);
    }

    //    std::transform(buffer.begin(),  buffer.end(), sample.energies.begin(), []);
    //    sample.energies.insert(sample.energies.end(), )

    return sample;
  };
};
