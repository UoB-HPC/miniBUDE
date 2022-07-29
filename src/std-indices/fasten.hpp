#pragma once

#include "../bude.h"
#include "dpl_shim.h"
#include <cstdint>
#include <cstdlib>
#include <string>

#ifdef IMPL_CLS
  #error IMPL_CLS was already defined
#endif
#define IMPL_CLS StdIndicesBude

// A lightweight counting iterator which will be used by the STL algorithms
// NB: C++ <= 17 doesn't have this built-in, and it's only added later in ranges-v3 (C++2a) which this
// implementation doesn't target
template <typename N> class ranged {
public:
  class iterator {
    friend class ranged;

  public:
    using difference_type = N;
    using value_type = N;
    using pointer = const N *;
    using reference = N;
    using iterator_category = std::random_access_iterator_tag;

    // XXX This is not part of the iterator spec, it gets picked up by oneDPL if enabled.
    // Without this, the DPL SYCL backend collects the iterator data on the host and copies to the device.
    // This type is unused for any other STL impl.
    using is_passed_directly = std::true_type;

    reference operator*() const { return i_; }
    iterator &operator++() {
      ++i_;
      return *this;
    }
    iterator operator++(int) {
      iterator copy(*this);
      ++i_;
      return copy;
    }

    iterator &operator--() {
      --i_;
      return *this;
    }
    iterator operator--(int) {
      iterator copy(*this);
      --i_;
      return copy;
    }

    iterator &operator+=(N by) {
      i_ += by;
      return *this;
    }

    value_type operator[](const difference_type &i) const { return i_ + i; }

    difference_type operator-(const iterator &it) const { return i_ - it.i_; }
    iterator operator+(const value_type v) const { return iterator(i_ + v); }

    bool operator==(const iterator &other) const { return i_ == other.i_; }
    bool operator!=(const iterator &other) const { return i_ != other.i_; }
    bool operator<(const iterator &other) const { return i_ < other.i_; }

  protected:
    explicit iterator(N start) : i_(start) {}

  private:
    N i_;
  };

  [[nodiscard]] iterator begin() const { return begin_; }
  [[nodiscard]] iterator end() const { return end_; }
  ranged(N begin, N end) : begin_(begin), end_(end) {}

private:
  iterator begin_;
  iterator end_;
};

template <size_t PPWI> class IMPL_CLS final : public Bude<PPWI> {

  static void fasten_main(const ranged<int> &range, size_t natlig, size_t natpro,                          //
                          const Atom *proteins, const Atom *ligands,                                       //
                          const float *transforms_0, const float *transforms_1, const float *transforms_2, //
                          const float *transforms_3, const float *transforms_4, const float *transforms_5, //
                          const FFParams *forcefield, float *energies) {

    std::for_each(exec_policy, range.begin(), range.end(), [=](int group) {
      std::array<std::array<Vec4<float>, 3>, PPWI> transform = {};
      std::array<float, PPWI> etot = {};

#pragma omp simd
      for (int l = 0; l < PPWI; l++) {
        int ix = group * PPWI + l;

        const float sx = std::sin(transforms_0[ix]);
        const float cx = std::cos(transforms_0[ix]);
        const float sy = std::sin(transforms_1[ix]);
        const float cy = std::cos(transforms_1[ix]);
        const float sz = std::sin(transforms_2[ix]);
        const float cz = std::cos(transforms_2[ix]);

        transform[l][0].x = cy * cz;
        transform[l][0].y = sx * sy * cz - cx * sz;
        transform[l][0].z = cx * sy * cz + sx * sz;
        transform[l][0].w = transforms_3[ix];
        transform[l][1].x = cy * sz;
        transform[l][1].y = sx * sy * sz + cx * cz;
        transform[l][1].z = cx * sy * sz - sx * cz;
        transform[l][1].w = transforms_4[ix];
        transform[l][2].x = -sy;
        transform[l][2].y = sx * cy;
        transform[l][2].z = cx * cy;
        transform[l][2].w = transforms_5[ix];
      }

      for (int l = 0; l < natlig; l++) {
        // Loop over ligand atoms
        const Atom l_atom = ligands[l];
        const FFParams l_params = forcefield[l_atom.type];
        const int lhphb_ltz = l_params.hphb < ZERO;
        const int lhphb_gtz = l_params.hphb > ZERO;

        // Transform ligand atom
        std::array<Vec3<float>, PPWI> lpos = {};
#pragma omp simd
        for (int l = 0; l < PPWI; l++) {
          lpos[l].x = transform[l][0].w + l_atom.x * transform[l][0].x + l_atom.y * transform[l][0].y +
                      l_atom.z * transform[l][0].z;
          lpos[l].y = transform[l][1].w + l_atom.x * transform[l][1].x + l_atom.y * transform[l][1].y +
                      l_atom.z * transform[l][1].z;
          lpos[l].z = transform[l][2].w + l_atom.x * transform[l][2].x + l_atom.y * transform[l][2].y +
                      l_atom.z * transform[l][2].z;
        }

        // Loop over protein atoms
        for (int p = 0; p < natpro; p++) {
          const Atom p_atom = proteins[p];
          // Load protein atom data
          const FFParams p_params = forcefield[p_atom.type];

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

      // Write result

#pragma omp simd
      for (int l = 0; l < PPWI; l++) {
        energies[group * PPWI + l] = etot[l] * HALF;
      }
    });
  }

  template <typename C> auto alloc(const C &source, bool copy = true) const {
    auto ptr = alloc_raw<typename C::value_type>(source.size());
    if (copy) std::copy(source.begin(), source.end(), ptr);
    return ptr;
  };

public:
  IMPL_CLS() = default;

  [[nodiscard]] std::string name() override { return "std"; };

  [[nodiscard]] std::vector<Device> enumerateDevices() override { return {{0, "(not exposed)"}}; };

  [[nodiscard]] Sample fasten(const Params &p, size_t wgsize, size_t) const override {

    if (wgsize != 1 && wgsize != 0) {
      throw std::invalid_argument("Only wgsize = {1|0} (i.e no workgroup) are supported for std-indices, got " +
                                  std::to_string((wgsize)));
    }

    Sample sample(PPWI, wgsize, p.nposes());

    auto contextStart = now();
    const auto proteins = alloc(p.protein);
    const auto ligands = alloc(p.ligand);
    const auto forcefield = alloc(p.forcefield);
    const auto transforms_0 = alloc(p.poses[0]);
    const auto transforms_1 = alloc(p.poses[1]);
    const auto transforms_2 = alloc(p.poses[2]);
    const auto transforms_3 = alloc(p.poses[3]);
    const auto transforms_4 = alloc(p.poses[4]);
    const auto transforms_5 = alloc(p.poses[5]);
    auto energies = alloc(sample.energies, false);
    auto contextEnd = now();
    sample.contextTime = {contextStart, contextEnd};

    const auto range = ranged<int>(0, p.nposes() / PPWI);

    for (size_t i = 0; i < p.totalIterations(); ++i) {
      auto kernelStart = now();
      fasten_main(range, p.natlig(), p.natpro(),                                                      //
                  proteins, ligands,                                                                  //
                  transforms_0, transforms_1, transforms_2, transforms_3, transforms_4, transforms_5, //
                  forcefield, energies);
      auto kernelEnd = now();
      sample.kernelTimes.emplace_back(kernelStart, kernelEnd);
    }

    std::copy(energies, energies + sample.energies.size(), sample.energies.begin());

    dealloc_raw(proteins);
    dealloc_raw(ligands);
    dealloc_raw(forcefield);
    dealloc_raw(transforms_0);
    dealloc_raw(transforms_1);
    dealloc_raw(transforms_2);
    dealloc_raw(transforms_3);
    dealloc_raw(transforms_4);
    dealloc_raw(transforms_5);
    dealloc_raw(energies);

    return sample;
  };
};
