#pragma once

#include <cstdint>
#include <string>
#include <iomanip>
#include <CL/sycl.hpp>



#ifndef DEFAULT_PPWI
#define DEFAULT_PPWI 1
#endif
#ifndef DEFAULT_WGSIZE
#define DEFAULT_WGSIZE 4
#endif

#ifndef NUM_TD_PER_THREAD
#define NUM_TD_PER_THREAD DEFAULT_PPWI
#endif

#define DEFAULT_ITERS  8
#define DEFAULT_NPOSES 65536
#define REF_NPOSES     65536

#define DATA_DIR          "../data/bm1"
#define FILE_LIGAND       "/ligand.in"
#define FILE_PROTEIN      "/protein.in"
#define FILE_FORCEFIELD   "/forcefield.in"
#define FILE_POSES        "/poses.in"
#define FILE_REF_ENERGIES "/ref_energies.out"

namespace clsycl = cl::sycl;

static constexpr clsycl::access::mode R = clsycl::access::mode::read;
static constexpr clsycl::access::mode W = clsycl::access::mode::write;
static constexpr clsycl::access::mode RW = clsycl::access::mode::read_write;

static constexpr clsycl::access::target Global = clsycl::access::target::global_buffer;
static constexpr clsycl::access::target Local = clsycl::access::target::local;

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

