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

#define DEFAULT_ITERS  8
#define DEFAULT_NPOSES 65536

#define DATA_DIR        "../data"
#define FILE_LIGAND     DATA_DIR "/ligand.dat"
#define FILE_PROTEIN    DATA_DIR "/protein.dat"
#define FILE_FORCEFIELD DATA_DIR "/forcefield.dat"
#define FILE_POSES      DATA_DIR "/poses.dat"


static constexpr sycl::access::mode R = sycl::access::mode::read;
static constexpr sycl::access::mode RW = sycl::access::mode::read_write;

static constexpr sycl::target Global = sycl::target::global_buffer;
static constexpr sycl::target Local = sycl::target::local;

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

