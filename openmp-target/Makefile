SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:

MAKEFLAGS += --warn-undefined-variables --no-builtin-rules

# -------

ifndef TARGET
$(warning TARGET not set (use AMD, INTEL, or NVIDIA))
TARGET = NVIDIA
endif

EXTRA_FLAGS =

# -------

CC_AMD    = aompcc
CC_INTEL  = icx
CC_NVIDIA = cc

LD_AMD_aomp = $(dir $(shell which $(CC_AMD)))clang
LD_AMD_gcc  = gcc

LD_AMD    = $(LD_AMD_$(CC))
LD_INTEL  = $(CC_INTEL)
LD_NVIDIA = $(CC_NVIDIA)

OFLAGS_AMD    = -Ofast
OFLAGS_INTEL  = -Ofast
OFLAGS_NVIDIA = -O3

ARCH_AMD_aomp = gfx906
ARCH_AMD_gcc  = native

ARCH_AMD    = $(ARCH_AMD_$(CC))
ARCH_INTEL  = native
ARCH_NVIDIA = sm_70

TFLAGS_AMD_aomp = -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa
TFLAGS_AMD_gcc  = -fopenmp -foffload=amdgcn-amdhsa='-march=$(ARCH_AMD_aomp)' -foffload=-lm

TFLAGS_AMD    = $(TFLAGS_AMD_$(CC))
TFLAGS_INTEL  = -fiopenmp -fopenmp-targets=spir64
TFLAGS_NVIDIA = -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target

TD_PER_THREAD_AMD    = 1
TD_PER_THREAD_INTEL  = 4
# 2 iteratiuons per loop is best for Pascal, 4 for Volta
TD_PER_THREAD_NVIDIA = $(if $(filter %_60, $(ARCH)),2,4)

LDFLAGS_AMD_aomp  = --target=$(shell uname -m)-pc-linux-gnu
LDFLAGS_AMD_gcc   =

LDFLAGS_AMD     = $(LDFLAGS_AMD_$(CC))
LDFLAGS_INTEL   =
LDFLAGS_NVIDIA  =

# -------

CC = $(CC_$(TARGET))
LD = $(LD_$(TARGET))

OFLAGS = $(OFLAGS_$(TARGET))

ARCH   = $(ARCH_$(TARGET))
TFLAGS = $(TFLAGS_$(TARGET)) -march=$(ARCH)

TD_PER_THREAD = $(TD_PER_THREAD_$(TARGET))

# -------

CFLAGS = $(OFLAGS) $(TFLAGS) -DNUM_TD_PER_THREAD=$(TD_PER_THREAD) $(EXTRA_FLAGS)
LDFLAGS = -lm $(LDFLAGS_$(TARGET))

OBJ = $(patsubst %.c,%.o,$(wildcard *.c))

bude: Makefile $(OBJ)
	$(LD) $(CFLAGS) $(OBJ) -o bude $(LDFLAGS)

%.o: %.c Makefile make.deps
	$(CC) $(CFLAGS) -c $<

.PHONY: clean

clean:
	rm -f bude *.ptx *.cub *.lst *.o *.optrpt
