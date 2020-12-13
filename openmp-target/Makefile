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

LD_AMD    = $(dir $(shell which $(CC_AMD)))clang
LD_INTEL  = $(CC_INTEL)
LD_NVIDIA = $(CC_NVIDIA)

OFLAGS_AMD    = -Ofast
OFLAGS_INTEL  = -Ofast
OFLAGS_NVIDIA = -O3

ARCH_AMD    = gfx906
ARCH_INTEL  = native
ARCH_NVIDIA = sm_70

TFLAGS_AMD    = -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa
TFLAGS_INTEL  = -fiopenmp -fopenmp-targets=spir64
TFLAGS_NVIDIA = -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target

TD_PER_THREAD_AMD    = 1
TD_PER_THREAD_INTEL  = 4
# 2 iteratiuons per loop is best for Pascal, 4 for Volta
TD_PER_THREAD_NVIDIA = $(if $(filter %_60, $(ARCH)),2,4)

LDFLAGS_AMD     = --target=$(shell uname -m)-pc-linux-gnu
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
