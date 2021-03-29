SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
# .ONESHELL: # This doesn't work on all systems. Disabling for now.
.DELETE_ON_ERROR:

MAKEFLAGS += --warn-undefined-variables --no-builtin-rules

# -------

ifndef COMPILER
$(warning COMPILER not set (use ARM, CLANG, CRAY, FUIJTSU, GNU, or INTEL))
COMPILER=GNU
endif

WGSIZE = 64
ARCH   = native
AVX512 =
EXTRA_FLAGS =

# -------

MACHINE = $(shell uname -m)

ifeq ($(MACHINE), x86_64)
ARCHFLAG = march

# On Skylake and Cascade Lake, 256-bit vectors are used by default, but 512-bit is more beneficial
AVX512   =  $(if $(filter skylake% cascadelake% %avx512, $(ARCH)),yes,)

# Intel platforms benefit more from unrolling, specially Skylake and later
WGSIZE = 256

else
# The target CPU is specificed differently on x86 and on aarch64
# https://community.arm.com/developer/tools-software/tools/b/tools-software-ides-blog/posts/compiler-flags-across-architectures-march-mtune-and-mcpu
ARCHFLAG = mcpu
endif

# -------

CC_ARM     = armclang
CC_CLANG   = clang
CC_CRAY    = cc
CC_FUJITSU = fcc
CC_GNU     = gcc
CC_INTEL   = icc
CC = $(CC_$(COMPILER))

CFLAGS_ARM     = -std=c99 -Wall -Ofast -ffast-math -fopenmp -fsimdmath -$(ARCHFLAG)=$(ARCH)
CFLAGS_CLANG   = -std=c99 -Wall -Ofast -ffast-math -fopenmp -$(ARCHFLAG)=$(ARCH) $(if $(AVX512),-mprefer-vector-width=512)
CFLAGS_CRAY    = -O3 -homp -hfp4 -haggress -hflex_mp=tolerant -hvector3 -hscalar3 -Gfast
CFLAGS_FUJITSU = -Nclang -std=c99 -Wall -Ofast -ffast-math -fopenmp -$(ARCHFLAG)=$(ARCH) -mllvm -force-vector-interleave=4
CFLAGS_GNU     = -std=c99 -Wall -Ofast -ffast-math -fopenmp -$(ARCHFLAG)=$(ARCH) $(if $(AVX512),-mprefer-vector-width=512)
CFLAGS_INTEL   = -std=c99 -Wall -Ofast -ffast-math -fopenmp -$(ARCHFLAG)=$(ARCH) $(if $(AVX512),-qopt-zmm-usage=high)
CFLAGS = $(CFLAGS_$(COMPILER)) -DWGSIZE=$(WGSIZE)

LDFLAGS = -lm


# -------

EXE = bude

.PHONY: all $(EXE) clean

all: $(EXE)

$(EXE): bude.c bude.h vec-pose-inner.c
	$(CC) $(CFLAGS) bude.c vec-pose-inner.c -o $@ $(LDFLAGS) $(EXTRA_FLAGS)

clean:
	rm -f $(EXE)
