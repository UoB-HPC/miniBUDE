SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DELETE_ON_ERROR:

MAKEFLAGS += --warn-undefined-variables --no-builtin-rules

# -------

ifndef COMPILER
$(error COMPILER not set (use ARM, CLANG, CRAY, GNU, or INTEL))
endif

MACHINE = $(shell uname -m)
OS      = $(shell uname -s)

# -------

WGSIZE = 64
ARCH   = native

CC_ARM    = armclang
CC_CLANG  = clang
CC_CRAY   = cc
CC_GNU    = gcc
CC_INTEL  = icc
CC = $(CC_$(COMPILER))

# The target CPU is specificed differently on x86 and on aarch64
# https://community.arm.com/developer/tools-software/tools/b/tools-software-ides-blog/posts/compiler-flags-across-architectures-march-mtune-and-mcpu
ifeq ($(MACHINE), aarch64)
ARCHFLAG = mcpu
else
ARCHFLAG = march
endif

CFLAGS_ARM   = -std=c99 -Wall -Ofast -ffast-math -fopenmp -fsimdmath
CFLAGS_CLANG = -std=c99 -Wall -Ofast -ffast-math -fopenmp
CFLAGS_CRAY  = -O3 -hfp4 -haggress -hflex_mp=tolerant -hvector3 -hscalar3 -Gfast
CFLAGS_GNU   = -std=c99 -Wall -Ofast -ffast-math -fopenmp
CFLAGS_INTEL = -std=c99 -Wall -Ofast -ffast-math -fopenmp
CFLAGS = $(CFLAGS_$(COMPILER)) -DWGSIZE=$(WGSIZE) -$(ARCHFLAG)=$(ARCH)

LDFLAGS = -lm


# -------

EXE = bude

.PHONY: all $(EXE) clean

all: $(EXE)

$(EXE): bude.c bude.h vec-pose-inner.c
	$(CC) $(CFLAGS) bude.c vec-pose-inner.c -o $@ $(LDFLAGS)

clean:
	rm -f $(EXE)
