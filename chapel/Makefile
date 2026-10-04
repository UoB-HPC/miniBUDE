SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:

MAKEFLAGS += --warn-undefined-variables --no-builtin-rules

WGSIZE = 64
ARCH   = native

# -------

MACHINE = $(shell uname -m)

ifeq ($(MACHINE), x86_64)

# On Skylake and Cascade Lake, 256-bit vectors are used by default, but 512-bit is more beneficial
AVX512   =  $(if $(filter skylake% cascadelake% %avx512, $(ARCH)),yes,)

# Intel platforms benefit more from unrolling, specially Skylake and later
WGSIZE = 256

endif

VI = 4
ifeq ($(shell expr $(WGSIZE) \< 32), 1)
   VI = 1
endif

CHPLFLAGS = --warnings --fast --no-ieee-float --mllvm "-force-vector-interleave=$(VI)" -s WGSIZE=$(WGSIZE) $(if $(AVX512),--mllvm "-force-vector-width=512")
CHPL = chpl

# -------
EXE = bude


.PHONY: all $(EXE) clean

all: $(EXE)

$(EXE): Bude.chpl
	$(CHPL) $(CHPLFLAGS) Bude.chpl -o $@

clean:
	rm -f $(EXE)
