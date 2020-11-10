SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:

MAKEFLAGS += --warn-undefined-variables --no-builtin-rules

# -------

CC = cc

ARCH = sm_70

# 2 iteratiuons per loop is best for Pascal, 4 for Volta
TD_PER_THREAD = $(if $(filter %_60, $(ARCH)),2,4)

CFLAGS = -O3 -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=$(ARCH) -DNUM_TD_PER_THREAD=$(TD_PER_THREAD)
LDFLAGS = -lm

OBJ = $(patsubst %.c,%.o,$(wildcard *.c))

bude: Makefile $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o bude $(LDFLAGS)

%.o: %.c Makefile make.deps
	$(CC) $(CFLAGS) -c $<

.PHONY: clean

clean:
	rm -f bude *.ptx *.cub *.lst *.o *.optrpt
