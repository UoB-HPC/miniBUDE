SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:

MAKEFLAGS += --warn-undefined-variables --no-builtin-rules

# -------

CC = cc

ARCH = sm_70

CFLAGS = -O3 -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=$(ARCH)
LDFLAGS = -lm

OBJ = $(patsubst %.c,%.o,$(wildcard *.c))

bude: Makefile $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o bude $(LDFLAGS)

%.o: %.c Makefile make.deps
	$(CC) $(CFLAGS) -c $<

.PHONY: clean

clean:
	rm -f bude *.ptx *.cub *.lst *.o *.optrpt
