SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:

MAKEFLAGS += --warn-undefined-variables --no-builtin-rules

# -------

ifndef COMPILER
$(warning COMPILER not set (use CRAY, GNU, or PGI))
COMPILER=GNU
endif

ifndef TARGET
$(warning TARGET not set (use AMD or NVIDIA))
TAREGT=NVIDIA
endif

# -------

CC_CRAY   = cc
CC_GNU    = gcc
CC_PGI    = pgcc
CC = $(CC_$(COMPILER))

CFLAGS_GNU_AMD    = -fopenacc -foffload=amdgcn-amdhsa='-march=gfx906' -foffload=-lm -DNUM_TD_PER_THREAD=4
CFLAGS_GNU_NVIDIA = -foffload=nvptx-none

OPTIONS 	  = -DUSE_SHARED
CFLAGS_GNU    = -Ofast -march=native -std=c99 $(CFLAGS_GNU_$(TARGET))
CFLAGS_PGI    = -O3 -acc -ta=nvidia -fast -Minfo=accel
CFLAGS_CRAY   = -h list=a
CFLAGS = $(CFLAGS_$(COMPILER)) $(OPTIONS)
LDFLAGS = -lm

OBJ = $(patsubst %.c,%.o,$(wildcard *.c))

bude: Makefile $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o bude $(LDFLAGS)

%.o: %.c Makefile make.deps
	$(CC) $(CFLAGS) -c $<

.PHONY: clean

clean:
	rm -f bude *.ptx *.cub *.lst *.o *.optrpt
