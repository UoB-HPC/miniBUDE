SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:

MAKEFLAGS += --warn-undefined-variables --no-builtin-rules

# -------

ifndef COMPILER
$(warning COMPILER not set (use CRAY, GNU, or PGI))
COMPILER=GNU
endif

CC_CRAY   = cc
CC_GNU    = gcc
CC_PGI    = pgcc
CC = $(CC_$(COMPILER))

OPTIONS 	    = -DUSE_SHARED
CFLAGS_GNU    = -Ofast -march=native -std=c99 -foffload=nvptx-none
CFLAGS_PGI    = -O3 -acc -ta=nvidia -fast -Minfo=accel
CFLAGS_CRAY   = -h list=a
CFLAGS = $(CFLAGS_$(COMPILER)) $(OPTIONS)
LDFLAGS =

OBJ = $(patsubst %.c,%.o,$(wildcard *.c))

bude: Makefile $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o bude $(LDFLAGS)

%.o: %.c Makefile make.deps
	$(CC) $(CFLAGS) -c $<

.PHONY: clean

clean:
	rm -f bude *.ptx *.cub *.lst *.o *.optrpt
