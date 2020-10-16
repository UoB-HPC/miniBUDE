CC      = gcc
CFLAGS  = -std=c99 -Wall -O3 -ffast-math -march=native
LDFLAGS =

PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LDFLAGS = -framework OpenCL
else
	CFLAGS += -fopenmp
	LDFLAGS = -lOpenCL -lm
endif

# -------

EXE = bude-opencl

.PHONY: all $(EXE) clean

all: $(EXE)

$(EXE): bude.c
	$(CC) $(CFLAGS) bude.c $(LDFLAGS) -o $@

clean:
	rm -f $(EXE)
