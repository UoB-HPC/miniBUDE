SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:

MAKEFLAGS += --warn-undefined-variables --no-builtin-rules

# -------

ifndef COMPILER
$(warning COMPILER not set (use CRAY, GNU, INTEL, or PGI))
COMPILER=GNU
endif

CC_CRAY   = cc
CC_GNU    = gcc
CC_INTEL  = icc
CC_PGI    = pgcc
CC = $(CC_$(COMPILER))

OPTIONS 	    = -DUSE_SHARED
CFLAGS_GNU    = -Ofast -march=native -std=c99
CFLAGS_PGI    = -O3 -acc -ta=nvidia -fast -Minfo=accel
CFLAGS_CRAY   = -h list=a -Wx,'--maxrregcount=50'
CFLAGS_INTEL  = -O3 -Ofast -xhost -std=c99 -restrict
CFLAGS = $(CFLAGS_$(COMPILER)) $(OPTIONS)
LDFLAGS =

# NV_FLAGS  = -I$(CUDA_PATH)/include -gencode arch=compute_35,code=sm_35 \
# 			-restrict -Xcompiler "-O3 $(OPTIONS)" -use_fast_math -maxrregcount=62
NV_FLAGS  = -O3 --ptxas-options="-v" -use_fast_math \
		-gencode arch=compute_35,code=sm_35 -restrict \
		-gencode arch=compute_60,code=sm_60 -restrict \
		-gencode arch=compute_70,code=sm_70 -restrict

HIP_FLAGS = -O3

OBJ  = $(patsubst %.cu,%.o,$(wildcard *.cu))
OBJ += $(patsubst %.c,%.o,$(wildcard *.c))

ifdef USE_HIP
LINK = hipcc
else
LINK = $(CC)

# On V100, -DUSE_SHARED introduces correctness issues, -maxrregcount lower performance
LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart
endif

bude: Makefile $(OBJ)
	$(LINK) $(CFLAGS) $(OBJ) -o bude $(LDFLAGS)

%.o: %.c Makefile make.deps
	$(CC) $(CFLAGS) -c $<

ifdef USE_HIP
%.cpp: %.cu
	hipify-perl $< > $@
%.o: %.cpp Makefile make.deps
	hipcc $(HIP_FLAGS) -c $<
else
%.o: %.cu Makefile make.deps
	nvcc $(NV_FLAGS) -c $<
endif

.PHONY: clean

clean:
	rm -f bude bude_cuda.cpp *.ptx *.cub *.lst *.o *.optrpt
