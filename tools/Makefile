# SPDX-License-Identifier: GPL-2.0
CC      ?= gcc
CFLAGS  ?= -O2 -g -Wall -Wextra -Wno-unused-parameter
CUDA_HOME ?= /usr/local/cuda

ifneq ($(wildcard $(CUDA_HOME)/include/cuda.h),)
CFLAGS += -DHAVE_CUDA -I$(CUDA_HOME)/include
LDLIBS += -lcuda
endif

TARGETS = dmabctl
ifneq ($(and $(wildcard $(CUDA_HOME)/include/cuda.h),$(wildcard /usr/include/infiniband/verbs.h)),)
TARGETS += rdma_dmabuf_check
ifneq ($(wildcard /usr/include/rdma/rdma_cma.h),)
TARGETS += ringbw
endif
endif
ifneq ($(wildcard $(CUDA_HOME)/include/nvrtc.h),)
NVRTC_FLAGS = -DHAVE_NVRTC -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lnvrtc
endif

all: $(TARGETS)

dmabctl: dmabctl.c ../include/uapi/gb10_dmab.h
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

rdma_dmabuf_check: rdma_dmabuf_check.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS) -libverbs

ringbw: ringbw.c
	$(CC) $(CFLAGS) -pthread -o $@ $< $(LDLIBS) $(NVRTC_FLAGS) -lrdmacm -libverbs

clean:
	rm -f dmabctl rdma_dmabuf_check ringbw

.PHONY: all clean
