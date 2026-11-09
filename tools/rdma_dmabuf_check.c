// SPDX-License-Identifier: GPL-2.0
/*
 * rdma_dmabuf_check - which GPUDirect-style paths work between the GB10 GPU
 * and a ConnectX-7 on this node, for GPU traffic over the ring:
 *
 *   stage 0  CUDA device attributes that decide the answer
 *   stage 1  cudaMalloc -> dma-buf fd -> ibv_reg_dmabuf_mr        (dma-buf GPUDirect)
 *   stage 2  ibv_reg_mr() directly on the cudaMalloc VA          (nvidia-peermem / ATS page tables)
 *   stage 3  ibv_reg_mr() on malloc() memory + GPU writes to it  (unified memory via ATS)
 *
 *   rdma_dmabuf_check [ibdev] [size_mib]     default: first ACTIVE device, 256 MiB
 */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <infiniband/verbs.h>
#include <cuda.h>

static int cu_warn(CUresult r, const char *what)
{
	const char *s = "?";

	if (r == CUDA_SUCCESS)
		return 0;
	cuGetErrorString(r, &s);
	printf("  %s: %s (%d)\n", what, s, (int)r);
	return -1;
}

static void cu_check(CUresult r, const char *what)
{
	if (cu_warn(r, what))
		exit(1);
}

static double now_s(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int attr(CUdevice dev, CUdevice_attribute a)
{
	int v = -1;

	cuDeviceGetAttribute(&v, a, dev);
	return v;
}

#define ACCESS (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ)

int main(int argc, char **argv)
{
	const char *want = argc > 1 ? argv[1] : NULL;
	size_t len = (argc > 2 ? strtoul(argv[2], NULL, 0) : 256) << 20;
	struct ibv_device **list, *dev = NULL;
	struct ibv_context *ctx;
	struct ibv_device_attr_ex dattr;
	struct ibv_port_attr port;
	struct ibv_pd *pd;
	struct ibv_mr *mr;
	CUdevice cudev;
	CUcontext cuctx;
	CUdeviceptr raw, dptr;
	char name[64];
	int fd = -1, n, i, ok_dmabuf = 0, ok_direct = 0, ok_unified = 0;
	void *host;
	double t0, t1;

	memset(&dattr, 0, sizeof(dattr));

	/* ---- stage 0: what CUDA says about this GPU ---- */
	cu_check(cuInit(0), "cuInit");
	cu_check(cuDeviceGet(&cudev, 0), "cuDeviceGet");
	cuDeviceGetName(name, sizeof(name), cudev);
	printf("GPU 0: %s\n", name);
	printf("  integrated                         %d\n", attr(cudev, CU_DEVICE_ATTRIBUTE_INTEGRATED));
	printf("  pageable memory access             %d\n", attr(cudev, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS));
	printf("  pageable access uses host PTs      %d\n", attr(cudev, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES));
	printf("  concurrent managed access          %d\n", attr(cudev, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS));
	printf("  dma-buf export supported           %d\n", attr(cudev, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED));
	printf("  GPUDirect RDMA supported           %d\n", attr(cudev, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED));
	printf("  GPUDirect RDMA writes ordering     %d\n", attr(cudev, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING));
	printf("  GPUDirect RDMA flush options       %d\n", attr(cudev, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS));
	printf("  host NUMA id                       %d\n", attr(cudev, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID));

	cu_check(cuDevicePrimaryCtxRetain(&cuctx, cudev), "cuDevicePrimaryCtxRetain");
	cu_check(cuCtxSetCurrent(cuctx), "cuCtxSetCurrent");
	cu_check(cuMemAlloc(&raw, len + (2u << 20)), "cuMemAlloc");
	dptr = (raw + (2u << 20) - 1) & ~(CUdeviceptr)((2u << 20) - 1);
	cu_check(cuMemsetD8(dptr, 0, len), "cuMemsetD8");

	/* ---- RDMA device ---- */
	list = ibv_get_device_list(&n);
	if (!list || !n) {
		fprintf(stderr, "no RDMA devices\n");
		return 1;
	}
	for (i = 0; i < n; i++) {
		if (want) {
			if (!strcmp(ibv_get_device_name(list[i]), want)) {
				dev = list[i];
				break;
			}
			continue;
		}
		ctx = ibv_open_device(list[i]);
		if (!ctx)
			continue;
		if (!ibv_query_port(ctx, 1, &port) && port.state == IBV_PORT_ACTIVE) {
			ibv_close_device(ctx);
			dev = list[i];
			break;
		}
		ibv_close_device(ctx);
	}
	if (!dev) {
		fprintf(stderr, "no %s RDMA device found\n", want ? want : "ACTIVE");
		return 1;
	}
	ctx = ibv_open_device(dev);
	if (!ctx) {
		perror("ibv_open_device");
		return 1;
	}
	ibv_query_device_ex(ctx, NULL, &dattr);
	ibv_query_port(ctx, 1, &port);
	printf("RDMA device %s: fw %s, port %s, active_mtu %u, max_mr_size 0x%llx\n",
	       ibv_get_device_name(dev), dattr.orig_attr.fw_ver,
	       port.state == IBV_PORT_ACTIVE ? "ACTIVE" : "not active",
	       128u << port.active_mtu, (unsigned long long)dattr.orig_attr.max_mr_size);
	pd = ibv_alloc_pd(ctx);
	if (!pd) {
		perror("ibv_alloc_pd");
		return 1;
	}

	/* ---- stage 1: dma-buf ---- */
	printf("stage 1: cudaMalloc -> dma-buf -> ibv_reg_dmabuf_mr\n");
	if (!attr(cudev, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED)) {
		printf("  skipped: CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 0\n");
	} else if (!cu_warn(cuMemGetHandleForAddressRange(&fd, dptr, len,
					CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0),
			    "cuMemGetHandleForAddressRange")) {
		t0 = now_s();
		mr = ibv_reg_dmabuf_mr(pd, 0, len, (uint64_t)dptr, fd, ACCESS);
		t1 = now_s();
		if (mr) {
			printf("  OK: %zu MiB in %.1f ms (lkey 0x%x rkey 0x%x)\n", len >> 20,
			       (t1 - t0) * 1e3, mr->lkey, mr->rkey);
			ibv_dereg_mr(mr);
			ok_dmabuf = 1;
		} else {
			printf("  FAILED: ibv_reg_dmabuf_mr: %s\n", strerror(errno));
		}
		close(fd);
	}

	/* ---- stage 2: plain ibv_reg_mr on the device pointer ---- */
	printf("stage 2: ibv_reg_mr() on the cudaMalloc VA (nvidia-peermem or CPU-mapped GPU memory)\n");
	t0 = now_s();
	mr = ibv_reg_mr(pd, (void *)(uintptr_t)dptr, len, ACCESS);
	t1 = now_s();
	if (mr) {
		printf("  OK: %zu MiB in %.1f ms (lkey 0x%x rkey 0x%x)\n", len >> 20,
		       (t1 - t0) * 1e3, mr->lkey, mr->rkey);
		ibv_dereg_mr(mr);
		ok_direct = 1;
	} else {
		printf("  FAILED: %s (errno %d)\n", strerror(errno), errno);
	}

	/* ---- stage 3: system memory registered with the NIC, written by the GPU ---- */
	printf("stage 3: ibv_reg_mr() on malloc() memory, then the GPU writes it (ATS / unified memory)\n");
	host = aligned_alloc(2u << 20, len);
	memset(host, 0, len);
	t0 = now_s();
	mr = ibv_reg_mr(pd, host, len, ACCESS);
	t1 = now_s();
	if (!mr) {
		printf("  FAILED: ibv_reg_mr(host): %s\n", strerror(errno));
	} else {
		printf("  NIC MR: %zu MiB in %.1f ms (lkey 0x%x rkey 0x%x)\n", len >> 20,
		       (t1 - t0) * 1e3, mr->lkey, mr->rkey);
		if (!cu_warn(cuMemsetD8((CUdeviceptr)(uintptr_t)host, 0xA5, len), "cuMemsetD8(host ptr)") &&
		    !cu_warn(cuCtxSynchronize(), "cuCtxSynchronize")) {
			size_t bad = 0, k;

			for (k = 0; k < len; k += 4096)
				if (((uint8_t *)host)[k] != 0xA5)
					bad++;
			if (!bad) {
				printf("  OK: GPU wrote all %zu MiB of NIC-registered system memory\n", len >> 20);
				ok_unified = 1;
			} else {
				printf("  FAILED: %zu pages not written by the GPU\n", bad);
			}
		}
		ibv_dereg_mr(mr);
	}
	free(host);

	printf("RESULT: dma-buf GPUDirect=%s  direct cudaMalloc MR=%s  unified (NIC MR + GPU via ATS)=%s\n",
	       ok_dmabuf ? "yes" : "no", ok_direct ? "yes" : "no", ok_unified ? "yes" : "no");

	ibv_dealloc_pd(pd);
	ibv_close_device(ctx);
	ibv_free_device_list(list);
	cuMemFree(raw);
	cuDevicePrimaryCtxRelease(cudev);
	return (ok_dmabuf || ok_direct || ok_unified) ? 0 : 2;
}
