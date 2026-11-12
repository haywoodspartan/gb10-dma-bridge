// SPDX-License-Identifier: GPL-2.0
/*
 * gpuhostbw - GPU SM bandwidth to system memory on GB10 (ATS / C2C path).
 *
 * The GPU reaches ordinary process memory through NVLink-C2C with the SMMU
 * translating its ATS requests, so a plain pointer from mmap()/malloc() can be
 * handed straight to a kernel.  This tool measures read, write and copy
 * bandwidth of GPU SMs on such memory, to compare against the CPU-core
 * engines of gb10_dmab and to see what page size does to the ATS/SMMU path
 * (run it under `perf stat -e smmuv3_pmcg_*` to count translations).
 *
 *   gpuhostbw [-s MiB] [-n iters] [-m read|write|copy|all] [-T] [-P|-M|-D]
 *             [-b blocks] [-t threads] [-L seconds] [-V]
 *
 *   -s   buffer size (default 1024 MiB; copy uses two)
 *   -n   launches per measurement (default 10), best and mean reported
 *   -m   which kernels (default all)
 *   -T   4 KiB pages: madvise(MADV_NOHUGEPAGE) instead of MADV_HUGEPAGE
 *   -P   cudaHostAlloc (pinned, driver-mapped) instead of mmap
 *   -M   cudaMallocManaged instead of mmap
 *   -D   cudaMalloc (GPU-owned memory, for reference)
 *   -b/-t grid geometry (default 4*SMs blocks of 512 threads)
 *   -L   background mode: loop the read kernel for N seconds, print nothing
 *        until the end (used to load the memory system while membw runs)
 *   -V   verify the copy on the CPU
 */
#include <cuda_runtime.h>
#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

enum alloc_kind { AK_MMAP, AK_PINNED, AK_MANAGED, AK_DEVICE };

static size_t len = 1024ull << 20;
static int iters = 10, no_thp = 0, verify = 0, blocks = 0, threads = 512;
static int do_read = 1, do_write = 1, do_copy = 1;
static enum alloc_kind kind = AK_MMAP;
static double loop_s = 0;

__global__ void k_read(const uint4 *__restrict__ p, size_t n, unsigned long long *sink)
{
	uint4 acc = make_uint4(0, 0, 0, 0);
	for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
	     i += (size_t)gridDim.x * blockDim.x) {
		uint4 v = p[i];
		acc.x += v.x; acc.y ^= v.y; acc.z += v.z; acc.w ^= v.w;
	}
	unsigned s = acc.x + acc.y + acc.z + acc.w;
	if (s == 0xdeadbeef)		/* practically never: keeps the loads alive */
		atomicAdd(sink, 1ull);
}

__global__ void k_write(uint4 *__restrict__ p, size_t n, unsigned seed)
{
	for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
	     i += (size_t)gridDim.x * blockDim.x)
		p[i] = make_uint4(seed, (unsigned)i, seed ^ (unsigned)i, ~seed);
}

__global__ void k_copy(uint4 *__restrict__ d, const uint4 *__restrict__ s, size_t n)
{
	for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
	     i += (size_t)gridDim.x * blockDim.x)
		d[i] = s[i];
}

static void *alloc_buf(size_t n)
{
	void *p = NULL;

	switch (kind) {
	case AK_MMAP: {
		size_t align = 2u << 20, m = n + align;
		char *raw = (char *)mmap(NULL, m, PROT_READ | PROT_WRITE,
					 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		if (raw == MAP_FAILED) { perror("mmap"); exit(1); }
		p = (void *)(((uintptr_t)raw + align - 1) & ~(uintptr_t)(align - 1));
		if (madvise(p, n, no_thp ? MADV_NOHUGEPAGE : MADV_HUGEPAGE))
			perror("madvise");
		memset(p, 1, n);
		break;
	}
	case AK_PINNED:
		CK(cudaHostAlloc(&p, n, cudaHostAllocDefault));
		memset(p, 1, n);
		break;
	case AK_MANAGED:
		CK(cudaMallocManaged(&p, n));
		memset(p, 1, n);
		break;
	case AK_DEVICE:
		CK(cudaMalloc(&p, n));
		CK(cudaMemset(p, 1, n));
		break;
	}
	return p;
}

static long anon_huge_kb(void)
{
	FILE *f = fopen("/proc/self/smaps_rollup", "r");
	char line[256];
	long kb = -1;

	if (!f)
		return -1;
	while (fgets(line, sizeof line, f))
		if (sscanf(line, "AnonHugePages: %ld kB", &kb) == 1)
			break;
	fclose(f);
	return kb;
}

static double run(const char *name, void (*launch)(void), size_t bytes_moved)
{
	cudaEvent_t a, b;
	float ms, best = 1e30f, sum = 0;

	CK(cudaEventCreate(&a));
	CK(cudaEventCreate(&b));
	launch();			/* warm-up: first touch, ATC fill */
	CK(cudaDeviceSynchronize());
	for (int i = 0; i < iters; i++) {
		CK(cudaEventRecord(a));
		launch();
		CK(cudaEventRecord(b));
		CK(cudaEventSynchronize(b));
		CK(cudaEventElapsedTime(&ms, a, b));
		if (ms < best)
			best = ms;
		sum += ms;
	}
	double gbs_best = bytes_moved / (best * 1e-3) / 1e9;
	double gbs_mean = bytes_moved / (sum / iters * 1e-3) / 1e9;
	printf("%-6s %8.2f GB/s best  %8.2f GB/s mean  (%zu MiB, %d launches)\n",
	       name, gbs_best, gbs_mean, bytes_moved >> 20, iters);
	return gbs_best;
}

static uint4 *A, *B;
static unsigned long long *sink;
static size_t n4;

static void l_read(void) { k_read<<<blocks, threads>>>(A, n4, sink); }
static void l_write(void) { k_write<<<blocks, threads>>>(A, n4, 0x1234567u); }
static void l_copy(void) { k_copy<<<blocks, threads>>>(B, A, n4); }

int main(int argc, char **argv)
{
	int opt, dev = 0, sms = 0;

	while ((opt = getopt(argc, argv, "s:n:m:TPMDb:t:L:V")) != -1) {
		switch (opt) {
		case 's': len = strtoull(optarg, NULL, 0) << 20; break;
		case 'n': iters = atoi(optarg); break;
		case 'm':
			do_read = !!strstr(optarg, "read") || !strcmp(optarg, "all");
			do_write = !!strstr(optarg, "write") || !strcmp(optarg, "all");
			do_copy = !!strstr(optarg, "copy") || !strcmp(optarg, "all");
			break;
		case 'T': no_thp = 1; break;
		case 'P': kind = AK_PINNED; break;
		case 'M': kind = AK_MANAGED; break;
		case 'D': kind = AK_DEVICE; break;
		case 'b': blocks = atoi(optarg); break;
		case 't': threads = atoi(optarg); break;
		case 'L': loop_s = atof(optarg); break;
		case 'V': verify = 1; break;
		default:
			fprintf(stderr, "usage: gpuhostbw [-s MiB] [-n iters] [-m read|write|copy|all] [-T] [-P|-M|-D] [-b blocks] [-t threads] [-L sec] [-V]\n");
			return 2;
		}
	}
	CK(cudaSetDevice(dev));
	CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
	if (!blocks)
		blocks = sms * 4;
	{
		int pa = 0, hmm = 0;
		cudaDeviceProp pr;
		CK(cudaGetDeviceProperties(&pr, dev));
		CK(cudaDeviceGetAttribute(&pa, cudaDevAttrPageableMemoryAccess, dev));
		CK(cudaDeviceGetAttribute(&hmm, cudaDevAttrPageableMemoryAccessUsesHostPageTables, dev));
		printf("%s sm_%d%d %d SMs, pageableMemoryAccess=%d usesHostPageTables=%d, grid %dx%d\n",
		       pr.name, pr.major, pr.minor, sms, pa, hmm, blocks, threads);
	}

	n4 = len / sizeof(uint4);
	long huge0 = anon_huge_kb();
	A = (uint4 *)alloc_buf(len);
	if (do_copy || loop_s > 0)
		B = (uint4 *)alloc_buf(len);
	long huge1 = anon_huge_kb();
	CK(cudaMalloc(&sink, sizeof *sink));
	CK(cudaMemset(sink, 0, sizeof *sink));
	printf("buffers: %s%s, %zu MiB each, AnonHugePages +%ld MiB\n",
	       kind == AK_MMAP ? "mmap" : kind == AK_PINNED ? "cudaHostAlloc" :
	       kind == AK_MANAGED ? "cudaMallocManaged" : "cudaMalloc",
	       kind == AK_MMAP ? (no_thp ? " (4 KiB pages)" : " (THP hint)") : "",
	       len >> 20, (huge1 - huge0) / 1024);

	if (loop_s > 0) {
		struct timespec t0, t1;
		unsigned long launches = 0;

		clock_gettime(CLOCK_MONOTONIC, &t0);
		do {
			for (int i = 0; i < 4; i++) {
				l_copy();
				launches++;
			}
			CK(cudaDeviceSynchronize());
			clock_gettime(CLOCK_MONOTONIC, &t1);
		} while ((t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9 < loop_s);
		double s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
		printf("background copy: %lu launches, %.1f GB/s copied (%.1f GB/s traffic) for %.1f s\n",
		       launches, launches * (double)len / s / 1e9, 2 * launches * (double)len / s / 1e9, s);
		return 0;
	}

	if (do_read)
		run("read", l_read, len);
	if (do_write)
		run("write", l_write, len);
	if (do_copy) {
		run("copy", l_copy, len);
		if (verify) {
			CK(cudaDeviceSynchronize());
			if (kind == AK_DEVICE) {
				printf("verify: skipped for cudaMalloc\n");
			} else {
				int bad = memcmp(A, B, len);
				printf("verify: %s\n", bad ? "MISMATCH" : "OK");
				if (bad)
					return 1;
			}
		}
	}
	return 0;
}
