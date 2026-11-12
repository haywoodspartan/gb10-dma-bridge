// SPDX-License-Identifier: GPL-2.0
/*
 * memlat - dependent-load (pointer chase) memory latency from one CPU.
 *
 *   memlat [-s MiB] [-n loads] [-T] CPU
 *
 * Builds a random cyclic permutation over cache lines of a buffer (default
 * 512 MiB, far beyond L3), then walks it. Reports ns per load. With THP
 * (default) the TLB is out of the picture and the number is the DRAM path:
 * core -> cluster -> fabric -> memory controller -> LPDDR5X and back. Use
 * it to recognise the memory system's operating point: on this GB10 the
 * same CPU saw memory bandwidth differ 7x between two days (README section
 * 13); latency is the direct observable for that.
 */
#define _GNU_SOURCE
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static double now_s(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint64_t next_rand(void)
{
	rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
	return rng;
}

int main(int argc, char **argv)
{
	size_t len = 512ull << 20, nloads = 20000000;
	int opt, no_thp = 0, cpu;
	cpu_set_t set;

	while ((opt = getopt(argc, argv, "s:n:T")) != -1) {
		switch (opt) {
		case 's': len = strtoull(optarg, NULL, 0) << 20; break;
		case 'n': nloads = strtoull(optarg, NULL, 0); break;
		case 'T': no_thp = 1; break;
		default:
			fprintf(stderr, "usage: memlat [-s MiB] [-n loads] [-T] CPU\n");
			return 2;
		}
	}
	if (optind >= argc) {
		fprintf(stderr, "usage: memlat [-s MiB] [-n loads] [-T] CPU\n");
		return 2;
	}
	cpu = atoi(argv[optind]);
	CPU_ZERO(&set);
	CPU_SET(cpu, &set);
	if (sched_setaffinity(0, sizeof set, &set)) {
		perror("sched_setaffinity");
		return 1;
	}

	size_t align = 2u << 20;
	char *raw = mmap(NULL, len + align, PROT_READ | PROT_WRITE,
			 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (raw == MAP_FAILED) {
		perror("mmap");
		return 1;
	}
	char *buf = (char *)(((uintptr_t)raw + align - 1) & ~(uintptr_t)(align - 1));
	madvise(buf, len, no_thp ? MADV_NOHUGEPAGE : MADV_HUGEPAGE);
	memset(buf, 0, len);

	/* random cyclic permutation over 64-byte lines (Sattolo) */
	size_t nlines = len / 64;
	uint32_t *perm = malloc(nlines * sizeof *perm);
	if (!perm)
		return 1;
	for (size_t i = 0; i < nlines; i++)
		perm[i] = i;
	for (size_t i = nlines - 1; i > 0; i--) {
		size_t j = next_rand() % i;
		uint32_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
	}
	for (size_t i = 0; i < nlines; i++)
		*(uint64_t *)(buf + (size_t)perm[i] * 64) = (uint64_t)perm[(i + 1) % nlines] * 64;
	free(perm);

	/* warm up, then time */
	uint64_t p = 0;
	for (size_t i = 0; i < 1000000; i++)
		p = *(volatile uint64_t *)(buf + p);
	double t0 = now_s();
	for (size_t i = 0; i < nloads; i++)
		p = *(volatile uint64_t *)(buf + p);
	double t1 = now_s();
	printf("cpu %d: %.1f ns per dependent load over %zu MiB (%s)%s\n", cpu,
	       (t1 - t0) / nloads * 1e9, len >> 20, no_thp ? "4K pages" : "THP",
	       p == 1 ? " " : "");
	return 0;
}
