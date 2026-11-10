// SPDX-License-Identifier: GPL-2.0
/*
 * membw - memory bandwidth of pinned CPU threads: read, write and copy, each
 * thread on its own private 2 MiB-aligned (THP-hinted) buffers.
 *
 *   membw [-s MiB_per_thread] [-n iters] [-T] [-l] CPULIST
 *
 *   CPULIST  e.g. 16  or 16-19  or 6,7,16,17   (one thread per CPU)
 *   -s       buffer size per thread (default 256 MiB; copy uses two)
 *   -n       repetitions (default 5), best run reported
 *   -T       disable the transparent-hugepage hint (4 KiB pages)
 *   -l       long form: per-thread results
 *
 * Prints aggregate GB/s for read (sum of 64-bit words), write (store a
 * pattern) and copy (memcpy, counted as bytes copied, so traffic is 2x).
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static size_t len = 256ull << 20;
static int iters = 5, no_thp = 0, longform = 0;

struct th {
	int cpu;
	uint8_t *a, *b;
	double rd, wr, cp;	/* best GB/s */
	uint64_t sink;
	pthread_t t;
};

static pthread_barrier_t bar;

static double now_s(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void *alloc(size_t n)
{
	void *p;

	if (posix_memalign(&p, 2u << 20, n))
		return NULL;
	madvise(p, n, no_thp ? MADV_NOHUGEPAGE : MADV_HUGEPAGE);
	memset(p, 1, n);
	return p;
}

static uint64_t sum_words(const uint8_t *p, size_t n)
{
	const uint64_t *q = (const uint64_t *)p;
	uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
	size_t i, m = n / 8;

	for (i = 0; i + 4 <= m; i += 4) {
		s0 += q[i]; s1 += q[i + 1]; s2 += q[i + 2]; s3 += q[i + 3];
	}
	return s0 ^ s1 ^ s2 ^ s3;
}

static void fill_words(uint8_t *p, size_t n, uint64_t v)
{
	uint64_t *q = (uint64_t *)p;
	size_t i, m = n / 8;

	for (i = 0; i < m; i++)
		q[i] = v + i;
}

static void *worker(void *arg)
{
	struct th *t = arg;
	cpu_set_t set;
	int k;
	double t0, dt;

	CPU_ZERO(&set);
	CPU_SET(t->cpu, &set);
	if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set)) {
		fprintf(stderr, "cannot pin to cpu %d\n", t->cpu);
		exit(1);
	}
	t->a = alloc(len);
	t->b = alloc(len);
	if (!t->a || !t->b) {
		fprintf(stderr, "alloc failed\n");
		exit(1);
	}
	t->rd = t->wr = t->cp = 0;

	for (k = 0; k < iters; k++) {
		pthread_barrier_wait(&bar);
		t0 = now_s();
		t->sink += sum_words(t->a, len);
		dt = now_s() - t0;
		if (len / dt / 1e9 > t->rd)
			t->rd = len / dt / 1e9;

		pthread_barrier_wait(&bar);
		t0 = now_s();
		fill_words(t->b, len, k);
		dt = now_s() - t0;
		if (len / dt / 1e9 > t->wr)
			t->wr = len / dt / 1e9;

		pthread_barrier_wait(&bar);
		t0 = now_s();
		memcpy(t->b, t->a, len);
		dt = now_s() - t0;
		if (len / dt / 1e9 > t->cp)
			t->cp = len / dt / 1e9;
	}
	free(t->a);
	free(t->b);
	return NULL;
}

static int parse_cpulist(const char *s, int *cpus, int max)
{
	int n = 0;
	char *dup = strdup(s), *tok, *save;

	for (tok = strtok_r(dup, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
		int a, b, i;

		if (sscanf(tok, "%d-%d", &a, &b) == 2) {
			for (i = a; i <= b && n < max; i++)
				cpus[n++] = i;
		} else if (sscanf(tok, "%d", &a) == 1 && n < max) {
			cpus[n++] = a;
		}
	}
	free(dup);
	return n;
}

int main(int argc, char **argv)
{
	int cpus[256], n, i, opt;
	struct th *th;
	double rd = 0, wr = 0, cp = 0;

	while ((opt = getopt(argc, argv, "s:n:Tl")) != -1) {
		switch (opt) {
		case 's': len = strtoull(optarg, NULL, 0) << 20; break;
		case 'n': iters = atoi(optarg); break;
		case 'T': no_thp = 1; break;
		case 'l': longform = 1; break;
		default:
			fprintf(stderr, "usage: membw [-s MiB] [-n iters] [-T] [-l] CPULIST\n");
			return 2;
		}
	}
	if (optind >= argc) {
		fprintf(stderr, "usage: membw [-s MiB] [-n iters] [-T] [-l] CPULIST\n");
		return 2;
	}
	n = parse_cpulist(argv[optind], cpus, 256);
	if (n <= 0)
		return 2;

	th = calloc(n, sizeof(*th));
	pthread_barrier_init(&bar, NULL, n);
	for (i = 0; i < n; i++) {
		th[i].cpu = cpus[i];
		pthread_create(&th[i].t, NULL, worker, &th[i]);
	}
	for (i = 0; i < n; i++) {
		pthread_join(th[i].t, NULL);
		rd += th[i].rd;
		wr += th[i].wr;
		cp += th[i].cp;
		if (longform)
			printf("  cpu %-3d read %6.2f  write %6.2f  copy %6.2f GB/s\n",
			       th[i].cpu, th[i].rd, th[i].wr, th[i].cp);
	}
	printf("%-14s %2d thread(s) %s: read %6.2f  write %6.2f  copy %6.2f GB/s  (copy traffic %.1f)\n",
	       argv[optind], n, no_thp ? "4K  " : "THP ", rd, wr, cp, cp * 2);
	return 0;
}
