// SPDX-License-Identifier: GPL-2.0
/*
 * dmabctl - exercise and benchmark the GB10 CPU-core DMA bridge.
 *
 *   dmabctl info
 *   dmabctl bench   [-s MiB] [-n iters] [-S] [-e engine] [-H]
 *   dmabctl udmabuf [-s MiB]                 (dma-buf import path, needs /dev/udmabuf)
 *   dmabctl cuda    [-s MiB] [-n iters] [-S] (cudaMalloc -> dma-buf -> bridge, needs libcuda)
 *   dmabctl bar -d DDDD:BB:DD.F -b BAR [-o off] [-l len] [-w]   (root)
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <linux/memfd.h>
#include <linux/udmabuf.h>

#include "../include/uapi/gb10_dmab.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#endif

static const char *devpath = "/dev/gb10_dmab";

static void die(const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	fprintf(stderr, "dmabctl: ");
	vfprintf(stderr, fmt, ap);
	fputc('\n', stderr);
	va_end(ap);
	exit(1);
}

static double now_s(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int open_dev(void)
{
	int fd = open(devpath, O_RDWR | O_CLOEXEC);

	if (fd < 0)
		die("open %s: %s (is gb10_dmab loaded?)", devpath, strerror(errno));
	return fd;
}

static void get_info(int fd, struct gb10_dmab_info *info)
{
	if (ioctl(fd, GB10_DMAB_IOC_INFO, info))
		die("INFO: %s", strerror(errno));
}

static uint32_t reg_user(int fd, const void *p, size_t len, uint32_t flags)
{
	struct gb10_dmab_reg_user r = {
		.addr = (uintptr_t)p, .len = len, .flags = flags,
	};

	if (ioctl(fd, GB10_DMAB_IOC_REG_USER, &r)) {
		if (errno == ENOMEM)
			fprintf(stderr, "hint: raise RLIMIT_MEMLOCK (ulimit -l unlimited) or run as root\n");
		die("REG_USER(%zu bytes): %s", len, strerror(errno));
	}
	return r.handle;
}

static void unreg(int fd, uint32_t h)
{
	ioctl(fd, GB10_DMAB_IOC_UNREG, &h);
}

/* synchronous copy; returns 0 or -errno; *elapsed_ns gets kernel-measured time */
static int copy_sync(int fd, uint32_t src, uint64_t soff, uint32_t dst,
		     uint64_t doff, uint64_t len, uint32_t flags, int engine,
		     uint64_t *elapsed_ns)
{
	struct gb10_dmab_copy c = {
		.src_handle = src, .dst_handle = dst,
		.src_off = soff, .dst_off = doff, .len = len,
		.flags = flags | GB10_DMAB_COPY_F_WAIT, .engine = engine,
	};

	if (ioctl(fd, GB10_DMAB_IOC_COPY, &c))
		return -errno;
	if (c.status)
		return -(int)c.status;
	if (elapsed_ns)
		*elapsed_ns = c.elapsed_ns;
	return 0;
}

static void *alloc_buf(size_t len, int huge)
{
	void *p;

	if (posix_memalign(&p, 2u << 20, len))
		die("posix_memalign(%zu)", len);
	if (huge)
		madvise(p, len, MADV_HUGEPAGE);
	memset(p, 0, len);
	return p;
}

static void fill_pattern(void *p, size_t len, uint64_t seed)
{
	uint64_t *q = p;
	size_t i, n = len / 8;

	for (i = 0; i < n; i++)
		q[i] = seed ^ (i * 0x9E3779B97F4A7C15ull);
}

static long verify_pattern(const void *p, size_t len, uint64_t seed)
{
	const uint64_t *q = p;
	size_t i, n = len / 8;

	for (i = 0; i < n; i++)
		if (q[i] != (seed ^ (i * 0x9E3779B97F4A7C15ull)))
			return (long)(i * 8);
	return -1;
}

static double gbps(uint64_t bytes, double seconds)
{
	return seconds > 0 ? bytes / seconds / 1e9 : 0;
}

static int parse_bdf(const char *s, struct gb10_dmab_pci_addr *a)
{
	unsigned int dom, bus, slot, fn;

	if (sscanf(s, "%x:%x:%x.%x", &dom, &bus, &slot, &fn) != 4 ||
	    bus > 255 || slot > 31 || fn > 7)
		return -1;
	a->domain = dom;
	a->bus = bus;
	a->devfn = (slot << 3) | fn;
	return 0;
}

/* ------------------------------------------------------------------------ */

static int cmd_info(int argc, char **argv)
{
	int fd = open_dev();
	struct gb10_dmab_info info;
	char line[256];
	FILE *f;

	get_info(fd, &info);
	printf("abi version      %u\n", info.abi_version);
	printf("engines          %u  (cpumask 0x%llx)\n", info.nr_engines,
	       (unsigned long long)info.engine_cpumask);
	printf("scheduling       %s, poll %u us, chunk %u KiB\n",
	       info.flags & GB10_DMAB_INFO_F_RT ? "SCHED_FIFO" : "SCHED_OTHER",
	       info.poll_us, info.chunk_bytes >> 10);
	printf("isolation        isolcpus=%s nohz_full=%s\n",
	       info.flags & GB10_DMAB_INFO_F_ISOLATED ? "yes" : "NO",
	       info.flags & GB10_DMAB_INFO_F_NOHZ_FULL ? "yes" : "NO");
	printf("peers            I/O %scoherent\n",
	       info.flags & GB10_DMAB_INFO_F_IO_COHERENT ? "" : "non-");
	if (info.default_peer.domain || info.default_peer.bus || info.default_peer.devfn)
		printf("default peer     %04x:%02x:%02x.%x\n", info.default_peer.domain,
		       info.default_peer.bus, info.default_peer.devfn >> 3,
		       info.default_peer.devfn & 7);
	else
		printf("default peer     none\n");
	printf("lifetime         %llu jobs, %.3f GB\n",
	       (unsigned long long)info.jobs_completed, info.bytes_copied / 1e9);

	f = fopen("/sys/class/misc/gb10_dmab/engines", "r");
	if (f) {
		printf("\n");
		while (fgets(line, sizeof(line), f))
			fputs(line, stdout);
		fclose(f);
	}
	close(fd);
	return 0;
}

static int cmd_bench(int argc, char **argv)
{
	size_t mib = 256;
	int iters = 10, stripe = 0, engine = -1, huge = 0, opt;
	int fd;
	void *src, *dst;
	uint32_t hs, hd;
	uint64_t ns, total_ns = 0, best_ns = UINT64_MAX;
	double t0, t1;
	long bad;
	int i, ret;

	while ((opt = getopt(argc, argv, "s:n:Se:H")) != -1) {
		switch (opt) {
		case 's': mib = strtoul(optarg, NULL, 0); break;
		case 'n': iters = atoi(optarg); break;
		case 'S': stripe = 1; break;
		case 'e': engine = atoi(optarg); break;
		case 'H': huge = 1; break;
		default: die("usage: bench [-s MiB] [-n iters] [-S] [-e engine] [-H]");
		}
	}
	if (!mib || iters < 1)
		die("bad size/iters");

	size_t len = mib << 20;
	fd = open_dev();
	src = alloc_buf(len, huge);
	dst = alloc_buf(len, huge);
	fill_pattern(src, len, 0x1234);

	t0 = now_s();
	hs = reg_user(fd, src, len, GB10_DMAB_REG_F_READONLY);
	hd = reg_user(fd, dst, len, 0);
	t1 = now_s();
	printf("registered 2 x %zu MiB in %.1f ms (%.1f GB/s pin rate)\n", mib,
	       (t1 - t0) * 1e3, gbps(2 * len, t1 - t0));

	ret = copy_sync(fd, hs, 0, hd, 0, len, stripe ? GB10_DMAB_COPY_F_STRIPE : 0,
			engine, &ns);		/* warm-up */
	if (ret)
		die("COPY: %s", strerror(-ret));

	t0 = now_s();
	for (i = 0; i < iters; i++) {
		ret = copy_sync(fd, hs, 0, hd, 0, len,
				stripe ? GB10_DMAB_COPY_F_STRIPE : 0, engine, &ns);
		if (ret)
			die("COPY: %s", strerror(-ret));
		total_ns += ns;
		if (ns < best_ns)
			best_ns = ns;
	}
	t1 = now_s();

	bad = verify_pattern(dst, len, 0x1234);
	printf("bridge  %s%s: %d x %zu MiB  avg %.2f GB/s  best %.2f GB/s  wall %.2f GB/s  %s\n",
	       stripe ? "striped" : "single engine",
	       engine >= 0 ? " (pinned engine)" : "", iters, mib,
	       gbps(len, total_ns / 1e9 / iters), gbps(len, best_ns / 1e9),
	       gbps((uint64_t)len * iters, t1 - t0),
	       bad < 0 ? "verify OK" : "VERIFY FAILED");
	if (bad >= 0)
		die("first mismatch at offset %ld", bad);

	/* baseline: the calling core doing the same memcpy */
	memset(dst, 0, len);
	t0 = now_s();
	for (i = 0; i < iters; i++)
		memcpy(dst, src, len);
	t1 = now_s();
	printf("memcpy  on caller core: %.2f GB/s\n", gbps((uint64_t)len * iters, t1 - t0));

	unreg(fd, hs);
	unreg(fd, hd);
	close(fd);
	free(src);
	free(dst);
	return 0;
}

static int cmd_udmabuf(int argc, char **argv)
{
	size_t mib = 64;
	int opt, fd, memfd, ufd, dbuf, ret;
	void *src, *dst, *view;
	uint32_t hs, hd, hb;
	struct udmabuf_create uc;
	struct gb10_dmab_import_dmabuf imp = { 0 };
	struct gb10_dmab_buf_info bi;
	uint64_t ns;
	long bad;

	while ((opt = getopt(argc, argv, "s:")) != -1) {
		if (opt == 's') mib = strtoul(optarg, NULL, 0);
		else die("usage: udmabuf [-s MiB]");
	}
	size_t len = mib << 20;

	memfd = memfd_create("gb10_dmab_test", MFD_ALLOW_SEALING);
	if (memfd < 0)
		die("memfd_create: %s", strerror(errno));
	if (ftruncate(memfd, len) || fcntl(memfd, F_ADD_SEALS, F_SEAL_SHRINK))
		die("memfd setup: %s", strerror(errno));

	ufd = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
	if (ufd < 0)
		die("open /dev/udmabuf: %s (needs root or a udev rule)", strerror(errno));
	memset(&uc, 0, sizeof(uc));
	uc.memfd = memfd;
	uc.flags = UDMABUF_FLAGS_CLOEXEC;
	uc.offset = 0;
	uc.size = len;
	dbuf = ioctl(ufd, UDMABUF_CREATE, &uc);
	if (dbuf < 0)
		die("UDMABUF_CREATE: %s", strerror(errno));

	fd = open_dev();
	imp.fd = dbuf;			/* peer left zero: module default */
	if (ioctl(fd, GB10_DMAB_IOC_IMPORT_DMABUF, &imp))
		die("IMPORT_DMABUF: %s", strerror(errno));
	hb = imp.handle;
	bi.handle = hb;
	if (ioctl(fd, GB10_DMAB_IOC_BUF_INFO, &bi))
		die("BUF_INFO: %s", strerror(errno));
	printf("imported udmabuf: %llu bytes, %u segment(s), type %u\n",
	       (unsigned long long)imp.size, bi.nr_segments, bi.type);

	src = alloc_buf(len, 0);
	dst = alloc_buf(len, 0);
	fill_pattern(src, len, 0xABCD);
	hs = reg_user(fd, src, len, GB10_DMAB_REG_F_READONLY);
	hd = reg_user(fd, dst, len, 0);

	ret = copy_sync(fd, hs, 0, hb, 0, len, GB10_DMAB_COPY_F_STRIPE, -1, &ns);
	if (ret)
		die("COPY user->dmabuf: %s", strerror(-ret));
	printf("user -> dma-buf  %zu MiB in %.2f ms (%.2f GB/s)\n", mib, ns / 1e6,
	       gbps(len, ns / 1e9));

	view = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, 0);
	if (view == MAP_FAILED)
		die("mmap memfd: %s", strerror(errno));
	bad = verify_pattern(view, len, 0xABCD);
	printf("verify via memfd mapping: %s\n", bad < 0 ? "OK" : "FAILED");
	if (bad >= 0)
		die("first mismatch at %ld", bad);

	fill_pattern(view, len, 0x5EED);
	ret = copy_sync(fd, hb, 0, hd, 0, len, GB10_DMAB_COPY_F_STRIPE, -1, &ns);
	if (ret)
		die("COPY dmabuf->user: %s", strerror(-ret));
	bad = verify_pattern(dst, len, 0x5EED);
	printf("dma-buf -> user  %zu MiB in %.2f ms (%.2f GB/s), verify %s\n", mib,
	       ns / 1e6, gbps(len, ns / 1e9), bad < 0 ? "OK" : "FAILED");
	if (bad >= 0)
		die("first mismatch at %ld", bad);

	munmap(view, len);
	unreg(fd, hs);
	unreg(fd, hd);
	unreg(fd, hb);
	close(fd);
	close(dbuf);
	close(ufd);
	close(memfd);
	return 0;
}

#ifdef HAVE_CUDA
static void cu_check(CUresult r, const char *what)
{
	const char *s = "?";

	if (r == CUDA_SUCCESS)
		return;
	cuGetErrorString(r, &s);
	die("%s: %s (%d)", what, s, (int)r);
}

static int cmd_cuda(int argc, char **argv)
{
	size_t mib = 256;
	int iters = 10, stripe = 1, opt, fd, dbuf = -1, supported = 0, i, ret;
	CUdevice dev;
	CUcontext cuctx;
	CUdeviceptr raw = 0, dptr = 0;
	void *host, *host2, *gpuvis = NULL;
	uint32_t hs, hd, hg;
	struct gb10_dmab_import_dmabuf imp = { 0 };
	struct gb10_dmab_buf_info bi;
	uint64_t ns, total = 0;
	double t0, t1;
	long bad;

	while ((opt = getopt(argc, argv, "s:n:S1")) != -1) {
		switch (opt) {
		case 's': mib = strtoul(optarg, NULL, 0); break;
		case 'n': iters = atoi(optarg); break;
		case 'S': stripe = 1; break;
		case '1': stripe = 0; break;
		default: die("usage: cuda [-s MiB] [-n iters] [-S|-1]");
		}
	}
	size_t len = mib << 20;
	uint32_t cflags = stripe ? GB10_DMAB_COPY_F_STRIPE : 0;

	cu_check(cuInit(0), "cuInit");
	cu_check(cuDeviceGet(&dev, 0), "cuDeviceGet");
	cu_check(cuDevicePrimaryCtxRetain(&cuctx, dev), "cuDevicePrimaryCtxRetain");
	cu_check(cuCtxSetCurrent(cuctx), "cuCtxSetCurrent");
	cu_check(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev),
		 "cuDeviceGetAttribute");

	fd = open_dev();
	host = alloc_buf(len, 1);
	host2 = alloc_buf(len, 1);
	fill_pattern(host, len, 0xC0DA);
	hs = reg_user(fd, host, len, GB10_DMAB_REG_F_READONLY);
	hd = reg_user(fd, host2, len, 0);

	if (supported) {
		/* real GPU allocation exported as a dma-buf */
		cu_check(cuMemAlloc(&raw, len + (2u << 20)), "cuMemAlloc");
		dptr = (raw + (2u << 20) - 1) & ~(CUdeviceptr)((2u << 20) - 1);
		cu_check(cuMemsetD8(dptr, 0, len), "cuMemsetD8");
		cu_check(cuMemGetHandleForAddressRange(&dbuf, dptr, len,
						       CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0),
			 "cuMemGetHandleForAddressRange");
		imp.fd = dbuf;			/* default peer: the GPU itself */
		if (ioctl(fd, GB10_DMAB_IOC_IMPORT_DMABUF, &imp))
			die("IMPORT_DMABUF(cuda): %s", strerror(errno));
		hg = imp.handle;
		bi.handle = hg;
		if (ioctl(fd, GB10_DMAB_IOC_BUF_INFO, &bi))
			die("BUF_INFO: %s", strerror(errno));
		printf("GPU endpoint: cudaMalloc dma-buf, %llu bytes in %u segment(s)\n",
		       (unsigned long long)imp.size, bi.nr_segments);
	} else {
		/*
		 * GB10 with driver 580 / CUDA 13: no dma-buf export, no GPUDirect
		 * RDMA, and cudaMalloc() memory has no CPU page tables. The GPU
		 * addresses system memory through ATS instead, so the GPU-visible
		 * endpoint is ordinary system memory registered as a USER buffer.
		 */
		printf("GPU endpoint: system memory (CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED=0; GPU reaches it via ATS)\n");
		gpuvis = alloc_buf(len, 1);
		hg = reg_user(fd, gpuvis, len, 0);
		dptr = (CUdeviceptr)(uintptr_t)gpuvis;
	}

	/* host -> GPU endpoint via the bridge */
	ret = copy_sync(fd, hs, 0, hg, 0, len, cflags, -1, &ns);
	if (ret)
		die("COPY host->gpu: %s", strerror(-ret));
	t0 = now_s();
	for (i = 0; i < iters; i++) {
		ret = copy_sync(fd, hs, 0, hg, 0, len, cflags, -1, &ns);
		if (ret)
			die("COPY host->gpu: %s", strerror(-ret));
		total += ns;
	}
	t1 = now_s();
	printf("bridge host -> GPU endpoint %s: avg %.2f GB/s, wall %.2f GB/s\n",
	       stripe ? "striped" : "single", gbps(len, total / 1e9 / iters),
	       gbps((uint64_t)len * iters, t1 - t0));

	/* the GPU (copy engine) reads it back: proves the GPU sees what the bridge wrote */
	memset(host2, 0, len);
	cu_check(cuMemcpy((CUdeviceptr)(uintptr_t)host2, dptr, len), "cuMemcpy(GPU endpoint -> host)");
	cu_check(cuCtxSynchronize(), "cuCtxSynchronize");
	bad = verify_pattern(host2, len, 0xC0DA);
	printf("verify via GPU copy engine readback: %s\n", bad < 0 ? "OK" : "FAILED");
	if (bad >= 0)
		die("first mismatch at %ld", bad);

	/* the GPU writes a new pattern, the bridge pulls it out */
	fill_pattern(host2, len, 0x6B10);
	cu_check(cuMemcpy(dptr, (CUdeviceptr)(uintptr_t)host2, len), "cuMemcpy(host -> GPU endpoint)");
	cu_check(cuCtxSynchronize(), "cuCtxSynchronize");
	memset(host2, 0, len);
	ret = copy_sync(fd, hg, 0, hd, 0, len, cflags, -1, &ns);
	if (ret)
		die("COPY gpu->host: %s", strerror(-ret));
	bad = verify_pattern(host2, len, 0x6B10);
	printf("bridge GPU endpoint -> host   %.2f GB/s, verify %s\n", gbps(len, ns / 1e9),
	       bad < 0 ? "OK" : "FAILED");
	if (bad >= 0)
		die("first mismatch at %ld", bad);

	/* baseline: the GPU's own copy engine doing the same move */
	t0 = now_s();
	for (i = 0; i < iters; i++)
		cu_check(cuMemcpy(dptr, (CUdeviceptr)(uintptr_t)host, len), "cuMemcpy");
	cu_check(cuCtxSynchronize(), "cuCtxSynchronize");
	t1 = now_s();
	printf("GPU copy engine (cuMemcpy) baseline: %.2f GB/s\n",
	       gbps((uint64_t)len * iters, t1 - t0));

	unreg(fd, hs);
	unreg(fd, hd);
	unreg(fd, hg);
	close(fd);
	if (dbuf >= 0)
		close(dbuf);
	if (raw)
		cuMemFree(raw);
	free(gpuvis);
	cuDevicePrimaryCtxRelease(dev);
	free(host);
	free(host2);
	return 0;
}
#endif

static int cmd_bar(int argc, char **argv)
{
	struct gb10_dmab_map_bar mb = { 0 };
	const char *bdf = NULL;
	int bar = -1, wc = 0, opt, fd, ret;
	uint64_t off = 0, len = 256, ns;
	void *buf;
	uint32_t hb, hu;
	uint64_t i;

	while ((opt = getopt(argc, argv, "d:b:o:l:w")) != -1) {
		switch (opt) {
		case 'd': bdf = optarg; break;
		case 'b': bar = atoi(optarg); break;
		case 'o': off = strtoull(optarg, NULL, 0); break;
		case 'l': len = strtoull(optarg, NULL, 0); break;
		case 'w': wc = 1; break;
		default: die("usage: bar -d DDDD:BB:DD.F -b BAR [-o off] [-l len] [-w]");
		}
	}
	if (!bdf || bar < 0 || parse_bdf(bdf, &mb.dev))
		die("usage: bar -d DDDD:BB:DD.F -b BAR [-o off] [-l len] [-w]");
	mb.bar = bar;
	mb.offset = off;
	mb.len = len;
	mb.flags = wc ? GB10_DMAB_BAR_F_WC : 0;

	fd = open_dev();
	if (ioctl(fd, GB10_DMAB_IOC_MAP_BAR, &mb))
		die("MAP_BAR: %s", strerror(errno));
	hb = mb.handle;
	printf("%s BAR%d: size 0x%llx, window [0x%llx, +0x%llx)%s\n", bdf, bar,
	       (unsigned long long)mb.bar_size, (unsigned long long)off,
	       (unsigned long long)len, wc ? " write-combining" : "");

	buf = alloc_buf((len + 4095) & ~4095ull, 0);
	hu = reg_user(fd, buf, len, 0);
	ret = copy_sync(fd, hb, 0, hu, 0, len, 0, -1, &ns);
	if (ret)
		die("COPY bar->user: %s", strerror(-ret));
	printf("read %llu bytes in %.1f us\n", (unsigned long long)len, ns / 1e3);
	for (i = 0; i < len; i += 16) {
		uint64_t j;

		printf("%08llx:", (unsigned long long)(off + i));
		for (j = i; j < i + 16 && j < len; j++)
			printf(" %02x", ((uint8_t *)buf)[j]);
		printf("\n");
	}
	unreg(fd, hu);
	unreg(fd, hb);
	close(fd);
	free(buf);
	return 0;
}

static void usage(void)
{
	fprintf(stderr,
		"usage: dmabctl [-D /dev/gb10_dmab] <command> [options]\n"
		"  info                               engine summary\n"
		"  bench [-s MiB] [-n N] [-S] [-e E] [-H]   host->host copy through the bridge\n"
		"  udmabuf [-s MiB]                   dma-buf import round trip via /dev/udmabuf\n"
#ifdef HAVE_CUDA
		"  cuda [-s MiB] [-n N] [-S|-1]       GPU-visible memory <-> host through the bridge, verified by the GPU\n"
#endif
		"  bar -d BDF -b BAR [-o off] [-l len] [-w]   dump a PCI BAR window (root)\n");
	exit(2);
}

int main(int argc, char **argv)
{
	if (argc >= 3 && !strcmp(argv[1], "-D")) {
		devpath = argv[2];
		argv += 2;
		argc -= 2;
	}
	if (argc < 2)
		usage();
	optind = 2;
	if (!strcmp(argv[1], "info"))
		return cmd_info(argc, argv);
	if (!strcmp(argv[1], "bench"))
		return cmd_bench(argc, argv);
	if (!strcmp(argv[1], "udmabuf"))
		return cmd_udmabuf(argc, argv);
#ifdef HAVE_CUDA
	if (!strcmp(argv[1], "cuda"))
		return cmd_cuda(argc, argv);
#endif
	if (!strcmp(argv[1], "bar"))
		return cmd_bar(argc, argv);
	usage();
	return 2;
}
