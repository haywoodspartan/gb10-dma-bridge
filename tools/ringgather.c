// SPDX-License-Identifier: GPL-2.0
/*
 * ringgather - combine the neighbouring GB10 nodes into the head: their GPUs
 *              produce, every ConnectX-7 path carries, the head's GPU consumes.
 *
 * A DGX Spark's ConnectX-7 hangs off two PCIe Gen5 x4 links, and each link
 * exposes both QSFP ports. Into one node each x4 link takes 120.6 Gb/s no
 * matter which port feeds it, so the most a node can receive is 241 Gb/s
 * (30.1 GB/s), and only when both links are busy. One neighbour can deliver
 * 196 Gb/s (its cable's line rate); two neighbours, each on both of its
 * PCIe halves, fill both links of the head. That is what this does.
 *
 * Worker (every neighbour):   ringgather --worker
 * Head:                       ringgather WORKER_IP [WORKER_IP ...]
 *
 * WORKER_IP is a worker's address on a ring link, one per link to use; for a
 * three-node ring that is four addresses, two per neighbour (one per PCIe
 * half). The head opens one RC connection per address and then, per gather:
 *
 *   1. splits its destination buffer into one slice per link,
 *   2. FILL: asks each worker to produce its slice; the worker's GPU writes
 *      the data into NIC-registered system memory and acks with the time,
 *   3. PUSH: tells every worker where its slice lives in the head's memory;
 *      the workers RDMA-WRITE in chunks, the last one with an immediate,
 *   4. returns when every slice has landed. The head's GPU then reads the
 *      whole buffer straight from the NIC-written pages (ATS, no copy).
 *
 * The slices carry one continuous pattern (word i = seed ^ i * K over the
 * whole destination), so a single verify pass on the head proves that the
 * GB10s together produced one coherent buffer.
 *
 * The four functions rg_open/rg_bind/rg_gather/rg_close are the reusable
 * part; main() is a bandwidth test built on them. On the worker, produce()
 * is the one place a real workload would put its own output.
 *
 * Head options:   -s MiB total (4096)  -n iters (5)  -c chunk MiB (1)
 *                 -q writes in flight per link (32)  -p port (18520)
 *                 -w secs to wait for the workers (60)  -C CPU fill/verify
 * Worker options: -p port  -k N exit after N connections have closed
 *                 -i secs exit after that long without a connection  -C CPU fill
 *
 * There is no authentication: run it on the private ring only.
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <getopt.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <cuda.h>
#ifdef HAVE_NVRTC
#include <nvrtc.h>
#endif

#define RG_MAGIC	0x31484752u	/* "RGH1" */
#define RG_MAX_LINKS	16
#define RG_MAX_DEPTH	256
#define RG_FILL		1
#define RG_PUSH		2
#define RG_MSG_SIZE	64
#define HUGE_2M		((size_t)2 << 20)
#define PATTERN_K	0x9E3779B97F4A7C15ull
#define ITER_SEED(base, i)	((base) ^ ((uint64_t)(i) * 0x0123456789ABCDEFull))

static struct {
	int worker;
	size_t len;
	unsigned int iters;
	size_t chunk;
	unsigned int depth;
	int port;
	int cpu_only;
	int connect_wait;
	int serve;		/* worker: exit after this many connections closed */
	int idle;		/* worker: exit if nobody connects for this long */
	size_t max_len;		/* worker: largest slice it will stage */
} opt = {
	.len = 4096ull << 20, .iters = 5, .chunk = 1ull << 20, .depth = 32,
	.port = 18520, .connect_wait = 60, .max_len = 64ull << 30,
};

/* CM private data, both directions */
struct rg_hello {
	uint32_t magic;
	char host[32];
} __attribute__((packed));

/* head -> worker, one inline SEND per request */
struct rg_req {
	uint32_t op;		/* RG_FILL, RG_PUSH */
	uint32_t iter;
	uint32_t chunk;		/* PUSH: bytes per RDMA WRITE */
	uint32_t depth;		/* PUSH: writes in flight */
	uint64_t len;		/* bytes in this link's slice */
	uint64_t word_base;	/* FILL: index of the slice's first 64-bit word in the whole buffer */
	uint64_t seed;		/* FILL */
	uint64_t addr;		/* PUSH: where the slice lands in the head's memory */
	uint32_t rkey;		/* PUSH */
	uint32_t pad;
} __attribute__((packed));

static double now_s(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double gbps(uint64_t bytes, double s)
{
	return s > 0 ? bytes / s / 1e9 : 0;
}

/* system memory the NIC and the GPU both address: 2 MiB aligned, THP hinted, populated */
static void *alloc_sysmem(size_t len)
{
	size_t sz = (len + HUGE_2M - 1) & ~(HUGE_2M - 1);
	void *p = aligned_alloc(HUGE_2M, sz);

	if (!p)
		return NULL;
	madvise(p, sz, MADV_HUGEPAGE);
	memset(p, 0, sz);
	return p;
}

/* ------------------------------------------------------------------------
 * GPU: fill / verify kernels compiled at runtime, running on system memory
 * ------------------------------------------------------------------------ */

static struct {
	int ok;
	int pageable;
	CUdevice dev;
	CUcontext ctx;
	CUmodule mod;
	CUfunction fill, verify;
} G;

struct gpu_thread {
	CUstream stream;
	CUdeviceptr bad;
	CUevent e0, e1;
};

#ifdef HAVE_NVRTC
static const char *kernel_src =
"extern \"C\" __global__ void fill(unsigned long long *p, unsigned long long n,\n"
"                                  unsigned long long seed, unsigned long long base)\n"
"{\n"
"    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;\n"
"         i < n; i += (unsigned long long)gridDim.x * blockDim.x)\n"
"        p[i] = seed ^ ((base + i) * 0x9E3779B97F4A7C15ull);\n"
"}\n"
"extern \"C\" __global__ void verify(const unsigned long long *p, unsigned long long n,\n"
"                                    unsigned long long seed, unsigned long long base,\n"
"                                    unsigned long long *bad)\n"
"{\n"
"    unsigned long long cnt = 0;\n"
"    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;\n"
"         i < n; i += (unsigned long long)gridDim.x * blockDim.x)\n"
"        if (p[i] != (seed ^ ((base + i) * 0x9E3779B97F4A7C15ull)))\n"
"            cnt++;\n"
"    if (cnt)\n"
"        atomicAdd(bad, cnt);\n"
"}\n";

static char *nvrtc_build(const char *arch)
{
	nvrtcProgram prog;
	const char *opts[] = { arch };
	size_t sz;
	char *ptx;

	if (nvrtcCreateProgram(&prog, kernel_src, "ringgather.cu", 0, NULL, NULL) != NVRTC_SUCCESS)
		return NULL;
	if (nvrtcCompileProgram(prog, 1, opts) != NVRTC_SUCCESS) {
		if (nvrtcGetProgramLogSize(prog, &sz) == NVRTC_SUCCESS && sz > 1) {
			char *log = malloc(sz);

			nvrtcGetProgramLog(prog, log);
			fprintf(stderr, "nvrtc (%s): %s\n", arch, log);
			free(log);
		}
		nvrtcDestroyProgram(&prog);
		return NULL;
	}
	nvrtcGetPTXSize(prog, &sz);
	ptx = malloc(sz);
	nvrtcGetPTX(prog, ptx);
	nvrtcDestroyProgram(&prog);
	return ptx;
}
#endif

static const char *cu_err(CUresult r)
{
	const char *s = "?";

	cuGetErrorString(r, &s);
	return s;
}

static int gpu_init(void)
{
	CUresult r;
	int major = 0, minor = 0;

	r = cuInit(0);
	if (r != CUDA_SUCCESS) {
		fprintf(stderr, "cuInit: %s; using the CPU\n", cu_err(r));
		return -1;
	}
	cuDeviceGet(&G.dev, 0);
	cuDeviceGetAttribute(&G.pageable, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, G.dev);
	cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, G.dev);
	cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, G.dev);
	if (!G.pageable) {
		fprintf(stderr, "GPU cannot address pageable system memory; using the CPU\n");
		return -1;
	}
	r = cuDevicePrimaryCtxRetain(&G.ctx, G.dev);
	if (r != CUDA_SUCCESS) {
		fprintf(stderr, "cuDevicePrimaryCtxRetain: %s; using the CPU\n", cu_err(r));
		return -1;
	}
	cuCtxSetCurrent(G.ctx);

#ifdef HAVE_NVRTC
	{
		char native[48];
		const char *archs[] = { native, "--gpu-architecture=compute_120",
					"--gpu-architecture=compute_90", NULL };
		char *ptx = NULL;
		int i;

		snprintf(native, sizeof(native), "--gpu-architecture=compute_%d%d", major, minor);
		for (i = 0; archs[i] && !ptx; i++)
			ptx = nvrtc_build(archs[i]);
		if (!ptx) {
			fprintf(stderr, "no kernel could be built; using the CPU\n");
			return -1;
		}
		r = cuModuleLoadData(&G.mod, ptx);
		free(ptx);
		if (r != CUDA_SUCCESS ||
		    cuModuleGetFunction(&G.fill, G.mod, "fill") != CUDA_SUCCESS ||
		    cuModuleGetFunction(&G.verify, G.mod, "verify") != CUDA_SUCCESS) {
			fprintf(stderr, "kernel load failed: %s; using the CPU\n", cu_err(r));
			return -1;
		}
		G.ok = 1;
		printf("GPU: compute %d.%d, fill/verify kernels built, system memory via ATS\n",
		       major, minor);
		return 0;
	}
#else
	fprintf(stderr, "built without NVRTC; using the CPU for fill/verify\n");
	return -1;
#endif
}

static int gpu_thread_init(struct gpu_thread *t)
{
	if (!G.ok)
		return -1;
	cuCtxSetCurrent(G.ctx);
	if (cuStreamCreate(&t->stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS ||
	    cuMemAlloc(&t->bad, 8) != CUDA_SUCCESS ||
	    cuEventCreate(&t->e0, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
	    cuEventCreate(&t->e1, CU_EVENT_DEFAULT) != CUDA_SUCCESS)
		return -1;
	return 0;
}

static unsigned int grid_for(size_t n)
{
	size_t blocks = (n + 255) / 256;

	return blocks > 8192 ? 8192 : (unsigned int)blocks;
}

/* returns ms, < 0 on error */
static double gpu_fill(struct gpu_thread *t, void *buf, size_t len, uint64_t seed, uint64_t base)
{
	unsigned long long n = len / 8, s = seed, b = base;
	void *args[] = { &buf, &n, &s, &b };
	float ms;

	cuCtxSetCurrent(G.ctx);
	cuEventRecord(t->e0, t->stream);
	if (cuLaunchKernel(G.fill, grid_for(n), 1, 1, 256, 1, 1, 0, t->stream, args, NULL) != CUDA_SUCCESS)
		return -1;
	cuEventRecord(t->e1, t->stream);
	if (cuStreamSynchronize(t->stream) != CUDA_SUCCESS)
		return -1;
	cuEventElapsedTime(&ms, t->e0, t->e1);
	return ms;
}

/* returns mismatching words, < 0 on error; *ms gets kernel time */
static long long gpu_verify(struct gpu_thread *t, void *buf, size_t len, uint64_t seed,
			    uint64_t base, double *ms)
{
	unsigned long long n = len / 8, s = seed, b = base, bad = 0;
	void *args[] = { &buf, &n, &s, &b, &t->bad };
	float fms;

	cuCtxSetCurrent(G.ctx);
	cuMemsetD8Async(t->bad, 0, 8, t->stream);
	cuEventRecord(t->e0, t->stream);
	if (cuLaunchKernel(G.verify, grid_for(n), 1, 1, 256, 1, 1, 0, t->stream, args, NULL) != CUDA_SUCCESS)
		return -1;
	cuEventRecord(t->e1, t->stream);
	if (cuStreamSynchronize(t->stream) != CUDA_SUCCESS)
		return -1;
	cuMemcpyDtoH(&bad, t->bad, 8);
	cuEventElapsedTime(&fms, t->e0, t->e1);
	*ms = fms;
	return (long long)bad;
}

static void cpu_fill(void *buf, size_t len, uint64_t seed, uint64_t base)
{
	uint64_t *p = buf;
	size_t i, n = len / 8;

	for (i = 0; i < n; i++)
		p[i] = seed ^ ((base + i) * PATTERN_K);
}

static long long cpu_verify(const void *buf, size_t len, uint64_t seed, uint64_t base)
{
	const uint64_t *p = buf;
	size_t i, n = len / 8;
	long long bad = 0;

	for (i = 0; i < n; i++)
		if (p[i] != (seed ^ ((base + i) * PATTERN_K)))
			bad++;
	return bad;
}

/* ------------------------------------------------------------------------
 * small verbs helpers shared by both sides
 * ------------------------------------------------------------------------ */

/* zero-length receive: catches a SEND_WITH_IMM or the immediate of an RDMA WRITE */
static int post_recv0(struct ibv_qp *qp)
{
	struct ibv_recv_wr wr, *bad;

	memset(&wr, 0, sizeof(wr));
	return ibv_post_recv(qp, &wr, &bad);
}

static int post_send_imm0(struct ibv_qp *qp, uint32_t imm)
{
	struct ibv_send_wr wr, *bad;

	memset(&wr, 0, sizeof(wr));
	wr.opcode = IBV_WR_SEND_WITH_IMM;
	wr.send_flags = IBV_SEND_SIGNALED;
	wr.imm_data = htonl(imm);
	return ibv_post_send(qp, &wr, &bad);
}

static void conn_params(struct rdma_conn_param *cp, const struct rg_hello *h)
{
	memset(cp, 0, sizeof(*cp));
	cp->private_data = h;
	cp->private_data_len = sizeof(*h);
	cp->responder_resources = 1;
	cp->initiator_depth = 1;
	cp->retry_count = 7;
	cp->rnr_retry_count = 7;
}

/* ------------------------------------------------------------------------
 * head: rg_open / rg_bind / rg_gather / rg_close
 * ------------------------------------------------------------------------ */

struct rg_link {
	char ip[48];			/* the worker's address on this link */
	char host[33];			/* the worker's hostname */
	char dev[IBV_SYSFS_NAME_MAX];	/* local RoCE device the link runs on */
	struct rdma_event_channel *ec;
	struct rdma_cm_id *id;
	struct ibv_pd *pd;
	struct ibv_cq *cq;
	struct ibv_mr *mr;
	uint8_t *slice;			/* this link's part of the destination */
	size_t len;
	uint64_t word_base;
	int recvs;			/* zero-length receives posted and not yet consumed */
};

struct rg_stats {
	double fill_ms[RG_MAX_LINKS];	/* each worker's own time to produce its slice */
	double land_ms[RG_MAX_LINKS];	/* PUSH sent -> slice complete, per link */
	double produce_ms;		/* FILL sent -> every worker ready, on the head's clock */
	double wire_ms;			/* PUSH sent -> every slice landed, on the head's clock */
};

struct ring_gather {
	int n;
	size_t len, chunk;
	unsigned int depth;
	double timeout_s;
	struct rg_link l[RG_MAX_LINKS];
};

static void rg_link_reset(struct rg_link *l)
{
	if (l->id && l->id->qp)
		rdma_destroy_qp(l->id);
	if (l->mr)
		ibv_dereg_mr(l->mr);
	if (l->cq)
		ibv_destroy_cq(l->cq);
	if (l->pd)
		ibv_dealloc_pd(l->pd);
	if (l->id)
		rdma_destroy_id(l->id);
	l->id = NULL;
	l->mr = NULL;
	l->cq = NULL;
	l->pd = NULL;
	l->recvs = 0;
}

static int rg_link_verbs(struct rg_link *l)
{
	struct ibv_qp_init_attr qa;

	l->pd = ibv_alloc_pd(l->id->verbs);
	l->cq = l->pd ? ibv_create_cq(l->id->verbs, 64, NULL, NULL, 0) : NULL;
	if (!l->cq)
		return -1;
	memset(&qa, 0, sizeof(qa));
	qa.send_cq = l->cq;
	qa.recv_cq = l->cq;
	qa.qp_type = IBV_QPT_RC;
	qa.cap.max_send_wr = 8;
	qa.cap.max_recv_wr = 16;
	qa.cap.max_send_sge = 1;
	qa.cap.max_recv_sge = 1;
	qa.cap.max_inline_data = RG_MSG_SIZE;
	return rdma_create_qp(l->id, l->pd, &qa);
}

/* connect one link; the worker may still be starting, so retry until wait_s is over */
static int rg_link_connect(struct rg_link *l, const char *ip, int port, int wait_s)
{
	struct sockaddr_in sin;
	struct rg_hello h = { .magic = RG_MAGIC }, peer;
	struct rdma_conn_param cp;
	struct rdma_cm_event *ev;
	double deadline = now_s() + wait_s;
	int attempts = 0;

	snprintf(l->ip, sizeof(l->ip), "%s", ip);
	memset(&sin, 0, sizeof(sin));
	sin.sin_family = AF_INET;
	sin.sin_port = htons(port);
	if (inet_pton(AF_INET, ip, &sin.sin_addr) != 1) {
		fprintf(stderr, "[head] bad worker address %s\n", ip);
		return -1;
	}
	gethostname(h.host, sizeof(h.host) - 1);
	l->ec = rdma_create_event_channel();
	if (!l->ec)
		goto sys;

	for (;;) {
		int retry = 0;

		attempts++;
		if (rdma_create_id(l->ec, &l->id, NULL, RDMA_PS_TCP) ||
		    rdma_resolve_addr(l->id, NULL, (struct sockaddr *)&sin, 2000) ||
		    rdma_get_cm_event(l->ec, &ev))
			goto sys;
		if (ev->event != RDMA_CM_EVENT_ADDR_RESOLVED) {
			retry = 1;
		} else {
			rdma_ack_cm_event(ev);
			if (rdma_resolve_route(l->id, 2000) || rdma_get_cm_event(l->ec, &ev))
				goto sys;
			if (ev->event != RDMA_CM_EVENT_ROUTE_RESOLVED)
				retry = 1;
		}
		if (!retry) {
			rdma_ack_cm_event(ev);
			if (rg_link_verbs(l))
				goto sys;
			conn_params(&cp, &h);
			if (rdma_connect(l->id, &cp) || rdma_get_cm_event(l->ec, &ev))
				goto sys;
			if (ev->event == RDMA_CM_EVENT_ESTABLISHED)
				break;
		}
		if (now_s() > deadline || (ev->event != RDMA_CM_EVENT_REJECTED &&
					   ev->event != RDMA_CM_EVENT_UNREACHABLE &&
					   ev->event != RDMA_CM_EVENT_CONNECT_ERROR &&
					   ev->event != RDMA_CM_EVENT_ADDR_ERROR &&
					   ev->event != RDMA_CM_EVENT_ROUTE_ERROR)) {
			fprintf(stderr, "[head] giving up on %s after %d attempt(s): %s (status %d)\n",
				ip, attempts, rdma_event_str(ev->event), ev->status);
			rdma_ack_cm_event(ev);
			return -1;
		}
		if (attempts == 1)
			printf("[head] %s not listening yet (%s), retrying for up to %ds\n", ip,
			       rdma_event_str(ev->event), wait_s);
		rdma_ack_cm_event(ev);
		rg_link_reset(l);
		usleep(300000);
	}

	memset(&peer, 0, sizeof(peer));
	if (ev->param.conn.private_data_len >= sizeof(peer))
		memcpy(&peer, ev->param.conn.private_data, sizeof(peer));
	rdma_ack_cm_event(ev);
	if (peer.magic != RG_MAGIC) {
		fprintf(stderr, "[head] %s is not a ringgather worker\n", ip);
		return -1;
	}
	memcpy(l->host, peer.host, sizeof(peer.host));
	snprintf(l->dev, sizeof(l->dev), "%s", ibv_get_device_name(l->id->verbs->device));
	return 0;
sys:
	fprintf(stderr, "[head] connecting to %s: %s\n", ip, strerror(errno));
	return -1;
}

static void rg_close(struct ring_gather *g)
{
	int k;

	if (!g)
		return;
	for (k = 0; k < g->n; k++) {
		struct rg_link *l = &g->l[k];
		struct rdma_cm_event *ev;

		if (l->id && l->id->qp && !rdma_disconnect(l->id) &&
		    !rdma_get_cm_event(l->ec, &ev))
			rdma_ack_cm_event(ev);
		rg_link_reset(l);
		if (l->ec)
			rdma_destroy_event_channel(l->ec);
	}
	free(g);
}

/*
 * Connect to the workers, one link per address. Give every path that should
 * carry data: for the full 30 GB/s into a node that is both PCIe halves of
 * both neighbours.
 */
static struct ring_gather *rg_open(char **worker_ips, int n, int port, int wait_s)
{
	struct ring_gather *g;
	int k;

	if (n < 1 || n > RG_MAX_LINKS) {
		fprintf(stderr, "[head] 1 to %d links\n", RG_MAX_LINKS);
		return NULL;
	}
	g = calloc(1, sizeof(*g));
	if (!g)
		return NULL;
	g->depth = 32;
	g->timeout_s = 60;
	for (k = 0; k < n; k++) {
		g->n = k + 1;
		if (rg_link_connect(&g->l[k], worker_ips[k], port, wait_s)) {
			rg_close(g);
			return NULL;
		}
	}
	return g;
}

/*
 * Make dst[0..len) the destination: one contiguous slice per link, registered
 * on that link's device only. len must be a multiple of chunk. Call again to
 * move to another buffer.
 */
static int rg_bind(struct ring_gather *g, void *dst, size_t len, size_t chunk)
{
	size_t per = len / chunk / g->n * chunk, off = 0;
	int k;

	if (!per || len % chunk || chunk % 8 || chunk > (1u << 30)) {
		fprintf(stderr, "[head] size must be a multiple of the chunk and give every link at least one\n");
		return -1;
	}
	g->len = len;
	g->chunk = chunk;
	for (k = 0; k < g->n; k++) {
		struct rg_link *l = &g->l[k];

		if (l->mr)
			ibv_dereg_mr(l->mr);
		l->slice = (uint8_t *)dst + off;
		l->len = k == g->n - 1 ? len - off : per;
		l->word_base = off / 8;
		l->mr = ibv_reg_mr(l->pd, l->slice, l->len,
				   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
		if (!l->mr) {
			fprintf(stderr, "[head] ibv_reg_mr on %s: %s (raise ulimit -l?)\n", l->dev,
				strerror(errno));
			return -1;
		}
		off += l->len;
	}
	return 0;
}

static int rg_send_req(struct rg_link *l, const struct rg_req *rq)
{
	struct ibv_send_wr wr, *bad;
	struct ibv_sge sge;

	while (l->recvs < 2) {
		if (post_recv0(l->id->qp))
			return -1;
		l->recvs++;
	}
	sge.addr = (uintptr_t)rq;
	sge.length = sizeof(*rq);
	sge.lkey = 0;
	memset(&wr, 0, sizeof(wr));
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.opcode = IBV_WR_SEND;
	wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
	return ibv_post_send(l->id->qp, &wr, &bad);
}

/* busy-poll every link until each has delivered one completion of kind `want` */
static int rg_wait(struct ring_gather *g, enum ibv_wc_opcode want, double t0, double *out)
{
	char got[RG_MAX_LINKS] = { 0 };
	unsigned int spins = 0;
	int left = g->n, k;

	while (left) {
		for (k = 0; k < g->n; k++) {
			struct ibv_wc wc[4];
			int i, n = ibv_poll_cq(g->l[k].cq, 4, wc);

			if (n < 0) {
				fprintf(stderr, "[head] ibv_poll_cq on %s failed\n", g->l[k].dev);
				return -1;
			}
			for (i = 0; i < n; i++) {
				if (wc[i].status != IBV_WC_SUCCESS) {
					fprintf(stderr, "[head] %s (%s): %s\n", g->l[k].host,
						g->l[k].dev, ibv_wc_status_str(wc[i].status));
					return -1;
				}
				if (wc[i].opcode == IBV_WC_SEND)
					continue;	/* our own request went out */
				g->l[k].recvs--;
				if (wc[i].opcode != want || got[k]) {
					fprintf(stderr, "[head] %s (%s): unexpected completion %d\n",
						g->l[k].host, g->l[k].dev, wc[i].opcode);
					return -1;
				}
				out[k] = want == IBV_WC_RECV ? ntohl(wc[i].imm_data) / 1e3
							     : (now_s() - t0) * 1e3;
				got[k] = 1;
				left--;
			}
		}
		if (!(++spins & 0xffff) && now_s() - t0 > g->timeout_s) {
			fprintf(stderr, "[head] no answer from the workers for %.0f s\n", g->timeout_s);
			return -1;
		}
	}
	return 0;
}

/*
 * One gather: every worker produces its slice of the bound buffer (pattern
 * `seed`), then all of them push at once. Returns 0 when every byte of
 * dst[0..len) has landed; the buffer is then ready for the head's GPU.
 */
static int rg_gather(struct ring_gather *g, uint64_t seed, uint32_t iter, struct rg_stats *st)
{
	struct rg_req rq;
	double t0, t1;
	int k;

	memset(&rq, 0, sizeof(rq));
	rq.iter = iter;
	rq.seed = seed;
	rq.chunk = (uint32_t)g->chunk;
	rq.depth = g->depth;

	rq.op = RG_FILL;
	t0 = now_s();
	for (k = 0; k < g->n; k++) {
		rq.len = g->l[k].len;
		rq.word_base = g->l[k].word_base;
		if (rg_send_req(&g->l[k], &rq))
			goto sys;
	}
	if (rg_wait(g, IBV_WC_RECV, t0, st->fill_ms))
		return -1;
	t1 = now_s();
	st->produce_ms = (t1 - t0) * 1e3;

	rq.op = RG_PUSH;
	for (k = 0; k < g->n; k++) {
		rq.len = g->l[k].len;
		rq.addr = (uintptr_t)g->l[k].slice;
		rq.rkey = g->l[k].mr->rkey;
		if (rg_send_req(&g->l[k], &rq))
			goto sys;
	}
	if (rg_wait(g, IBV_WC_RECV_RDMA_WITH_IMM, t1, st->land_ms))
		return -1;
	st->wire_ms = (now_s() - t1) * 1e3;
	return 0;
sys:
	fprintf(stderr, "[head] request to %s failed: %s\n", g->l[k].host, strerror(errno));
	return -1;
}

static int head_main(int n, char **ips)
{
	struct ring_gather *g;
	struct gpu_thread gt;
	struct rg_stats st;
	uint64_t base = ((uint64_t)getpid() << 32) ^ (uint64_t)time(NULL) ^ 0x5EEDull;
	double total_s = 0, best_s = 1e9;
	int use_gpu = !opt.cpu_only && !gpu_thread_init(&gt);
	unsigned int i;
	int k, hosts = 0;
	void *dst;

	dst = alloc_sysmem(opt.len);
	if (!dst) {
		fprintf(stderr, "[head] cannot allocate %zu MiB\n", opt.len >> 20);
		return 1;
	}
	g = rg_open(ips, n, opt.port, opt.connect_wait);
	if (!g)
		return 1;
	g->depth = opt.depth;
	if (rg_bind(g, dst, opt.len, opt.chunk)) {
		rg_close(g);
		return 1;
	}
	for (k = 0; k < g->n; k++) {
		int j, seen = 0;

		for (j = 0; j < k; j++)
			seen |= !strcmp(g->l[j].host, g->l[k].host);
		hosts += !seen;
		printf("[head] link %d: %s at %s via %s, slice %zu MiB at +%zu MiB\n", k, g->l[k].host,
		       g->l[k].ip, g->l[k].dev, g->l[k].len >> 20,
		       (size_t)(g->l[k].slice - (uint8_t *)dst) >> 20);
	}

	for (i = 0; i < opt.iters; i++) {
		uint64_t seed = ITER_SEED(base, i);
		double fill = 0, ms = 0, t0;
		long long bad;

		if (rg_gather(g, seed, i, &st)) {
			rg_close(g);
			return 1;
		}
		for (k = 0; k < g->n; k++)
			if (st.fill_ms[k] > fill)
				fill = st.fill_ms[k];
		if (use_gpu) {
			bad = gpu_verify(&gt, dst, opt.len, seed, 0, &ms);
		} else {
			t0 = now_s();
			bad = cpu_verify(dst, opt.len, seed, 0);
			ms = (now_s() - t0) * 1e3;
		}
		total_s += st.wire_ms / 1e3;
		if (st.wire_ms / 1e3 < best_s)
			best_s = st.wire_ms / 1e3;
		printf("[head] iter %u: produced in %.2f ms (slowest slice %.2f ms), %zu MiB landed in %.2f ms = %.2f GB/s (%.0f Gb/s) [",
		       i, st.produce_ms, fill, opt.len >> 20, st.wire_ms,
		       gbps(opt.len, st.wire_ms / 1e3), gbps(opt.len, st.wire_ms / 1e3) * 8);
		for (k = 0; k < g->n; k++)
			printf("%s%.0f", k ? " " : "", gbps(g->l[k].len, st.land_ms[k] / 1e3) * 8);
		printf(" Gb/s per link], %s verify %.2f ms (%.1f GB/s) %s\n", use_gpu ? "GPU" : "CPU",
		       ms, gbps(opt.len, ms / 1e3), bad == 0 ? "OK" : "MISMATCH");
		if (bad) {
			fprintf(stderr, "[head] %lld mismatching words in iteration %u\n", bad, i);
			rg_close(g);
			return 1;
		}
	}
	printf("[head] summary: %u x %zu MiB from %d GB10(s) over %d link(s), avg %.2f GB/s (%.0f Gb/s), best %.2f GB/s; every word verified on the head's %s\n",
	       opt.iters, opt.len >> 20, hosts, g->n, gbps((uint64_t)opt.len * opt.iters, total_s),
	       gbps((uint64_t)opt.len * opt.iters, total_s) * 8, gbps(opt.len, best_s),
	       use_gpu ? "GPU" : "CPU");
	rg_close(g);
	free(dst);
	return 0;
}

/* ------------------------------------------------------------------------
 * worker: one thread per connection from the head
 * ------------------------------------------------------------------------ */

struct wconn {
	char tag[80];
	struct rdma_cm_id *id;
	struct rdma_event_channel *ec;
	struct ibv_pd *pd;
	struct ibv_comp_channel *cc;
	struct ibv_cq *cq;
	struct ibv_mr *msg_mr, *mr;
	struct rg_req *msg;		/* receive slot for the head's requests */
	void *buf;			/* staging: what this GB10 produced */
	size_t cap;
	struct gpu_thread gt;
	int use_gpu;
};

static pthread_mutex_t w_lock = PTHREAD_MUTEX_INITIALIZER;
static int w_closed, w_active;

static int w_post_recv(struct wconn *c)
{
	struct ibv_recv_wr wr, *bad;
	struct ibv_sge sge;

	sge.addr = (uintptr_t)c->msg;
	sge.length = RG_MSG_SIZE;
	sge.lkey = c->msg_mr->lkey;
	memset(&wr, 0, sizeof(wr));
	wr.sg_list = &sge;
	wr.num_sge = 1;
	return ibv_post_recv(c->id->qp, &wr, &bad);
}

/*
 * Next completion. Sleeps while the head is quiet, so an idle worker costs
 * nothing. Returns 1 with *wc filled, 0 when the head has gone, -1 on error.
 */
static int w_next(struct wconn *c, struct ibv_wc *wc)
{
	for (;;) {
		struct pollfd pfd[2] = {
			{ .fd = c->cc->fd, .events = POLLIN },
			{ .fd = c->ec->fd, .events = POLLIN },
		};
		int n = ibv_poll_cq(c->cq, 1, wc);

		if (n == 0) {
			if (ibv_req_notify_cq(c->cq, 0))
				return -1;
			n = ibv_poll_cq(c->cq, 1, wc);
		}
		if (n < 0)
			return -1;
		if (n > 0) {
			if (wc->status == IBV_WC_WR_FLUSH_ERR)
				return 0;
			if (wc->status != IBV_WC_SUCCESS) {
				fprintf(stderr, "%s %s\n", c->tag, ibv_wc_status_str(wc->status));
				return -1;
			}
			return 1;
		}
		if (poll(pfd, 2, -1) < 0) {
			if (errno == EINTR)
				continue;
			return -1;
		}
		if (pfd[0].revents & POLLIN) {
			struct ibv_cq *cq;
			void *ctx;

			if (!ibv_get_cq_event(c->cc, &cq, &ctx))
				ibv_ack_cq_events(cq, 1);
		}
		if (pfd[1].revents & POLLIN) {
			struct rdma_cm_event *ev;
			int gone;

			if (rdma_get_cm_event(c->ec, &ev))
				return -1;
			gone = ev->event != RDMA_CM_EVENT_ESTABLISHED;
			rdma_ack_cm_event(ev);
			if (gone)
				return 0;
		}
	}
}

/*
 * What this GB10 contributes: len bytes for the slice that starts at 64-bit
 * word `base` of the head's buffer. Here it is the test pattern, written by
 * the GPU into NIC-registered system memory. A real workload replaces this
 * with its own output. Returns ms, < 0 on error.
 */
static double produce(struct wconn *c, size_t len, uint64_t seed, uint64_t base)
{
	double t0;

	if (c->use_gpu)
		return gpu_fill(&c->gt, c->buf, len, seed, base);
	t0 = now_s();
	cpu_fill(c->buf, len, seed, base);
	return (now_s() - t0) * 1e3;
}

static int w_stage(struct wconn *c, size_t len)
{
	if (len <= c->cap)
		return 0;
	if (c->mr)
		ibv_dereg_mr(c->mr);
	free(c->buf);
	c->mr = NULL;
	c->cap = 0;
	c->buf = alloc_sysmem(len);
	if (!c->buf)
		return -1;
	c->mr = ibv_reg_mr(c->pd, c->buf, len, IBV_ACCESS_LOCAL_WRITE);
	if (!c->mr)
		return -1;
	c->cap = len;
	return 0;
}

/* RDMA-WRITE the staged slice into the head; returns seconds, < 0 on error */
static double w_push(struct wconn *c, const struct rg_req *rq)
{
	size_t nchunks = (rq->len + rq->chunk - 1) / rq->chunk, posted = 0, completed = 0;
	unsigned int outstanding = 0, depth = rq->depth;
	double t0 = now_s();

	if (!depth || depth > RG_MAX_DEPTH)
		depth = RG_MAX_DEPTH;
	while (completed < nchunks) {
		struct ibv_wc wcs[16];
		int n, k;

		while (posted < nchunks && outstanding < depth) {
			size_t off = posted * rq->chunk;
			size_t len = rq->len - off < rq->chunk ? rq->len - off : rq->chunk;
			struct ibv_send_wr wr, *bad;
			struct ibv_sge sge;

			sge.addr = (uintptr_t)c->buf + off;
			sge.length = (uint32_t)len;
			sge.lkey = c->mr->lkey;
			memset(&wr, 0, sizeof(wr));
			wr.sg_list = &sge;
			wr.num_sge = 1;
			wr.opcode = posted == nchunks - 1 ? IBV_WR_RDMA_WRITE_WITH_IMM : IBV_WR_RDMA_WRITE;
			wr.send_flags = IBV_SEND_SIGNALED;
			wr.imm_data = htonl(rq->iter);
			wr.wr.rdma.remote_addr = rq->addr + off;
			wr.wr.rdma.rkey = rq->rkey;
			if (ibv_post_send(c->id->qp, &wr, &bad))
				return -1;
			posted++;
			outstanding++;
		}
		n = ibv_poll_cq(c->cq, 16, wcs);
		if (n < 0)
			return -1;
		for (k = 0; k < n; k++) {
			if (wcs[k].status != IBV_WC_SUCCESS) {
				fprintf(stderr, "%s write: %s\n", c->tag, ibv_wc_status_str(wcs[k].status));
				return -1;
			}
			if (wcs[k].opcode == IBV_WC_RDMA_WRITE) {
				completed++;
				outstanding--;
			}
		}
	}
	return now_s() - t0;
}

static void *worker_conn(void *arg)
{
	struct wconn *c = arg;
	struct rg_hello h = { .magic = RG_MAGIC };
	struct ibv_qp_init_attr qa;
	struct rdma_conn_param cp;
	struct rdma_cm_event *ev;
	struct ibv_wc wc;
	double fill_ms = 0;
	int r;

	c->ec = rdma_create_event_channel();
	if (!c->ec || rdma_migrate_id(c->id, c->ec))
		goto sys;
	c->pd = ibv_alloc_pd(c->id->verbs);
	c->cc = ibv_create_comp_channel(c->id->verbs);
	if (!c->pd || !c->cc)
		goto sys;
	c->cq = ibv_create_cq(c->id->verbs, RG_MAX_DEPTH + 32, NULL, c->cc, 0);
	if (!c->cq)
		goto sys;
	memset(&qa, 0, sizeof(qa));
	qa.send_cq = c->cq;
	qa.recv_cq = c->cq;
	qa.qp_type = IBV_QPT_RC;
	qa.cap.max_send_wr = RG_MAX_DEPTH + 8;
	qa.cap.max_recv_wr = 8;
	qa.cap.max_send_sge = 1;
	qa.cap.max_recv_sge = 1;
	if (rdma_create_qp(c->id, c->pd, &qa))
		goto sys;
	c->msg = aligned_alloc(4096, 4096);
	c->msg_mr = c->msg ? ibv_reg_mr(c->pd, c->msg, 4096, IBV_ACCESS_LOCAL_WRITE) : NULL;
	if (!c->msg_mr || w_post_recv(c))
		goto sys;
	c->use_gpu = !opt.cpu_only && !gpu_thread_init(&c->gt);

	gethostname(h.host, sizeof(h.host) - 1);
	conn_params(&cp, &h);
	if (rdma_accept(c->id, &cp) || rdma_get_cm_event(c->ec, &ev))
		goto sys;
	r = ev->event;
	rdma_ack_cm_event(ev);
	if (r != RDMA_CM_EVENT_ESTABLISHED) {
		fprintf(stderr, "%s accept: %s\n", c->tag, rdma_event_str(r));
		goto out;
	}
	snprintf(c->tag + strlen(c->tag), sizeof(c->tag) - strlen(c->tag), " via %s]",
		 ibv_get_device_name(c->id->verbs->device));
	printf("%s connected, produce on %s\n", c->tag, c->use_gpu ? "GPU" : "CPU");

	while ((r = w_next(c, &wc)) > 0) {
		struct rg_req rq;
		double s;

		if (wc.opcode != IBV_WC_RECV)
			continue;		/* one of our acks went out */
		rq = *c->msg;
		if (w_post_recv(c))
			goto sys;
		if (!rq.len || rq.len > opt.max_len || rq.len % 8) {
			fprintf(stderr, "%s refused a slice of %llu bytes\n", c->tag,
				(unsigned long long)rq.len);
			break;
		}
		switch (rq.op) {
		case RG_FILL:
			if (w_stage(c, rq.len))
				goto sys;
			fill_ms = produce(c, rq.len, rq.seed, rq.word_base);
			if (fill_ms < 0) {
				fprintf(stderr, "%s produce failed\n", c->tag);
				goto out;
			}
			if (post_send_imm0(c->id->qp, fill_ms * 1e3 > 4e9 ? ~0u : (uint32_t)(fill_ms * 1e3)))
				goto sys;
			break;
		case RG_PUSH:
			if (rq.len > c->cap || !rq.chunk) {
				fprintf(stderr, "%s PUSH without a matching FILL\n", c->tag);
				goto out;
			}
			s = w_push(c, &rq);
			if (s < 0)
				goto out;
			printf("%s iter %u: %s produced %llu MiB in %.2f ms (%.1f GB/s), pushed in %.2f ms = %.2f GB/s (%.0f Gb/s)\n",
			       c->tag, rq.iter, c->use_gpu ? "GPU" : "CPU",
			       (unsigned long long)rq.len >> 20, fill_ms, gbps(rq.len, fill_ms / 1e3),
			       s * 1e3, gbps(rq.len, s), gbps(rq.len, s) * 8);
			break;
		default:
			fprintf(stderr, "%s unknown request %u\n", c->tag, rq.op);
			goto out;
		}
	}
	if (r < 0)
		goto sys;
	printf("%s closed\n", c->tag);
	goto out;
sys:
	fprintf(stderr, "%s %s\n", c->tag, strerror(errno));
out:
	if (c->id->qp) {
		rdma_disconnect(c->id);
		rdma_destroy_qp(c->id);
	}
	if (c->mr)
		ibv_dereg_mr(c->mr);
	if (c->msg_mr)
		ibv_dereg_mr(c->msg_mr);
	if (c->cq)
		ibv_destroy_cq(c->cq);
	if (c->cc)
		ibv_destroy_comp_channel(c->cc);
	if (c->pd)
		ibv_dealloc_pd(c->pd);
	rdma_destroy_id(c->id);
	if (c->ec)
		rdma_destroy_event_channel(c->ec);
	free(c->buf);
	free(c->msg);
	free(c);

	pthread_mutex_lock(&w_lock);
	if (opt.serve && ++w_closed >= opt.serve) {
		printf("[worker] %d connection(s) served, exiting\n", w_closed);
		exit(0);
	}
	if (!--w_active && opt.idle)
		alarm(opt.idle);	/* the head left: do not linger */
	pthread_mutex_unlock(&w_lock);
	return NULL;
}

static void on_idle(int sig)
{
	static const char msg[] = "[worker] no connection, exiting\n";

	if (write(STDERR_FILENO, msg, sizeof(msg) - 1) < 0)
		_exit(3);
	_exit(3);
}

static int worker_main(void)
{
	struct rdma_event_channel *ec = rdma_create_event_channel();
	struct rdma_cm_id *lid;
	struct sockaddr_in sin;
	char host[64] = "";
	int nconn = 0;

	gethostname(host, sizeof(host) - 1);
	memset(&sin, 0, sizeof(sin));
	sin.sin_family = AF_INET;
	sin.sin_port = htons(opt.port);		/* INADDR_ANY: every ring link */
	if (!ec || rdma_create_id(ec, &lid, NULL, RDMA_PS_TCP) ||
	    rdma_bind_addr(lid, (struct sockaddr *)&sin) || rdma_listen(lid, 16)) {
		fprintf(stderr, "[worker] listen on port %d: %s\n", opt.port, strerror(errno));
		return 1;
	}
	if (opt.idle) {
		signal(SIGALRM, on_idle);
		alarm(opt.idle);
	}
	printf("[worker] %s listening on port %d, produce on %s\n", host, opt.port,
	       opt.cpu_only ? "CPU" : "GPU");

	for (;;) {
		struct rdma_cm_event *ev;
		struct rg_hello peer;
		struct wconn *c;
		pthread_t t;

		if (rdma_get_cm_event(ec, &ev)) {
			fprintf(stderr, "[worker] rdma_get_cm_event: %s\n", strerror(errno));
			return 1;
		}
		if (ev->event != RDMA_CM_EVENT_CONNECT_REQUEST) {
			rdma_ack_cm_event(ev);
			continue;
		}
		memset(&peer, 0, sizeof(peer));
		if (ev->param.conn.private_data_len >= sizeof(peer))
			memcpy(&peer, ev->param.conn.private_data, sizeof(peer));
		c = peer.magic == RG_MAGIC ? calloc(1, sizeof(*c)) : NULL;
		if (!c) {
			rdma_reject(ev->id, NULL, 0);
			rdma_destroy_id(ev->id);
			rdma_ack_cm_event(ev);
			continue;
		}
		c->id = ev->id;
		peer.host[sizeof(peer.host) - 1] = 0;
		snprintf(c->tag, sizeof(c->tag), "[worker %s #%d <- %s", host, nconn++, peer.host);
		rdma_ack_cm_event(ev);
		pthread_mutex_lock(&w_lock);
		w_active++;
		alarm(0);
		pthread_mutex_unlock(&w_lock);
		if (pthread_create(&t, NULL, worker_conn, c)) {
			fprintf(stderr, "[worker] pthread_create: %s\n", strerror(errno));
			return 1;
		}
		pthread_detach(t);
	}
}

/* ------------------------------------------------------------------------ */

static void usage(void)
{
	fprintf(stderr,
		"usage: ringgather --worker [-p port] [-k N] [-i secs] [-C]\n"
		"       ringgather [-s MiB] [-n iters] [-c chunk_MiB] [-q depth] [-p port] [-w secs] [-C] WORKER_IP...\n"
		"  --worker       serve the head: produce slices on this GB10 and push them\n"
		"  WORKER_IP      a worker's address on a ring link; one link per address\n"
		"  -s MiB         total size gathered per iteration (default 4096)\n"
		"  -n iters       iterations (default 5)\n"
		"  -c MiB         RDMA WRITE chunk size (default 1)\n"
		"  -q depth       writes in flight per link (default 32, at most %d)\n"
		"  -p port        RDMA CM port (default 18520)\n"
		"  -w secs        keep retrying the connects until the workers listen (default 60)\n"
		"  -k N           worker: exit after N connections have closed (default: serve forever)\n"
		"  -i secs        worker: exit after secs without any connection\n"
		"  -C             fill/verify on the CPU instead of the GPU\n", RG_MAX_DEPTH);
	exit(2);
}

int main(int argc, char **argv)
{
	static const struct option longopts[] = {
		{ "worker", no_argument, NULL, 'W' },
		{ "cpu", no_argument, NULL, 'C' },
		{ "help", no_argument, NULL, 'h' },
		{ 0 }
	};
	int o;

	setvbuf(stdout, NULL, _IOLBF, 0);
	while ((o = getopt_long(argc, argv, "s:n:c:q:p:w:k:i:CWh", longopts, NULL)) != -1) {
		switch (o) {
		case 'W': opt.worker = 1; break;
		case 's': opt.len = strtoull(optarg, NULL, 0) << 20; break;
		case 'n': opt.iters = atoi(optarg); break;
		case 'c': opt.chunk = strtoull(optarg, NULL, 0) << 20; break;
		case 'q': opt.depth = atoi(optarg); break;
		case 'p': opt.port = atoi(optarg); break;
		case 'w': opt.connect_wait = atoi(optarg); break;
		case 'k': opt.serve = atoi(optarg); break;
		case 'i': opt.idle = atoi(optarg); break;
		case 'C': opt.cpu_only = 1; break;
		default: usage();
		}
	}
	if (opt.worker ? optind != argc : optind == argc)
		usage();
	if (!opt.worker && (!opt.len || !opt.chunk || opt.len % opt.chunk || opt.chunk > (1u << 30) ||
			    !opt.iters || !opt.depth || opt.depth > RG_MAX_DEPTH)) {
		fprintf(stderr, "size must be a multiple of the chunk; chunk <= 1 GiB; depth 1..%d\n",
			RG_MAX_DEPTH);
		return 2;
	}

	if (!opt.cpu_only && gpu_init())
		opt.cpu_only = 1;
	return opt.worker ? worker_main() : head_main(argc - optind, argv + optind);
}
