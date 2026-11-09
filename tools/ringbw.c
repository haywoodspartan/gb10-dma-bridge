// SPDX-License-Identifier: GPL-2.0
/*
 * ringbw - RDMA ring bandwidth test for GB10 (DGX Spark) nodes, fed and
 *          consumed by the GPU.
 *
 * One process per node. --listen receives on one ring link, --connect sends
 * on the other; with both, the process is one segment of a full ring and all
 * three nodes can run at once.
 *
 * Sender:    GPU kernel fills a NIC-registered system-memory buffer with a
 *            pattern, the NIC RDMA-WRITEs it to the neighbour in chunks, the
 *            last chunk carries an immediate (iteration number).
 * Receiver:  the immediate tells it the buffer landed; a GPU kernel verifies
 *            every 64-bit word straight from the NIC-written pages (ATS, no
 *            copy), then it acks with a SEND_WITH_IMM.
 *
 * This is the data path that works on GB10 with driver 580: no dma-buf, no
 * GPUDirect RDMA, just system memory shared by NIC and GPU.
 *
 *   ringbw --listen 10.100.24.2
 *   ringbw --connect 10.100.25.1
 *   ringbw --listen 10.100.24.2 --connect 10.100.25.1
 *
 * Options: -s MiB (1024)  -n iters (10)  -c chunk MiB (1)  -q depth (32)
 *          -p port (18516)  -C use CPU instead of the GPU for fill/verify
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <getopt.h>
#include <netinet/in.h>
#include <pthread.h>
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

#define PATTERN_K	0x9E3779B97F4A7C15ull
#define ITER_SEED(base, i)	((base) ^ ((uint64_t)(i) * 0x0123456789ABCDEFull))

static struct {
	const char *listen_ip;
	const char *connect_ip;
	size_t len;
	unsigned int iters;
	size_t chunk;
	unsigned int depth;
	int port;
	int cpu_only;
	int connect_wait;	/* seconds to keep retrying until the peer listens */
} opt = {
	.len = 1024ull << 20, .iters = 10, .chunk = 1ull << 20, .depth = 32,
	.port = 18516, .connect_wait = 60,
};

struct hello {
	uint64_t addr;
	uint64_t len;
	uint64_t seed;
	uint32_t rkey;
	uint32_t iters;
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

/* ------------------------------------------------------------------------
 * GPU: fill / verify kernels compiled at runtime, running on system memory
 * ------------------------------------------------------------------------ */

static struct {
	int ok;				/* kernels available */
	int pageable;			/* GPU can address system memory */
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
"                                  unsigned long long seed)\n"
"{\n"
"    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;\n"
"         i < n; i += (unsigned long long)gridDim.x * blockDim.x)\n"
"        p[i] = seed ^ (i * 0x9E3779B97F4A7C15ull);\n"
"}\n"
"extern \"C\" __global__ void verify(const unsigned long long *p, unsigned long long n,\n"
"                                    unsigned long long seed, unsigned long long *bad)\n"
"{\n"
"    unsigned long long cnt = 0;\n"
"    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;\n"
"         i < n; i += (unsigned long long)gridDim.x * blockDim.x)\n"
"        if (p[i] != (seed ^ (i * 0x9E3779B97F4A7C15ull)))\n"
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

	if (nvrtcCreateProgram(&prog, kernel_src, "ringbw.cu", 0, NULL, NULL) != NVRTC_SUCCESS)
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
static double gpu_fill(struct gpu_thread *t, void *buf, size_t len, uint64_t seed)
{
	unsigned long long n = len / 8, s = seed;
	void *args[] = { &buf, &n, &s };
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
static long long gpu_verify(struct gpu_thread *t, void *buf, size_t len, uint64_t seed, double *ms)
{
	unsigned long long n = len / 8, s = seed, bad = 0;
	void *args[] = { &buf, &n, &s, &t->bad };
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

static void cpu_fill(void *buf, size_t len, uint64_t seed)
{
	uint64_t *p = buf;
	size_t i, n = len / 8;

	for (i = 0; i < n; i++)
		p[i] = seed ^ (i * PATTERN_K);
}

static long long cpu_verify(const void *buf, size_t len, uint64_t seed)
{
	const uint64_t *p = buf;
	size_t i, n = len / 8;
	long long bad = 0;

	for (i = 0; i < n; i++)
		if (p[i] != (seed ^ (i * PATTERN_K)))
			bad++;
	return bad;
}

/* ------------------------------------------------------------------------
 * RDMA connection
 * ------------------------------------------------------------------------ */

struct conn {
	const char *tag;
	struct rdma_event_channel *ec;
	struct rdma_cm_id *id;
	struct rdma_cm_id *listen_id;
	struct ibv_pd *pd;
	struct ibv_cq *cq;
	struct ibv_mr *mr;
	void *buf;
	size_t len;
	unsigned int iters;
	uint64_t seed;
	struct hello peer;
	struct gpu_thread gt;
	int use_gpu;
};

static void fail(struct conn *c, const char *what)
{
	fprintf(stderr, "%s %s: %s\n", c->tag, what, strerror(errno));
	exit(1);
}

static struct rdma_cm_event *wait_event(struct conn *c, enum rdma_cm_event_type want)
{
	struct rdma_cm_event *ev;

	if (rdma_get_cm_event(c->ec, &ev))
		fail(c, "rdma_get_cm_event");
	if (ev->event != want) {
		fprintf(stderr, "%s expected %s, got %s (status %d)\n", c->tag,
			rdma_event_str(want), rdma_event_str(ev->event), ev->status);
		exit(1);
	}
	return ev;
}

static void conn_alloc_buf(struct conn *c)
{
	/* system memory: 2 MiB aligned, THP hinted, populated */
	c->buf = aligned_alloc(2u << 20, c->len);
	if (!c->buf)
		fail(c, "aligned_alloc");
	madvise(c->buf, c->len, MADV_HUGEPAGE);
	memset(c->buf, 0, c->len);
	c->use_gpu = !opt.cpu_only && !gpu_thread_init(&c->gt);
}

static void conn_setup_verbs(struct conn *c)
{
	struct ibv_qp_init_attr qa;
	struct ibv_port_attr pa;
	size_t cqe = opt.depth + c->iters + 16;

	c->pd = ibv_alloc_pd(c->id->verbs);
	if (!c->pd)
		fail(c, "ibv_alloc_pd");
	c->cq = ibv_create_cq(c->id->verbs, cqe > 65536 ? 65536 : (int)cqe, NULL, NULL, 0);
	if (!c->cq)
		fail(c, "ibv_create_cq");

	memset(&qa, 0, sizeof(qa));
	qa.send_cq = c->cq;
	qa.recv_cq = c->cq;
	qa.qp_type = IBV_QPT_RC;
	qa.cap.max_send_wr = opt.depth + 4;
	qa.cap.max_recv_wr = c->iters + 4;
	qa.cap.max_send_sge = 1;
	qa.cap.max_recv_sge = 1;
	if (rdma_create_qp(c->id, c->pd, &qa))
		fail(c, "rdma_create_qp");

	c->mr = ibv_reg_mr(c->pd, c->buf, c->len,
			   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
	if (!c->mr)
		fail(c, "ibv_reg_mr (raise ulimit -l?)");

	ibv_query_port(c->id->verbs, c->id->port_num, &pa);
	printf("%s %s port %u, RoCE mtu %u, %zu MiB registered (lkey 0x%x rkey 0x%x)\n",
	       c->tag, ibv_get_device_name(c->id->verbs->device), c->id->port_num,
	       128u << pa.active_mtu, c->len >> 20, c->mr->lkey, c->mr->rkey);
}

/* undo conn_setup_verbs() and the cm_id, keep the buffer and GPU state */
static void conn_reset_verbs(struct conn *c)
{
	if (c->id && c->id->qp)
		rdma_destroy_qp(c->id);
	if (c->mr)
		ibv_dereg_mr(c->mr);
	if (c->cq)
		ibv_destroy_cq(c->cq);
	if (c->pd)
		ibv_dealloc_pd(c->pd);
	if (c->id)
		rdma_destroy_id(c->id);
	c->id = NULL;
	c->mr = NULL;
	c->cq = NULL;
	c->pd = NULL;
}

static void conn_teardown(struct conn *c)
{
	if (c->id && c->id->qp)
		rdma_destroy_qp(c->id);
	if (c->mr)
		ibv_dereg_mr(c->mr);
	if (c->cq)
		ibv_destroy_cq(c->cq);
	if (c->pd)
		ibv_dealloc_pd(c->pd);
	if (c->id)
		rdma_destroy_id(c->id);
	if (c->listen_id)
		rdma_destroy_id(c->listen_id);
	if (c->ec)
		rdma_destroy_event_channel(c->ec);
	free(c->buf);
}

static void post_recv(struct conn *c, uint64_t wr_id)
{
	struct ibv_recv_wr wr, *bad;

	memset(&wr, 0, sizeof(wr));
	wr.wr_id = wr_id;
	wr.num_sge = 0;
	if (ibv_post_recv(c->id->qp, &wr, &bad))
		fail(c, "ibv_post_recv");
}

static void post_send_imm(struct conn *c, uint32_t imm)
{
	struct ibv_send_wr wr, *bad;

	memset(&wr, 0, sizeof(wr));
	wr.wr_id = 0xACC;
	wr.opcode = IBV_WR_SEND_WITH_IMM;
	wr.send_flags = IBV_SEND_SIGNALED;
	wr.imm_data = htonl(imm);
	wr.num_sge = 0;
	if (ibv_post_send(c->id->qp, &wr, &bad))
		fail(c, "ibv_post_send(ack)");
}

static void post_write(struct conn *c, size_t off, size_t n, int last, uint32_t imm)
{
	struct ibv_send_wr wr, *bad;
	struct ibv_sge sge;

	sge.addr = (uintptr_t)c->buf + off;
	sge.length = (uint32_t)n;
	sge.lkey = c->mr->lkey;
	memset(&wr, 0, sizeof(wr));
	wr.wr_id = off;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.opcode = last ? IBV_WR_RDMA_WRITE_WITH_IMM : IBV_WR_RDMA_WRITE;
	wr.send_flags = IBV_SEND_SIGNALED;
	wr.imm_data = htonl(imm);
	wr.wr.rdma.remote_addr = c->peer.addr + off;
	wr.wr.rdma.rkey = c->peer.rkey;
	if (ibv_post_send(c->id->qp, &wr, &bad))
		fail(c, "ibv_post_send(write)");
}

/* busy-poll one completion; exits on error */
static void poll_one(struct conn *c, struct ibv_wc *wc)
{
	int n;

	do {
		n = ibv_poll_cq(c->cq, 1, wc);
	} while (n == 0);
	if (n < 0)
		fail(c, "ibv_poll_cq");
	if (wc->status != IBV_WC_SUCCESS) {
		fprintf(stderr, "%s completion error: %s (wr_id 0x%llx)\n", c->tag,
			ibv_wc_status_str(wc->status), (unsigned long long)wc->wr_id);
		exit(1);
	}
}

static void make_hello(struct conn *c, struct hello *h)
{
	h->addr = (uintptr_t)c->buf;
	h->len = c->len;
	h->seed = c->seed;
	h->rkey = c->mr->rkey;
	h->iters = c->iters;
}

/* ------------------------------------------------------------------------
 * receiver
 * ------------------------------------------------------------------------ */

static void *server_main(void *arg)
{
	struct conn c = { .tag = "[recv]" };
	struct sockaddr_in sin;
	struct rdma_cm_event *ev;
	struct rdma_conn_param cp;
	struct hello h;
	unsigned int i;
	double sum_ms = 0;

	c.ec = rdma_create_event_channel();
	if (!c.ec || rdma_create_id(c.ec, &c.listen_id, NULL, RDMA_PS_TCP))
		fail(&c, "rdma_create_id");
	memset(&sin, 0, sizeof(sin));
	sin.sin_family = AF_INET;
	sin.sin_port = htons(opt.port);
	if (inet_pton(AF_INET, opt.listen_ip, &sin.sin_addr) != 1) {
		fprintf(stderr, "bad --listen address %s\n", opt.listen_ip);
		exit(1);
	}
	if (rdma_bind_addr(c.listen_id, (struct sockaddr *)&sin))
		fail(&c, "rdma_bind_addr (is that IP on a ring link?)");
	if (rdma_listen(c.listen_id, 1))
		fail(&c, "rdma_listen");
	printf("%s waiting on %s:%d\n", c.tag, opt.listen_ip, opt.port);

	ev = wait_event(&c, RDMA_CM_EVENT_CONNECT_REQUEST);
	c.id = ev->id;
	if (ev->param.conn.private_data_len < sizeof(c.peer)) {
		fprintf(stderr, "%s peer sent no hello\n", c.tag);
		exit(1);
	}
	memcpy(&c.peer, ev->param.conn.private_data, sizeof(c.peer));
	rdma_ack_cm_event(ev);

	c.len = c.peer.len;
	c.iters = c.peer.iters;
	c.seed = c.peer.seed;
	conn_alloc_buf(&c);
	conn_setup_verbs(&c);
	for (i = 0; i < c.iters; i++)
		post_recv(&c, i);

	make_hello(&c, &h);
	memset(&cp, 0, sizeof(cp));
	cp.private_data = &h;
	cp.private_data_len = sizeof(h);
	cp.responder_resources = 1;
	cp.initiator_depth = 1;
	cp.retry_count = 7;
	cp.rnr_retry_count = 7;
	if (rdma_accept(c.id, &cp))
		fail(&c, "rdma_accept");
	ev = wait_event(&c, RDMA_CM_EVENT_ESTABLISHED);
	rdma_ack_cm_event(ev);
	printf("%s connected, %u iterations of %zu MiB, verify on %s\n", c.tag, c.iters,
	       c.len >> 20, c.use_gpu ? "GPU" : "CPU");

	for (i = 0; i < c.iters; i++) {
		struct ibv_wc wc;
		uint64_t seed = ITER_SEED(c.seed, i);
		long long bad;
		double ms = 0, t0;
		uint32_t imm;

		poll_one(&c, &wc);
		if (wc.opcode != IBV_WC_RECV_RDMA_WITH_IMM) {
			fprintf(stderr, "%s unexpected opcode %d\n", c.tag, wc.opcode);
			exit(1);
		}
		imm = ntohl(wc.imm_data);
		if (c.use_gpu) {
			bad = gpu_verify(&c.gt, c.buf, c.len, seed, &ms);
		} else {
			t0 = now_s();
			bad = cpu_verify(c.buf, c.len, seed);
			ms = (now_s() - t0) * 1e3;
		}
		sum_ms += ms;
		printf("%s iter %u: %zu MiB landed, %s verify %.2f ms (%.1f GB/s read) %s\n",
		       c.tag, imm, c.len >> 20, c.use_gpu ? "GPU" : "CPU", ms,
		       gbps(c.len, ms / 1e3), bad == 0 ? "OK" : "MISMATCH");
		if (bad) {
			fprintf(stderr, "%s %lld mismatching words in iteration %u\n", c.tag, bad, imm);
			exit(1);
		}
		post_send_imm(&c, imm);
		poll_one(&c, &wc);		/* ack send completion */
	}
	printf("%s done: %u x %zu MiB verified on %s, avg %.2f ms per verify\n", c.tag,
	       c.iters, c.len >> 20, c.use_gpu ? "GPU" : "CPU", sum_ms / c.iters);

	ev = wait_event(&c, RDMA_CM_EVENT_DISCONNECTED);
	rdma_ack_cm_event(ev);
	rdma_disconnect(c.id);
	conn_teardown(&c);
	return NULL;
}

/* ------------------------------------------------------------------------
 * sender
 * ------------------------------------------------------------------------ */

static void *client_main(void *arg)
{
	struct conn c = { .tag = "[send]" };
	struct sockaddr_in sin;
	struct rdma_cm_event *ev;
	struct rdma_conn_param cp;
	struct hello h;
	unsigned int i;
	double total_s = 0, best_s = 1e9, deadline;
	int attempts = 0;

	c.len = opt.len;
	c.iters = opt.iters;
	c.seed = ((uint64_t)getpid() << 32) ^ (uint64_t)time(NULL) ^ 0x5EEDull;

	memset(&sin, 0, sizeof(sin));
	sin.sin_family = AF_INET;
	sin.sin_port = htons(opt.port);
	if (inet_pton(AF_INET, opt.connect_ip, &sin.sin_addr) != 1) {
		fprintf(stderr, "bad --connect address %s\n", opt.connect_ip);
		exit(1);
	}

	c.ec = rdma_create_event_channel();
	if (!c.ec)
		fail(&c, "rdma_create_event_channel");
	conn_alloc_buf(&c);

	/*
	 * The neighbour's receiver may not be listening yet (it is started over
	 * ssh at the same moment), so retry address/route/connect until it is
	 * or until --wait seconds have passed.
	 */
	deadline = now_s() + opt.connect_wait;
	for (;;) {
		int retry = 0;

		attempts++;
		if (rdma_create_id(c.ec, &c.id, NULL, RDMA_PS_TCP))
			fail(&c, "rdma_create_id");
		if (rdma_resolve_addr(c.id, NULL, (struct sockaddr *)&sin, 2000))
			fail(&c, "rdma_resolve_addr");
		if (rdma_get_cm_event(c.ec, &ev))
			fail(&c, "rdma_get_cm_event");
		if (ev->event != RDMA_CM_EVENT_ADDR_RESOLVED) {
			retry = 1;
		} else {
			rdma_ack_cm_event(ev);
			if (rdma_resolve_route(c.id, 2000))
				fail(&c, "rdma_resolve_route");
			if (rdma_get_cm_event(c.ec, &ev))
				fail(&c, "rdma_get_cm_event");
			if (ev->event != RDMA_CM_EVENT_ROUTE_RESOLVED)
				retry = 1;
		}
		if (!retry) {
			rdma_ack_cm_event(ev);
			conn_setup_verbs(&c);
			for (i = 0; i < c.iters; i++)
				post_recv(&c, 0xA000 + i);		/* acks */
			make_hello(&c, &h);
			memset(&cp, 0, sizeof(cp));
			cp.private_data = &h;
			cp.private_data_len = sizeof(h);
			cp.responder_resources = 1;
			cp.initiator_depth = 1;
			cp.retry_count = 7;
			cp.rnr_retry_count = 7;
			if (rdma_connect(c.id, &cp))
				fail(&c, "rdma_connect");
			if (rdma_get_cm_event(c.ec, &ev))
				fail(&c, "rdma_get_cm_event");
			if (ev->event == RDMA_CM_EVENT_ESTABLISHED)
				break;
			retry = 1;
		}
		/* not there yet */
		if (now_s() > deadline || (ev->event != RDMA_CM_EVENT_REJECTED &&
					   ev->event != RDMA_CM_EVENT_UNREACHABLE &&
					   ev->event != RDMA_CM_EVENT_CONNECT_ERROR &&
					   ev->event != RDMA_CM_EVENT_ADDR_ERROR &&
					   ev->event != RDMA_CM_EVENT_ROUTE_ERROR)) {
			fprintf(stderr, "%s giving up on %s after %d attempt(s): %s (status %d)\n", c.tag,
				opt.connect_ip, attempts, rdma_event_str(ev->event), ev->status);
			exit(1);
		}
		if (attempts == 1)
			printf("%s %s not listening yet (%s), retrying for up to %ds\n", c.tag,
			       opt.connect_ip, rdma_event_str(ev->event), opt.connect_wait);
		rdma_ack_cm_event(ev);
		conn_reset_verbs(&c);
		usleep(500000);
	}
	if (ev->param.conn.private_data_len < sizeof(c.peer)) {
		fprintf(stderr, "%s peer sent no hello\n", c.tag);
		exit(1);
	}
	memcpy(&c.peer, ev->param.conn.private_data, sizeof(c.peer));
	rdma_ack_cm_event(ev);
	if (c.peer.len < c.len) {
		fprintf(stderr, "%s peer buffer too small\n", c.tag);
		exit(1);
	}
	printf("%s connected to %s, peer buffer 0x%llx rkey 0x%x, fill on %s\n", c.tag,
	       opt.connect_ip, (unsigned long long)c.peer.addr, c.peer.rkey,
	       c.use_gpu ? "GPU" : "CPU");

	for (i = 0; i < c.iters; i++) {
		uint64_t seed = ITER_SEED(c.seed, i);
		size_t nchunks = c.len / opt.chunk, posted = 0, completed = 0;
		unsigned int outstanding = 0;
		int acked = 0;
		double fill_ms, t0, t1;

		if (c.use_gpu) {
			fill_ms = gpu_fill(&c.gt, c.buf, c.len, seed);
			if (fill_ms < 0) {
				fprintf(stderr, "%s GPU fill failed\n", c.tag);
				exit(1);
			}
		} else {
			t0 = now_s();
			cpu_fill(c.buf, c.len, seed);
			fill_ms = (now_s() - t0) * 1e3;
		}

		t0 = now_s();
		while (completed < nchunks) {
			struct ibv_wc wcs[16];
			int n, k;

			while (posted < nchunks && outstanding < opt.depth) {
				post_write(&c, posted * opt.chunk, opt.chunk,
					   posted == nchunks - 1, i);
				posted++;
				outstanding++;
			}
			n = ibv_poll_cq(c.cq, 16, wcs);
			if (n < 0)
				fail(&c, "ibv_poll_cq");
			for (k = 0; k < n; k++) {
				if (wcs[k].status != IBV_WC_SUCCESS) {
					fprintf(stderr, "%s completion error: %s\n", c.tag,
						ibv_wc_status_str(wcs[k].status));
					exit(1);
				}
				if (wcs[k].opcode == IBV_WC_RECV) {
					acked = 1;
				} else {
					completed++;
					outstanding--;
				}
			}
		}
		t1 = now_s();
		total_s += t1 - t0;
		if (t1 - t0 < best_s)
			best_s = t1 - t0;
		printf("%s iter %u: %s fill %.2f ms, %zu MiB on the wire in %.2f ms = %.2f GB/s (%.0f Gb/s)\n",
		       c.tag, i, c.use_gpu ? "GPU" : "CPU", fill_ms, c.len >> 20,
		       (t1 - t0) * 1e3, gbps(c.len, t1 - t0), gbps(c.len, t1 - t0) * 8);

		while (!acked) {
			struct ibv_wc wc;

			poll_one(&c, &wc);
			if (wc.opcode == IBV_WC_RECV && ntohl(wc.imm_data) == i)
				acked = 1;
		}
	}
	printf("%s summary: %u x %zu MiB, avg %.2f GB/s (%.0f Gb/s), best %.2f GB/s; peer verified every iteration\n",
	       c.tag, c.iters, c.len >> 20, gbps((uint64_t)c.len * c.iters, total_s),
	       gbps((uint64_t)c.len * c.iters, total_s) * 8, gbps(c.len, best_s));

	rdma_disconnect(c.id);
	ev = wait_event(&c, RDMA_CM_EVENT_DISCONNECTED);
	rdma_ack_cm_event(ev);
	conn_teardown(&c);
	return NULL;
}

/* ------------------------------------------------------------------------ */

static void usage(void)
{
	fprintf(stderr,
		"usage: ringbw [--listen IP] [--connect IP] [-s MiB] [-n iters] [-c chunk_MiB] [-q depth] [-p port] [-w secs] [-C]\n"
		"  --listen IP    receive on the ring link that owns IP\n"
		"  --connect IP   send to the neighbour at IP (its --listen address)\n"
		"  -s MiB         buffer size per iteration (default 1024)\n"
		"  -n iters       iterations (default 10)\n"
		"  -c MiB         RDMA WRITE chunk size (default 1)\n"
		"  -q depth       writes in flight (default 32)\n"
		"  -p port        RDMA CM port (default 18516)\n"
		"  -C             fill/verify on the CPU instead of the GPU\n"
		"  -w secs        keep retrying the connect until the peer listens (default 60)\n");
	exit(2);
}

int main(int argc, char **argv)
{
	static const struct option longopts[] = {
		{ "listen", required_argument, NULL, 'l' },
		{ "connect", required_argument, NULL, 'x' },
		{ "cpu", no_argument, NULL, 'C' },
		{ "help", no_argument, NULL, 'h' },
		{ 0 }
	};
	pthread_t ts, tc;
	int o;

	while ((o = getopt_long(argc, argv, "l:x:s:n:c:q:p:w:Ch", longopts, NULL)) != -1) {
		switch (o) {
		case 'l': opt.listen_ip = optarg; break;
		case 'x': opt.connect_ip = optarg; break;
		case 's': opt.len = strtoull(optarg, NULL, 0) << 20; break;
		case 'n': opt.iters = atoi(optarg); break;
		case 'c': opt.chunk = strtoull(optarg, NULL, 0) << 20; break;
		case 'q': opt.depth = atoi(optarg); break;
		case 'p': opt.port = atoi(optarg); break;
		case 'C': opt.cpu_only = 1; break;
		case 'w': opt.connect_wait = atoi(optarg); break;
		default: usage();
		}
	}
	if (!opt.listen_ip && !opt.connect_ip)
		usage();
	if (!opt.len || !opt.chunk || opt.len % opt.chunk || opt.chunk > (1u << 30) ||
	    !opt.iters || opt.iters > 10000 || !opt.depth) {
		fprintf(stderr, "size must be a multiple of the chunk; chunk <= 1 GiB; 1..10000 iterations\n");
		return 2;
	}

	if (!opt.cpu_only && gpu_init())
		opt.cpu_only = 1;

	if (opt.listen_ip && pthread_create(&ts, NULL, server_main, NULL))
		return 1;
	if (opt.connect_ip && pthread_create(&tc, NULL, client_main, NULL))
		return 1;
	if (opt.listen_ip)
		pthread_join(ts, NULL);
	if (opt.connect_ip)
		pthread_join(tc, NULL);
	return 0;
}
