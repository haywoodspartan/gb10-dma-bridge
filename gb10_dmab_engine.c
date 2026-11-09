// SPDX-License-Identifier: GPL-2.0
/*
 * gb10_dmab - copy engines: one SCHED_FIFO kernel thread pinned to each
 * dedicated CPU, each draining its own job queue.
 */
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/module.h>
#include <linux/kthread.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/ktime.h>
#include <linux/io.h>
#include <linux/dma-buf.h>
#include <linux/overflow.h>
#include <linux/minmax.h>
#include <linux/sizes.h>
#include <linux/cache.h>

#include "gb10_dmab.h"

#define GB10_DMAB_BOUNCE_SIZE	SZ_64K	/* MMIO -> MMIO staging */
#define GB10_DMAB_MIN_STRIPE	SZ_64K	/* smallest per-engine slice when striping */

/* ------------------------------------------------------------------------
 * the copy itself
 * ------------------------------------------------------------------------ */

static void copy_chunk(struct gb10_dmab_engine *eng,
		       struct gb10_dmab_buf *dst, u64 doff,
		       struct gb10_dmab_buf *src, u64 soff, size_t n)
{
	if (src->mmio && dst->mmio) {
		while (n) {
			size_t k = min_t(size_t, n, GB10_DMAB_BOUNCE_SIZE);

			memcpy_fromio(eng->bounce, src->mmio + soff, k);
			memcpy_toio(dst->mmio + doff, eng->bounce, k);
			soff += k;
			doff += k;
			n -= k;
		}
	} else if (src->mmio) {
		memcpy_fromio(dst->kva + doff, src->mmio + soff, n);
	} else if (dst->mmio) {
		memcpy_toio(dst->mmio + doff, src->kva + soff, n);
	} else if (src == dst) {
		memmove(dst->kva + doff, src->kva + soff, n);
	} else {
		memcpy(dst->kva + doff, src->kva + soff, n);
	}
}

static void engine_run_job(struct gb10_dmab_engine *eng,
			   struct gb10_dmab_job *job)
{
	struct gb10_dmab_req *req = job->req;
	struct gb10_dmab_dev *bd = eng->bd;
	u64 done = 0;
	u64 t0 = ktime_get_ns();

	while (done < job->len) {
		size_t n = min_t(u64, job->len - done, bd->chunk);

		copy_chunk(eng, req->dst, job->dst_off + done,
			   req->src, job->src_off + done, n);
		done += n;
		cond_resched();
	}

	eng->bytes += job->len;
	eng->jobs++;
	eng->busy_ns += ktime_get_ns() - t0;
}

/* Last job finished: close CPU access, publish the completion. */
static void req_complete(struct gb10_dmab_req *req)
{
	struct gb10_dmab_ctx *ctx = req->ctx;

	req->t_done = ktime_get_ns();

	if (req->flags & GB10_DMAB_COPY_F_FLUSH)
		gb10_dmab_buf_sync(req->dst, DMA_TO_DEVICE, true);
	if (req->src->type == GB10_DMAB_BUF_DMABUF)
		dma_buf_end_cpu_access(req->src->dmabuf, DMA_FROM_DEVICE);
	if (req->dst->type == GB10_DMAB_BUF_DMABUF)
		dma_buf_end_cpu_access(req->dst->dmabuf, DMA_TO_DEVICE);

	/*
	 * Publish, retire from inflight and wake under ctx->lock: release()
	 * waits for inflight == 0 and then takes ctx->lock once, which cannot
	 * succeed until this wake-up has finished touching @ctx.
	 */
	spin_lock(&ctx->lock);
	list_add_tail(&req->node, &ctx->done);
	atomic_dec(&ctx->inflight);
	wake_up_all(&ctx->wq);
	spin_unlock(&ctx->lock);
}

static void engine_job_done(struct gb10_dmab_job *job)
{
	struct gb10_dmab_req *req = job->req;

	if (job->status)
		atomic_cmpxchg(&req->status, 0, job->status);
	if (atomic_dec_and_test(&req->remaining))
		req_complete(req);
}

static struct gb10_dmab_job *engine_dequeue(struct gb10_dmab_engine *eng)
{
	struct gb10_dmab_job *job;

	spin_lock(&eng->lock);
	job = list_first_entry_or_null(&eng->queue, struct gb10_dmab_job, node);
	if (job) {
		list_del(&job->node);
		atomic_dec(&eng->queued);
	}
	spin_unlock(&eng->lock);
	return job;
}

static int engine_thread(void *data)
{
	struct gb10_dmab_engine *eng = data;
	struct gb10_dmab_dev *bd = eng->bd;
	u64 idle_since = 0;

	while (!kthread_should_stop()) {
		struct gb10_dmab_job *job = engine_dequeue(eng);

		if (job) {
			idle_since = 0;
			engine_run_job(eng, job);
			engine_job_done(job);
			continue;
		}

		/* short busy-poll keeps submit->start latency in the microseconds */
		if (bd->poll_us) {
			u64 now = ktime_get_ns();

			if (!idle_since)
				idle_since = now;
			if (now - idle_since < (u64)bd->poll_us * NSEC_PER_USEC) {
				cpu_relax();
				cond_resched();
				continue;
			}
		}

		eng->sleeps++;
		wait_event_idle(eng->wq, !list_empty_careful(&eng->queue) ||
					 kthread_should_stop());
		idle_since = 0;
	}
	return 0;
}

/* ------------------------------------------------------------------------
 * engine lifecycle
 * ------------------------------------------------------------------------ */

void gb10_dmab_engines_stop(struct gb10_dmab_dev *bd)
{
	unsigned int i;

	if (!bd->engines)
		return;
	for (i = 0; i < bd->nr_engines; i++) {
		struct gb10_dmab_engine *eng = &bd->engines[i];

		if (eng->task)
			kthread_stop(eng->task);
		kfree(eng->bounce);
	}
	kfree(bd->engines);
	bd->engines = NULL;
	bd->nr_engines = 0;
}

int gb10_dmab_engines_start(struct gb10_dmab_dev *bd)
{
	unsigned int cpu, i = 0;
	int ret;

	bd->nr_engines = cpumask_weight(bd->cpus);
	bd->engines = kcalloc(bd->nr_engines, sizeof(*bd->engines), GFP_KERNEL);
	if (!bd->engines)
		return -ENOMEM;

	for_each_cpu(cpu, bd->cpus) {
		struct gb10_dmab_engine *eng = &bd->engines[i];

		eng->bd = bd;
		eng->index = i;
		eng->cpu = cpu;
		spin_lock_init(&eng->lock);
		INIT_LIST_HEAD(&eng->queue);
		init_waitqueue_head(&eng->wq);
		atomic_set(&eng->queued, 0);

		eng->bounce = kmalloc(GB10_DMAB_BOUNCE_SIZE, GFP_KERNEL);
		if (!eng->bounce) {
			ret = -ENOMEM;
			goto err;
		}

		eng->task = kthread_run_on_cpu(engine_thread, eng, cpu,
					       GB10_DMAB_NAME "/%u");
		if (IS_ERR(eng->task)) {
			ret = PTR_ERR(eng->task);
			eng->task = NULL;
			pr_err("cannot start engine on CPU %u: %d\n", cpu, ret);
			goto err;
		}
		if (bd->rt)
			sched_set_fifo_low(eng->task);
		i++;
	}
	return 0;

err:
	gb10_dmab_engines_stop(bd);
	return ret;
}

/* ------------------------------------------------------------------------
 * submission / completion
 * ------------------------------------------------------------------------ */

/* least-loaded engine, round-robin on ties */
static struct gb10_dmab_engine *engine_pick(struct gb10_dmab_dev *bd)
{
	unsigned int start = atomic_inc_return(&bd->rr) % bd->nr_engines;
	struct gb10_dmab_engine *best = &bd->engines[start];
	int best_q = atomic_read(&best->queued);
	unsigned int i;

	for (i = 1; i < bd->nr_engines; i++) {
		struct gb10_dmab_engine *eng =
			&bd->engines[(start + i) % bd->nr_engines];
		int q = atomic_read(&eng->queued);

		if (q < best_q) {
			best = eng;
			best_q = q;
		}
	}
	return best;
}

static void engine_enqueue(struct gb10_dmab_engine *eng,
			   struct gb10_dmab_job *job)
{
	spin_lock(&eng->lock);
	list_add_tail(&job->node, &eng->queue);
	atomic_inc(&eng->queued);
	spin_unlock(&eng->lock);
	wake_up(&eng->wq);
}

int gb10_dmab_submit(struct gb10_dmab_ctx *ctx, struct gb10_dmab_copy *arg)
{
	struct gb10_dmab_dev *bd = ctx->bd;
	struct gb10_dmab_buf *src, *dst;
	struct gb10_dmab_req *req;
	unsigned int nr, i;
	u64 end, chunk;
	int ret;

	if (!arg->len || arg->pad ||
	    (arg->flags & ~(GB10_DMAB_COPY_F_STRIPE | GB10_DMAB_COPY_F_FLUSH |
			    GB10_DMAB_COPY_F_WAIT)))
		return -EINVAL;
	if (arg->engine < -1 || arg->engine >= (int)bd->nr_engines)
		return -EINVAL;

	src = gb10_dmab_buf_get(ctx, arg->src_handle);
	if (!src)
		return -ENOENT;
	dst = gb10_dmab_buf_get(ctx, arg->dst_handle);
	if (!dst) {
		ret = -ENOENT;
		goto put_src;
	}

	if (check_add_overflow(arg->src_off, arg->len, &end) || end > src->size ||
	    check_add_overflow(arg->dst_off, arg->len, &end) || end > dst->size) {
		ret = -ERANGE;
		goto put;
	}
	if (dst->type == GB10_DMAB_BUF_USER &&
	    (dst->flags & GB10_DMAB_REG_F_READONLY)) {
		ret = -EPERM;
		goto put;
	}

	nr = 1;
	if ((arg->flags & GB10_DMAB_COPY_F_STRIPE) && arg->engine < 0 &&
	    bd->nr_engines > 1)
		nr = clamp_t(u64, arg->len / GB10_DMAB_MIN_STRIPE, 1,
			     bd->nr_engines);

	req = kzalloc(struct_size(req, jobs, nr), GFP_KERNEL);
	if (!req) {
		ret = -ENOMEM;
		goto put;
	}
	req->ctx = ctx;
	req->src = src;
	req->dst = dst;
	req->flags = arg->flags;
	req->len = arg->len;
	req->cookie = atomic64_inc_return(&ctx->next_cookie);
	INIT_LIST_HEAD(&req->node);

	/*
	 * dma-buf CPU access protocol. For the NVIDIA exporter this is a
	 * no-op apart from waiting on any fences attached to the buffer.
	 */
	if (src->type == GB10_DMAB_BUF_DMABUF) {
		ret = dma_buf_begin_cpu_access(src->dmabuf, DMA_FROM_DEVICE);
		if (ret)
			goto free;
	}
	if (dst->type == GB10_DMAB_BUF_DMABUF) {
		ret = dma_buf_begin_cpu_access(dst->dmabuf, DMA_TO_DEVICE);
		if (ret) {
			if (src->type == GB10_DMAB_BUF_DMABUF)
				dma_buf_end_cpu_access(src->dmabuf,
						       DMA_FROM_DEVICE);
			goto free;
		}
	}
	if (arg->flags & GB10_DMAB_COPY_F_FLUSH)
		gb10_dmab_buf_sync(src, DMA_FROM_DEVICE, false);

	chunk = ALIGN(DIV_ROUND_UP(arg->len, nr), SMP_CACHE_BYTES);
	for (i = 0; i < nr; i++) {
		struct gb10_dmab_job *job = &req->jobs[i];
		u64 off = (u64)i * chunk;

		if (off >= arg->len)
			break;
		job->req = req;
		job->src_off = arg->src_off + off;
		job->dst_off = arg->dst_off + off;
		job->len = min(chunk, arg->len - off);
	}
	req->nr_jobs = i;
	atomic_set(&req->remaining, req->nr_jobs);
	atomic_set(&req->status, 0);
	req->t_submit = ktime_get_ns();
	atomic_inc(&ctx->inflight);
	arg->cookie = req->cookie;

	/*
	 * The request can only complete once every job has run, so @req stays
	 * valid throughout this loop even though engines start immediately.
	 */
	for (i = 0; i < req->nr_jobs; i++) {
		struct gb10_dmab_engine *eng;

		if (arg->engine >= 0)
			eng = &bd->engines[arg->engine];
		else if (req->nr_jobs > 1)
			eng = &bd->engines[i];
		else
			eng = engine_pick(bd);
		engine_enqueue(eng, &req->jobs[i]);
	}
	return 0;

free:
	kfree(req);
put:
	gb10_dmab_buf_put(dst);
put_src:
	gb10_dmab_buf_put(src);
	return ret;
}

static void req_free(struct gb10_dmab_req *req)
{
	gb10_dmab_buf_put(req->src);
	gb10_dmab_buf_put(req->dst);
	kfree(req);
}

/* Take one completed request off the done list; @cookie 0 matches any. */
static struct gb10_dmab_req *ctx_reap(struct gb10_dmab_ctx *ctx, u64 cookie)
{
	struct gb10_dmab_req *req, *found = NULL;

	spin_lock(&ctx->lock);
	list_for_each_entry(req, &ctx->done, node) {
		if (!cookie || req->cookie == cookie) {
			list_del(&req->node);
			found = req;
			break;
		}
	}
	spin_unlock(&ctx->lock);
	return found;
}

bool gb10_dmab_has_completions(struct gb10_dmab_ctx *ctx)
{
	return !list_empty_careful(&ctx->done);
}

int gb10_dmab_wait(struct gb10_dmab_ctx *ctx, struct gb10_dmab_wait *arg)
{
	struct gb10_dmab_req *req = NULL;
	ktime_t timeout;
	int status, ret;

	if (arg->cookie && arg->cookie > (u64)atomic64_read(&ctx->next_cookie))
		return -ENOENT;

	timeout = arg->timeout_ns < 0 ? KTIME_MAX : ns_to_ktime(arg->timeout_ns);
	ret = wait_event_interruptible_hrtimeout(ctx->wq,
			(req = ctx_reap(ctx, arg->cookie)) != NULL, timeout);
	if (ret == -ETIME)
		return -ETIMEDOUT;
	if (ret)
		return ret;

	status = atomic_read(&req->status);
	arg->cookie = req->cookie;
	arg->status = status < 0 ? -status : status;
	arg->bytes = req->len;
	arg->elapsed_ns = req->t_done - req->t_submit;
	arg->nr_inflight = atomic_read(&ctx->inflight);
	req_free(req);
	return 0;
}

/* release(): wait for everything in flight, then free all completions. */
void gb10_dmab_ctx_drain(struct gb10_dmab_ctx *ctx)
{
	struct gb10_dmab_req *req;

	wait_event(ctx->wq, atomic_read(&ctx->inflight) == 0);
	/* lock handoff: the last completer has left req_complete() once we get this */
	spin_lock(&ctx->lock);
	spin_unlock(&ctx->lock);
	while ((req = ctx_reap(ctx, 0)))
		req_free(req);
}
