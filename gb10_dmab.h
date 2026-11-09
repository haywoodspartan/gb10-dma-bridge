/* SPDX-License-Identifier: GPL-2.0 */
/*
 * gb10_dmab - internal definitions for the GB10 CPU-core DMA bridge.
 */
#ifndef _GB10_DMAB_INTERNAL_H
#define _GB10_DMAB_INTERNAL_H

#include <linux/types.h>
#include <linux/kref.h>
#include <linux/list.h>
#include <linux/spinlock.h>
#include <linux/wait.h>
#include <linux/xarray.h>
#include <linux/atomic.h>
#include <linux/cpumask.h>
#include <linux/miscdevice.h>
#include <linux/dma-direction.h>
#include <linux/iosys-map.h>

#include "uapi/gb10_dmab.h"

#define GB10_DMAB_NAME		"gb10_dmab"
#define GB10_DMAB_VERSION	"1.0.0"

struct gb10_dmab_dev;
struct gb10_dmab_ctx;
struct gb10_dmab_req;

/*
 * A copy endpoint. Exactly one of @kva / @mmio is set once the buffer is
 * usable. @sgt describes the backing pages for the DMA API (FLUSH support)
 * and is mapped for @dma_dev.
 */
struct gb10_dmab_buf {
	struct kref kref;
	struct gb10_dmab_dev *bd;
	u32 type;			/* GB10_DMAB_BUF_* */
	u32 flags;			/* registration flags */
	u64 size;

	void *kva;			/* normal memory */
	void __iomem *mmio;		/* device memory */

	struct device *dma_dev;
	struct sg_table *sgt;
	void *vmap_base;		/* our own vmap() to undo, if any */

	/* GB10_DMAB_BUF_USER */
	struct page **pages;
	unsigned long npages;
	struct mm_struct *mm;		/* for locked_vm accounting */

	/* GB10_DMAB_BUF_DMABUF */
	struct dma_buf *dmabuf;
	struct dma_buf_attachment *attach;
	struct iosys_map map;
	bool exporter_vmap;

	/* GB10_DMAB_BUF_DMABUF (peer) and GB10_DMAB_BUF_PCI_BAR (target) */
	struct pci_dev *pdev;
	bool pci_enabled;
};

/* One dedicated CPU running one kernel thread that executes copy jobs. */
struct gb10_dmab_engine {
	struct gb10_dmab_dev *bd;
	unsigned int index;
	unsigned int cpu;
	struct task_struct *task;

	spinlock_t lock;
	struct list_head queue;		/* gb10_dmab_job.node */
	wait_queue_head_t wq;
	atomic_t queued;
	void *bounce;			/* MMIO -> MMIO staging */

	/* statistics, written by the engine thread only */
	u64 jobs;
	u64 bytes;
	u64 busy_ns;
	u64 sleeps;
};

/* A contiguous slice of a request executed by one engine. */
struct gb10_dmab_job {
	struct list_head node;
	struct gb10_dmab_req *req;
	u64 src_off;
	u64 dst_off;
	u64 len;
	int status;
};

/* A user copy request; completes when all of its jobs have run. */
struct gb10_dmab_req {
	struct list_head node;		/* gb10_dmab_ctx.done */
	struct gb10_dmab_ctx *ctx;
	struct gb10_dmab_buf *src;
	struct gb10_dmab_buf *dst;
	u64 cookie;
	u32 flags;
	u64 len;
	u64 t_submit;
	u64 t_done;
	atomic_t remaining;
	atomic_t status;		/* first error, negative errno */
	unsigned int nr_jobs;
	struct gb10_dmab_job jobs[];
};

/* Per-open-file state. */
struct gb10_dmab_ctx {
	struct gb10_dmab_dev *bd;
	struct xarray bufs;		/* handle -> gb10_dmab_buf */
	spinlock_t lock;
	struct list_head done;		/* completed, not yet reaped */
	wait_queue_head_t wq;
	atomic_t inflight;		/* submitted, not yet completed */
	atomic64_t next_cookie;
};

struct gb10_dmab_dev {
	struct miscdevice misc;
	struct platform_device *pdev;	/* pseudo device: DMA mapping of USER buffers */
	struct pci_dev *default_peer;	/* dma-buf importer when the caller names none */

	cpumask_var_t cpus;
	unsigned int nr_engines;
	struct gb10_dmab_engine *engines;
	atomic_t rr;

	unsigned int poll_us;
	unsigned int chunk;
	bool rt;
	bool io_coherent;
	bool try_vmap;
	bool isolated;
	bool nohz_full;
};

static inline struct gb10_dmab_dev *to_bd(struct miscdevice *misc)
{
	return container_of(misc, struct gb10_dmab_dev, misc);
}

/* gb10_dmab_buf.c */
int gb10_dmab_buf_reg_user(struct gb10_dmab_ctx *ctx, struct gb10_dmab_reg_user *arg);
int gb10_dmab_buf_import_dmabuf(struct gb10_dmab_ctx *ctx, struct gb10_dmab_import_dmabuf *arg);
int gb10_dmab_buf_map_bar(struct gb10_dmab_ctx *ctx, struct gb10_dmab_map_bar *arg);
int gb10_dmab_buf_info(struct gb10_dmab_ctx *ctx, struct gb10_dmab_buf_info *arg);
int gb10_dmab_buf_unreg(struct gb10_dmab_ctx *ctx, u32 handle);
struct gb10_dmab_buf *gb10_dmab_buf_get(struct gb10_dmab_ctx *ctx, u32 handle);
void gb10_dmab_buf_put(struct gb10_dmab_buf *buf);
void gb10_dmab_buf_release_all(struct gb10_dmab_ctx *ctx);
void gb10_dmab_buf_sync(struct gb10_dmab_buf *buf, enum dma_data_direction dir, bool for_device);

/* gb10_dmab_engine.c */
int gb10_dmab_engines_start(struct gb10_dmab_dev *bd);
void gb10_dmab_engines_stop(struct gb10_dmab_dev *bd);
int gb10_dmab_submit(struct gb10_dmab_ctx *ctx, struct gb10_dmab_copy *arg);
int gb10_dmab_wait(struct gb10_dmab_ctx *ctx, struct gb10_dmab_wait *arg);
bool gb10_dmab_has_completions(struct gb10_dmab_ctx *ctx);
void gb10_dmab_ctx_drain(struct gb10_dmab_ctx *ctx);

#endif /* _GB10_DMAB_INTERNAL_H */
