// SPDX-License-Identifier: GPL-2.0
/*
 * gb10_dmab - copy endpoints: pinned user memory, imported dma-bufs and
 * PCI BAR windows.
 */
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/module.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include <linux/sched/mm.h>
#include <linux/vmalloc.h>
#include <linux/scatterlist.h>
#include <linux/dma-mapping.h>
#include <linux/dma-buf.h>
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/capability.h>
#include <linux/overflow.h>
#include <linux/platform_device.h>
#include <linux/uaccess.h>
#include <linux/sizes.h>

#include "gb10_dmab.h"

/* pages per pin_user_pages_fast() call: 128 MiB of 4 KiB pages */
#define GB10_DMAB_PIN_BATCH	(SZ_128M >> PAGE_SHIFT)

static struct gb10_dmab_buf *buf_alloc(struct gb10_dmab_dev *bd, u32 type)
{
	struct gb10_dmab_buf *buf = kzalloc(sizeof(*buf), GFP_KERNEL);

	if (!buf)
		return NULL;
	kref_init(&buf->kref);
	buf->bd = bd;
	buf->type = type;
	return buf;
}

static int buf_install(struct gb10_dmab_ctx *ctx, struct gb10_dmab_buf *buf,
		       u32 *handle)
{
	u32 id;
	int ret;

	ret = xa_alloc(&ctx->bufs, &id, buf, XA_LIMIT(1, U32_MAX - 1),
		       GFP_KERNEL);
	if (ret)
		return ret;
	*handle = id;
	return 0;
}

/*
 * Resolve a PCI function named by userspace. An all-zero address selects the
 * module default peer when @allow_default is set.
 */
static struct pci_dev *lookup_pci(struct gb10_dmab_dev *bd,
				  const struct gb10_dmab_pci_addr *a,
				  bool allow_default)
{
	if (!a->domain && !a->bus && !a->devfn) {
		if (!allow_default || !bd->default_peer)
			return NULL;
		return pci_dev_get(bd->default_peer);
	}
	return pci_get_domain_bus_and_slot(a->domain, a->bus, a->devfn);
}

/* ------------------------------------------------------------------------
 * GB10_DMAB_BUF_USER
 * ------------------------------------------------------------------------ */

static void buf_release_user(struct gb10_dmab_buf *buf, unsigned long pinned)
{
	if (buf->sgt) {
		dma_unmap_sgtable(buf->dma_dev, buf->sgt, DMA_BIDIRECTIONAL,
				  DMA_ATTR_SKIP_CPU_SYNC);
		sg_free_table(buf->sgt);
		kfree(buf->sgt);
	}
	if (buf->vmap_base)
		vunmap(buf->vmap_base);
	if (buf->pages) {
		unpin_user_pages_dirty_lock(buf->pages, pinned,
				!(buf->flags & GB10_DMAB_REG_F_READONLY));
		kvfree(buf->pages);
	}
	if (buf->mm) {
		account_locked_vm(buf->mm, buf->npages, false);
		mmdrop(buf->mm);
	}
}

int gb10_dmab_buf_reg_user(struct gb10_dmab_ctx *ctx,
			   struct gb10_dmab_reg_user *arg)
{
	struct gb10_dmab_dev *bd = ctx->bd;
	struct gb10_dmab_buf *buf;
	unsigned long start, end, npages, pinned = 0;
	unsigned int gup_flags = FOLL_LONGTERM;
	int ret;

	if (!arg->len || (arg->flags & ~GB10_DMAB_REG_F_READONLY))
		return -EINVAL;
	if (check_add_overflow((unsigned long)arg->addr,
			       (unsigned long)arg->len, &end))
		return -EINVAL;
	if (!access_ok((void __user *)(unsigned long)arg->addr, arg->len))
		return -EFAULT;

	start = arg->addr & PAGE_MASK;
	npages = (PAGE_ALIGN(end) - start) >> PAGE_SHIFT;
	if (npages > UINT_MAX)
		return -E2BIG;
	if (!(arg->flags & GB10_DMAB_REG_F_READONLY))
		gup_flags |= FOLL_WRITE;

	buf = buf_alloc(bd, GB10_DMAB_BUF_USER);
	if (!buf)
		return -ENOMEM;
	buf->flags = arg->flags;
	buf->size = arg->len;
	buf->dma_dev = &bd->pdev->dev;
	buf->npages = npages;

	/* Respect RLIMIT_MEMLOCK like vfio does; CAP_IPC_LOCK bypasses. */
	ret = account_locked_vm(current->mm, npages, true);
	if (ret)
		goto err_free;
	mmgrab(current->mm);
	buf->mm = current->mm;

	buf->pages = kvmalloc_array(npages, sizeof(*buf->pages), GFP_KERNEL);
	if (!buf->pages) {
		ret = -ENOMEM;
		goto err_release;
	}

	while (pinned < npages) {
		unsigned long n = min_t(unsigned long, npages - pinned,
					GB10_DMAB_PIN_BATCH);

		ret = pin_user_pages_fast(start + (pinned << PAGE_SHIFT), n,
					  gup_flags, buf->pages + pinned);
		if (ret <= 0) {
			ret = ret ? ret : -EFAULT;
			goto err_release;
		}
		pinned += ret;
	}

	buf->vmap_base = vmap(buf->pages, npages, VM_MAP, PAGE_KERNEL);
	if (!buf->vmap_base) {
		ret = -ENOMEM;
		goto err_release;
	}
	buf->kva = buf->vmap_base + (arg->addr & ~PAGE_MASK);

	buf->sgt = kzalloc(sizeof(*buf->sgt), GFP_KERNEL);
	if (!buf->sgt) {
		ret = -ENOMEM;
		goto err_release;
	}
	ret = sg_alloc_table_from_pages_segment(buf->sgt, buf->pages, npages, 0,
						npages << PAGE_SHIFT, UINT_MAX,
						GFP_KERNEL);
	if (ret) {
		kfree(buf->sgt);
		buf->sgt = NULL;
		goto err_release;
	}
	ret = dma_map_sgtable(buf->dma_dev, buf->sgt, DMA_BIDIRECTIONAL,
			      DMA_ATTR_SKIP_CPU_SYNC);
	if (ret) {
		sg_free_table(buf->sgt);
		kfree(buf->sgt);
		buf->sgt = NULL;
		goto err_release;
	}

	ret = buf_install(ctx, buf, &arg->handle);
	if (ret)
		goto err_release;
	return 0;

err_release:
	buf_release_user(buf, pinned);
err_free:
	kfree(buf);
	return ret;
}

/* ------------------------------------------------------------------------
 * GB10_DMAB_BUF_DMABUF
 * ------------------------------------------------------------------------ */

/*
 * The NVIDIA open-kernel exporter has no vmap()/begin_cpu_access() hooks,
 * but on coherent GB10 (nv->coherent, "Addressing Mode: ATS") it fills the
 * scatter-list with real struct pages of the unified LPDDR5X pool. Build a
 * kernel mapping from those pages.
 */
static int dmabuf_map_pages(struct gb10_dmab_buf *buf)
{
	struct sg_table *sgt = buf->sgt;
	struct scatterlist *sg;
	struct page **pages;
	unsigned long npages = DIV_ROUND_UP(buf->size, PAGE_SIZE), k = 0;
	unsigned int i;
	int ret = 0;

	if (npages > UINT_MAX)
		return -E2BIG;
	pages = kvmalloc_array(npages, sizeof(*pages), GFP_KERNEL);
	if (!pages)
		return -ENOMEM;

	for_each_sgtable_sg(sgt, sg, i) {
		struct page *page = sg_page(sg);
		unsigned long n, j;

		/* PFN-only (MMIO/BAR1) exporters cannot be reached by the CPU here */
		if (!page || sg->offset) {
			ret = -EOPNOTSUPP;
			goto out;
		}
		n = DIV_ROUND_UP(sg->length, PAGE_SIZE);
		if ((sg->length & ~PAGE_MASK) && k + n < npages) {
			ret = -EOPNOTSUPP;	/* partial page mid-table */
			goto out;
		}
		/* pages of one segment are physically contiguous: contiguous memmap */
		for (j = 0; j < n && k < npages; j++)
			pages[k++] = page + j;
	}
	if (k < npages) {
		ret = -EINVAL;
		goto out;
	}

	buf->vmap_base = vmap(pages, npages, VM_MAP, PAGE_KERNEL);
	if (!buf->vmap_base)
		ret = -ENOMEM;
	else
		buf->kva = buf->vmap_base;
out:
	kvfree(pages);
	return ret;
}

static void buf_release_dmabuf(struct gb10_dmab_buf *buf)
{
	if (buf->exporter_vmap)
		dma_buf_vunmap_unlocked(buf->dmabuf, &buf->map);
	else if (buf->vmap_base)
		vunmap(buf->vmap_base);
	if (buf->sgt)
		dma_buf_unmap_attachment_unlocked(buf->attach, buf->sgt,
						  DMA_BIDIRECTIONAL);
	if (buf->attach)
		dma_buf_detach(buf->dmabuf, buf->attach);
	if (buf->dmabuf)
		dma_buf_put(buf->dmabuf);
	if (buf->pdev)
		pci_dev_put(buf->pdev);
}

int gb10_dmab_buf_import_dmabuf(struct gb10_dmab_ctx *ctx,
				struct gb10_dmab_import_dmabuf *arg)
{
	struct gb10_dmab_dev *bd = ctx->bd;
	struct gb10_dmab_buf *buf;
	struct dma_buf *dmabuf;
	struct dma_buf_attachment *attach;
	struct sg_table *sgt;
	struct pci_dev *peer;
	int ret;

	if (arg->flags)
		return -EINVAL;

	peer = lookup_pci(bd, &arg->peer, true);
	if (!peer)
		return -ENODEV;

	dmabuf = dma_buf_get(arg->fd);
	if (IS_ERR(dmabuf)) {
		pci_dev_put(peer);
		return PTR_ERR(dmabuf);
	}

	buf = buf_alloc(bd, GB10_DMAB_BUF_DMABUF);
	if (!buf) {
		dma_buf_put(dmabuf);
		pci_dev_put(peer);
		return -ENOMEM;
	}
	buf->dmabuf = dmabuf;
	buf->pdev = peer;
	buf->dma_dev = &peer->dev;
	buf->size = dmabuf->size;

	attach = dma_buf_attach(dmabuf, &peer->dev);
	if (IS_ERR(attach)) {
		ret = PTR_ERR(attach);
		pr_warn("dma-buf attach for peer %s failed: %d\n",
			pci_name(peer), ret);
		goto err;
	}
	buf->attach = attach;

	sgt = dma_buf_map_attachment_unlocked(attach, DMA_BIDIRECTIONAL);
	if (IS_ERR(sgt)) {
		ret = PTR_ERR(sgt);
		pr_warn("dma-buf map for peer %s failed: %d\n",
			pci_name(peer), ret);
		goto err;
	}
	buf->sgt = sgt;

	if (bd->try_vmap && !dma_buf_vmap_unlocked(dmabuf, &buf->map)) {
		buf->exporter_vmap = true;
		if (buf->map.is_iomem)
			buf->mmio = buf->map.vaddr_iomem;
		else
			buf->kva = buf->map.vaddr;
	} else {
		ret = dmabuf_map_pages(buf);
		if (ret) {
			pr_warn("dma-buf fd %d: no CPU mapping possible (%d); exporter has neither vmap nor struct pages\n",
				arg->fd, ret);
			goto err;
		}
	}

	arg->size = buf->size;
	ret = buf_install(ctx, buf, &arg->handle);
	if (ret)
		goto err;
	return 0;

err:
	buf_release_dmabuf(buf);
	kfree(buf);
	return ret;
}

/* ------------------------------------------------------------------------
 * GB10_DMAB_BUF_PCI_BAR
 * ------------------------------------------------------------------------ */

static void buf_release_bar(struct gb10_dmab_buf *buf)
{
	if (buf->mmio)
		pci_iounmap(buf->pdev, buf->mmio);
	if (buf->pci_enabled)
		pci_disable_device(buf->pdev);
	if (buf->pdev)
		pci_dev_put(buf->pdev);
}

int gb10_dmab_buf_map_bar(struct gb10_dmab_ctx *ctx,
			  struct gb10_dmab_map_bar *arg)
{
	struct gb10_dmab_dev *bd = ctx->bd;
	struct gb10_dmab_buf *buf;
	struct pci_dev *pdev;
	resource_size_t bar_size;
	unsigned long res_flags;
	u64 len, end;
	int ret;

	if (!capable(CAP_SYS_RAWIO))
		return -EPERM;
	if (arg->bar >= PCI_STD_NUM_BARS || (arg->flags & ~GB10_DMAB_BAR_F_WC))
		return -EINVAL;

	pdev = lookup_pci(bd, &arg->dev, false);
	if (!pdev)
		return -ENODEV;

	res_flags = pci_resource_flags(pdev, arg->bar);
	bar_size = pci_resource_len(pdev, arg->bar);
	if (!(res_flags & IORESOURCE_MEM) || !bar_size ||
	    arg->offset >= bar_size) {
		ret = -EINVAL;
		goto err_put;
	}
	len = arg->len ? arg->len : bar_size - arg->offset;
	if (check_add_overflow(arg->offset, len, &end) || end > bar_size) {
		ret = -EINVAL;
		goto err_put;
	}
	if ((arg->flags & GB10_DMAB_BAR_F_WC) &&
	    !(res_flags & IORESOURCE_PREFETCH)) {
		ret = -EINVAL;
		goto err_put;
	}

	buf = buf_alloc(bd, GB10_DMAB_BUF_PCI_BAR);
	if (!buf) {
		ret = -ENOMEM;
		goto err_put;
	}
	buf->pdev = pdev;
	buf->flags = arg->flags;
	buf->size = len;

	/* Only touch the enable state of driverless, disabled functions. */
	if (!pci_is_enabled(pdev)) {
		ret = pci_enable_device_mem(pdev);
		if (ret)
			goto err_release;
		buf->pci_enabled = true;
	}

	if (arg->flags & GB10_DMAB_BAR_F_WC)
		buf->mmio = pci_iomap_wc_range(pdev, arg->bar, arg->offset, len);
	else
		buf->mmio = pci_iomap_range(pdev, arg->bar, arg->offset, len);
	if (!buf->mmio) {
		ret = -ENOMEM;
		goto err_release;
	}

	arg->bar_size = bar_size;
	ret = buf_install(ctx, buf, &arg->handle);
	if (ret)
		goto err_release;
	return 0;

err_release:
	buf_release_bar(buf);	/* drops the pdev reference */
	kfree(buf);
	return ret;
err_put:
	pci_dev_put(pdev);
	return ret;
}

/* ------------------------------------------------------------------------
 * common
 * ------------------------------------------------------------------------ */

static void buf_release(struct kref *kref)
{
	struct gb10_dmab_buf *buf = container_of(kref, struct gb10_dmab_buf, kref);

	switch (buf->type) {
	case GB10_DMAB_BUF_USER:
		buf_release_user(buf, buf->npages);
		break;
	case GB10_DMAB_BUF_DMABUF:
		buf_release_dmabuf(buf);
		break;
	case GB10_DMAB_BUF_PCI_BAR:
		buf_release_bar(buf);
		break;
	}
	kfree(buf);
}

void gb10_dmab_buf_put(struct gb10_dmab_buf *buf)
{
	if (buf)
		kref_put(&buf->kref, buf_release);
}

struct gb10_dmab_buf *gb10_dmab_buf_get(struct gb10_dmab_ctx *ctx, u32 handle)
{
	struct gb10_dmab_buf *buf;

	xa_lock(&ctx->bufs);
	buf = xa_load(&ctx->bufs, handle);
	if (buf && !kref_get_unless_zero(&buf->kref))
		buf = NULL;
	xa_unlock(&ctx->bufs);
	return buf;
}

int gb10_dmab_buf_unreg(struct gb10_dmab_ctx *ctx, u32 handle)
{
	struct gb10_dmab_buf *buf = xa_erase(&ctx->bufs, handle);

	if (!buf)
		return -ENOENT;
	gb10_dmab_buf_put(buf);	/* in-flight requests keep their own reference */
	return 0;
}

int gb10_dmab_buf_info(struct gb10_dmab_ctx *ctx, struct gb10_dmab_buf_info *arg)
{
	struct gb10_dmab_buf *buf = gb10_dmab_buf_get(ctx, arg->handle);

	if (!buf)
		return -ENOENT;
	arg->type = buf->type;
	arg->size = buf->size;
	arg->flags = buf->flags;
	arg->nr_segments = buf->sgt ? buf->sgt->orig_nents : 0;
	memset(arg->reserved, 0, sizeof(arg->reserved));
	gb10_dmab_buf_put(buf);
	return 0;
}

void gb10_dmab_buf_release_all(struct gb10_dmab_ctx *ctx)
{
	struct gb10_dmab_buf *buf;
	unsigned long idx;

	xa_for_each(&ctx->bufs, idx, buf) {
		xa_erase(&ctx->bufs, idx);
		gb10_dmab_buf_put(buf);
	}
	xa_destroy(&ctx->bufs);
}

/*
 * Cache maintenance for non-coherent peers. A no-op when the mapping device
 * is marked dma-coherent, which is the case for every peer on GB10.
 */
void gb10_dmab_buf_sync(struct gb10_dmab_buf *buf, enum dma_data_direction dir,
			bool for_device)
{
	if (!buf->sgt || !buf->dma_dev)
		return;
	if (for_device)
		dma_sync_sgtable_for_device(buf->dma_dev, buf->sgt, dir);
	else
		dma_sync_sgtable_for_cpu(buf->dma_dev, buf->sgt, dir);
}
