// SPDX-License-Identifier: GPL-2.0
/*
 * gb10_dmab - CPU-core DMA bridge for the NVIDIA GB10 Grace Blackwell SoC
 *             (DGX Spark)
 *
 * Why this exists
 * ---------------
 * On GB10 the Blackwell GPU has no framebuffer of its own and no PCIe data
 * path: it sits behind a 2.5 GT/s x1 management link (BAR0 only) and reaches
 * the unified 128 GB LPDDR5X pool over NVLink-C2C in ATS addressing mode.
 * PCIe endpoints (ConnectX-7, NVMe, ...) hang off root ports on the CPU die
 * and DMA into that same pool through the SMMU. The hardware therefore
 * already joins GPU and PCIe through memory; what it does not offer is a
 * DMA engine the host can program to move data between arbitrary endpoints
 * without burning application cores.
 *
 * This driver dedicates a set of CPU cores as software DMA engines. Each
 * engine is a SCHED_FIFO kernel thread bound to one core, draining a queue
 * of copy jobs between three kinds of endpoints:
 *
 *   USER     pinned user pages (malloc, cudaHostAlloc, cudaMallocManaged)
 *   DMABUF   imported dma-bufs, e.g. cudaMalloc() memory exported with
 *            cuMemGetHandleForAddressRange(CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD)
 *   PCI_BAR  MMIO windows of PCI devices (FPGAs, driverless accelerators)
 *
 * Pair it with isolcpus=/nohz_full=/rcu_nocbs= for the same cores so the
 * engines own them outright (scripts/isolate-cores.sh).
 */
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/module.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/platform_device.h>
#include <linux/dma-mapping.h>
#include <linux/pci.h>
#include <linux/poll.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/sched/isolation.h>
#include <linux/cpumask.h>
#include <linux/sysfs.h>
#include <linux/sizes.h>
#include <linux/string.h>
#include <linux/minmax.h>

#include "gb10_dmab.h"

static char *cores = "16-19";
module_param(cores, charp, 0444);
MODULE_PARM_DESC(cores, "CPUs dedicated as copy engines, cpulist (default 16-19: four Cortex-X925 cores of the second cluster)");

static unsigned int poll_us = 200;
module_param(poll_us, uint, 0444);
MODULE_PARM_DESC(poll_us, "busy-poll window before an idle engine sleeps, microseconds (default 200, 0 = never spin)");

static bool rt = true;
module_param(rt, bool, 0444);
MODULE_PARM_DESC(rt, "run engines SCHED_FIFO (default Y)");

static bool io_coherent = true;
module_param(io_coherent, bool, 0444);
MODULE_PARM_DESC(io_coherent, "peers are I/O coherent so FLUSH is a no-op (default Y, correct for GB10)");

static bool require_isolation;
module_param(require_isolation, bool, 0444);
MODULE_PARM_DESC(require_isolation, "refuse to load unless every engine CPU is in isolcpus=domain (default N)");

static bool try_vmap = true;
module_param(try_vmap, bool, 0444);
MODULE_PARM_DESC(try_vmap, "use the exporter's vmap for dma-bufs when it has one (default Y)");

static unsigned int chunk_kb = 2048;
module_param(chunk_kb, uint, 0444);
MODULE_PARM_DESC(chunk_kb, "copy granularity between preemption points, KiB (default 2048)");

static char *peer_bdf;
module_param(peer_bdf, charp, 0444);
MODULE_PARM_DESC(peer_bdf, "default dma-buf importer as DDDD:BB:DD.F (default: first NVIDIA display-class function)");

static struct gb10_dmab_dev *gb10_dmab;

/* ------------------------------------------------------------------------
 * file operations
 * ------------------------------------------------------------------------ */

static int gb10_dmab_open(struct inode *inode, struct file *file)
{
	struct gb10_dmab_dev *bd = to_bd(file->private_data);
	struct gb10_dmab_ctx *ctx;

	ctx = kzalloc(sizeof(*ctx), GFP_KERNEL);
	if (!ctx)
		return -ENOMEM;
	ctx->bd = bd;
	xa_init_flags(&ctx->bufs, XA_FLAGS_ALLOC1);
	spin_lock_init(&ctx->lock);
	INIT_LIST_HEAD(&ctx->done);
	init_waitqueue_head(&ctx->wq);
	atomic_set(&ctx->inflight, 0);
	atomic64_set(&ctx->next_cookie, 0);
	file->private_data = ctx;
	return stream_open(inode, file);
}

static int gb10_dmab_release(struct inode *inode, struct file *file)
{
	struct gb10_dmab_ctx *ctx = file->private_data;

	gb10_dmab_ctx_drain(ctx);
	gb10_dmab_buf_release_all(ctx);
	kfree(ctx);
	return 0;
}

static __poll_t gb10_dmab_poll(struct file *file, poll_table *wait)
{
	struct gb10_dmab_ctx *ctx = file->private_data;
	__poll_t mask = 0;

	poll_wait(file, &ctx->wq, wait);
	if (gb10_dmab_has_completions(ctx))
		mask |= EPOLLIN | EPOLLRDNORM;
	return mask;
}

static void fill_pci_addr(struct gb10_dmab_pci_addr *a, struct pci_dev *pdev)
{
	memset(a, 0, sizeof(*a));
	if (!pdev)
		return;
	a->domain = pci_domain_nr(pdev->bus);
	a->bus = pdev->bus->number;
	a->devfn = pdev->devfn;
}

static void gb10_dmab_fill_info(struct gb10_dmab_dev *bd,
				struct gb10_dmab_info *info)
{
	unsigned int i;

	memset(info, 0, sizeof(*info));
	info->abi_version = GB10_DMAB_ABI_VERSION;
	info->nr_engines = bd->nr_engines;
	info->engine_cpumask = cpumask_bits(bd->cpus)[0];
	if (bd->io_coherent)
		info->flags |= GB10_DMAB_INFO_F_IO_COHERENT;
	if (bd->rt)
		info->flags |= GB10_DMAB_INFO_F_RT;
	if (bd->isolated)
		info->flags |= GB10_DMAB_INFO_F_ISOLATED;
	if (bd->nohz_full)
		info->flags |= GB10_DMAB_INFO_F_NOHZ_FULL;
	info->poll_us = bd->poll_us;
	info->chunk_bytes = bd->chunk;
	fill_pci_addr(&info->default_peer, bd->default_peer);
	for (i = 0; i < bd->nr_engines; i++) {
		info->bytes_copied += READ_ONCE(bd->engines[i].bytes);
		info->jobs_completed += READ_ONCE(bd->engines[i].jobs);
	}
}

static long gb10_dmab_ioctl(struct file *file, unsigned int cmd,
			    unsigned long arg)
{
	struct gb10_dmab_ctx *ctx = file->private_data;
	void __user *uarg = (void __user *)arg;
	int ret;

	switch (cmd) {
	case GB10_DMAB_IOC_INFO: {
		struct gb10_dmab_info info;

		gb10_dmab_fill_info(ctx->bd, &info);
		return copy_to_user(uarg, &info, sizeof(info)) ? -EFAULT : 0;
	}
	case GB10_DMAB_IOC_REG_USER: {
		struct gb10_dmab_reg_user a;

		if (copy_from_user(&a, uarg, sizeof(a)))
			return -EFAULT;
		ret = gb10_dmab_buf_reg_user(ctx, &a);
		if (ret)
			return ret;
		if (copy_to_user(uarg, &a, sizeof(a))) {
			gb10_dmab_buf_unreg(ctx, a.handle);
			return -EFAULT;
		}
		return 0;
	}
	case GB10_DMAB_IOC_IMPORT_DMABUF: {
		struct gb10_dmab_import_dmabuf a;

		if (copy_from_user(&a, uarg, sizeof(a)))
			return -EFAULT;
		ret = gb10_dmab_buf_import_dmabuf(ctx, &a);
		if (ret)
			return ret;
		if (copy_to_user(uarg, &a, sizeof(a))) {
			gb10_dmab_buf_unreg(ctx, a.handle);
			return -EFAULT;
		}
		return 0;
	}
	case GB10_DMAB_IOC_MAP_BAR: {
		struct gb10_dmab_map_bar a;

		if (copy_from_user(&a, uarg, sizeof(a)))
			return -EFAULT;
		ret = gb10_dmab_buf_map_bar(ctx, &a);
		if (ret)
			return ret;
		if (copy_to_user(uarg, &a, sizeof(a))) {
			gb10_dmab_buf_unreg(ctx, a.handle);
			return -EFAULT;
		}
		return 0;
	}
	case GB10_DMAB_IOC_UNREG: {
		u32 handle;

		if (get_user(handle, (u32 __user *)uarg))
			return -EFAULT;
		return gb10_dmab_buf_unreg(ctx, handle);
	}
	case GB10_DMAB_IOC_BUF_INFO: {
		struct gb10_dmab_buf_info a;

		if (copy_from_user(&a, uarg, sizeof(a)))
			return -EFAULT;
		ret = gb10_dmab_buf_info(ctx, &a);
		if (ret)
			return ret;
		return copy_to_user(uarg, &a, sizeof(a)) ? -EFAULT : 0;
	}
	case GB10_DMAB_IOC_COPY: {
		struct gb10_dmab_copy c;

		if (copy_from_user(&c, uarg, sizeof(c)))
			return -EFAULT;
		c.cookie = 0;
		c.status = 0;
		c.elapsed_ns = 0;
		ret = gb10_dmab_submit(ctx, &c);
		if (!ret && (c.flags & GB10_DMAB_COPY_F_WAIT)) {
			struct gb10_dmab_wait w = {
				.cookie = c.cookie,
				.timeout_ns = -1,
			};

			ret = gb10_dmab_wait(ctx, &w);
			if (!ret) {
				c.status = w.status;
				c.elapsed_ns = w.elapsed_ns;
			} else if (ret == -ERESTARTSYS) {
				/* still in flight: hand back the cookie, no restart */
				ret = -EINTR;
			}
		}
		if (copy_to_user(uarg, &c, sizeof(c)))
			return -EFAULT;
		return ret;
	}
	case GB10_DMAB_IOC_WAIT: {
		struct gb10_dmab_wait w;

		if (copy_from_user(&w, uarg, sizeof(w)))
			return -EFAULT;
		ret = gb10_dmab_wait(ctx, &w);
		if (ret)
			return ret;
		return copy_to_user(uarg, &w, sizeof(w)) ? -EFAULT : 0;
	}
	default:
		return -ENOTTY;
	}
}

static const struct file_operations gb10_dmab_fops = {
	.owner		= THIS_MODULE,
	.open		= gb10_dmab_open,
	.release	= gb10_dmab_release,
	.unlocked_ioctl	= gb10_dmab_ioctl,
	.compat_ioctl	= compat_ptr_ioctl,
	.poll		= gb10_dmab_poll,
};

/* ------------------------------------------------------------------------
 * sysfs: /sys/class/misc/gb10_dmab/{engines,cores,isolation,default_peer}
 * ------------------------------------------------------------------------ */

static struct gb10_dmab_dev *dev_to_bd(struct device *dev)
{
	return to_bd(dev_get_drvdata(dev));
}

static ssize_t engines_show(struct device *dev, struct device_attribute *attr,
			    char *buf)
{
	struct gb10_dmab_dev *bd = dev_to_bd(dev);
	ssize_t n;
	unsigned int i;

	n = sysfs_emit(buf, "engine cpu jobs bytes busy_ns queued sleeps\n");
	for (i = 0; i < bd->nr_engines; i++) {
		struct gb10_dmab_engine *e = &bd->engines[i];

		n += sysfs_emit_at(buf, n, "%u %u %llu %llu %llu %d %llu\n",
				   e->index, e->cpu, READ_ONCE(e->jobs),
				   READ_ONCE(e->bytes), READ_ONCE(e->busy_ns),
				   atomic_read(&e->queued), READ_ONCE(e->sleeps));
	}
	return n;
}
static DEVICE_ATTR_RO(engines);

static ssize_t cores_show(struct device *dev, struct device_attribute *attr,
			  char *buf)
{
	struct gb10_dmab_dev *bd = dev_to_bd(dev);

	return sysfs_emit(buf, "%*pbl\n", cpumask_pr_args(bd->cpus));
}
static DEVICE_ATTR_RO(cores);

static ssize_t isolation_show(struct device *dev, struct device_attribute *attr,
			      char *buf)
{
	struct gb10_dmab_dev *bd = dev_to_bd(dev);

	return sysfs_emit(buf, "isolcpus=%s nohz_full=%s sched=%s poll_us=%u\n",
			  bd->isolated ? "yes" : "no",
			  bd->nohz_full ? "yes" : "no",
			  bd->rt ? "fifo" : "normal", bd->poll_us);
}
static DEVICE_ATTR_RO(isolation);

static ssize_t default_peer_show(struct device *dev,
				 struct device_attribute *attr, char *buf)
{
	struct gb10_dmab_dev *bd = dev_to_bd(dev);

	if (!bd->default_peer)
		return sysfs_emit(buf, "none\n");
	return sysfs_emit(buf, "%s\n", pci_name(bd->default_peer));
}
static DEVICE_ATTR_RO(default_peer);

static struct attribute *gb10_dmab_attrs[] = {
	&dev_attr_engines.attr,
	&dev_attr_cores.attr,
	&dev_attr_isolation.attr,
	&dev_attr_default_peer.attr,
	NULL,
};
ATTRIBUTE_GROUPS(gb10_dmab);

/* ------------------------------------------------------------------------
 * module setup
 * ------------------------------------------------------------------------ */

/*
 * The scheduler-level view of isolcpus=/nohz_full=. housekeeping_cpumask()
 * returns the CPUs the kernel may still use for a purpose, so an engine CPU
 * is isolated iff it is absent from that mask.
 */
static void gb10_dmab_check_isolation(struct gb10_dmab_dev *bd)
{
	bd->isolated = housekeeping_enabled(HK_TYPE_DOMAIN) &&
		       !cpumask_intersects(bd->cpus,
					   housekeeping_cpumask(HK_TYPE_DOMAIN));
	bd->nohz_full = housekeeping_enabled(HK_TYPE_KERNEL_NOISE) &&
			!cpumask_intersects(bd->cpus,
					    housekeeping_cpumask(HK_TYPE_KERNEL_NOISE));
}

static struct pci_dev *gb10_dmab_find_peer(void)
{
	struct pci_dev *pdev = NULL;
	unsigned int dom, bus, slot, fn;

	if (peer_bdf && *peer_bdf) {
		if (sscanf(peer_bdf, "%x:%x:%x.%x", &dom, &bus, &slot, &fn) != 4 ||
		    bus > 255 || slot > 31 || fn > 7) {
			pr_err("peer_bdf=%s: expected DDDD:BB:DD.F\n", peer_bdf);
			return ERR_PTR(-EINVAL);
		}
		pdev = pci_get_domain_bus_and_slot(dom, bus, PCI_DEVFN(slot, fn));
		if (!pdev) {
			pr_err("peer_bdf=%s: no such PCI function\n", peer_bdf);
			return ERR_PTR(-ENODEV);
		}
		return pdev;
	}

	for_each_pci_dev(pdev) {
		if (pdev->vendor == PCI_VENDOR_ID_NVIDIA &&
		    (pdev->class >> 16) == PCI_BASE_CLASS_DISPLAY)
			return pdev;	/* reference held */
	}
	return NULL;
}

static int __init gb10_dmab_init(void)
{
	struct platform_device_info pinfo = {
		.name = GB10_DMAB_NAME,
		.id = PLATFORM_DEVID_NONE,
		.dma_mask = DMA_BIT_MASK(64),
	};
	struct gb10_dmab_dev *bd;
	struct pci_dev *peer;
	int ret;

	bd = kzalloc(sizeof(*bd), GFP_KERNEL);
	if (!bd)
		return -ENOMEM;
	if (!zalloc_cpumask_var(&bd->cpus, GFP_KERNEL)) {
		ret = -ENOMEM;
		goto err_free;
	}

	ret = cpulist_parse(cores, bd->cpus);
	if (ret || cpumask_empty(bd->cpus)) {
		pr_err("cores=%s: invalid cpulist\n", cores);
		ret = -EINVAL;
		goto err_mask;
	}
	if (!cpumask_subset(bd->cpus, cpu_online_mask)) {
		pr_err("cores=%s: includes offline CPUs (online: %*pbl)\n",
		       cores, cpumask_pr_args(cpu_online_mask));
		ret = -ENODEV;
		goto err_mask;
	}
	if (cpumask_weight(bd->cpus) >= num_online_cpus()) {
		pr_err("cores=%s: refusing to dedicate every online CPU\n", cores);
		ret = -EINVAL;
		goto err_mask;
	}
	if (cpumask_last(bd->cpus) >= 64)
		pr_warn("CPUs >= 64 are not representable in gb10_dmab_info.engine_cpumask\n");

	gb10_dmab_check_isolation(bd);
	if (require_isolation && !bd->isolated) {
		pr_err("CPUs %*pbl are not isolated (isolcpus=domain,...) and require_isolation=1\n",
		       cpumask_pr_args(bd->cpus));
		ret = -EPERM;
		goto err_mask;
	}

	bd->poll_us = min_t(unsigned int, poll_us, 100000);
	bd->rt = rt;
	bd->io_coherent = io_coherent;
	bd->try_vmap = try_vmap;
	bd->chunk = clamp_t(unsigned int, chunk_kb, 64, 65536) * SZ_1K;
	atomic_set(&bd->rr, 0);

	/* pseudo device the USER scatter-lists are mapped for */
	bd->pdev = platform_device_register_full(&pinfo);
	if (IS_ERR(bd->pdev)) {
		ret = PTR_ERR(bd->pdev);
		goto err_mask;
	}
	ret = dma_set_mask_and_coherent(&bd->pdev->dev, DMA_BIT_MASK(64));
	if (ret)
		goto err_pdev;
	dma_set_max_seg_size(&bd->pdev->dev, UINT_MAX);
#ifdef CONFIG_ARCH_HAS_SYNC_DMA_FOR_DEVICE
	bd->pdev->dev.dma_coherent = io_coherent;
#endif

	peer = gb10_dmab_find_peer();
	if (IS_ERR(peer)) {
		ret = PTR_ERR(peer);
		goto err_pdev;
	}
	bd->default_peer = peer;

	ret = gb10_dmab_engines_start(bd);
	if (ret)
		goto err_peer;

	bd->misc.minor = MISC_DYNAMIC_MINOR;
	bd->misc.name = GB10_DMAB_NAME;
	bd->misc.fops = &gb10_dmab_fops;
	bd->misc.mode = 0660;
	bd->misc.groups = gb10_dmab_groups;
	bd->misc.parent = &bd->pdev->dev;
	ret = misc_register(&bd->misc);
	if (ret)
		goto err_engines;

	gb10_dmab = bd;
	pr_info("%u copy engine(s) on CPUs %*pbl (%s, poll %u us, chunk %u KiB), peers I/O %scoherent\n",
		bd->nr_engines, cpumask_pr_args(bd->cpus),
		bd->rt ? "SCHED_FIFO" : "SCHED_OTHER", bd->poll_us,
		bd->chunk / SZ_1K, bd->io_coherent ? "" : "non-");
	if (bd->isolated)
		pr_info("engine CPUs are isolated (isolcpus=domain%s)\n",
			bd->nohz_full ? ", nohz_full" : "");
	else
		pr_warn("engine CPUs %*pbl are NOT isolated: the scheduler still places other tasks there. Run scripts/isolate-cores.sh --apply and reboot.\n",
			cpumask_pr_args(bd->cpus));
	if (peer)
		pr_info("default dma-buf peer %s [%04x:%04x]\n",
			pci_name(peer), peer->vendor, peer->device);
	else
		pr_warn("no NVIDIA GPU found; dma-buf imports must name a peer\n");
	return 0;

err_engines:
	gb10_dmab_engines_stop(bd);
err_peer:
	pci_dev_put(bd->default_peer);
err_pdev:
	platform_device_unregister(bd->pdev);
err_mask:
	free_cpumask_var(bd->cpus);
err_free:
	kfree(bd);
	return ret;
}

static void __exit gb10_dmab_exit(void)
{
	struct gb10_dmab_dev *bd = gb10_dmab;

	misc_deregister(&bd->misc);
	gb10_dmab_engines_stop(bd);
	pci_dev_put(bd->default_peer);
	platform_device_unregister(bd->pdev);
	free_cpumask_var(bd->cpus);
	kfree(bd);
}

module_init(gb10_dmab_init);
module_exit(gb10_dmab_exit);

MODULE_AUTHOR("DGX Spark platform team");
MODULE_DESCRIPTION("CPU-core DMA bridge between GPU memory, pinned host memory and PCIe endpoints on NVIDIA GB10");
MODULE_LICENSE("GPL");
MODULE_VERSION(GB10_DMAB_VERSION);
MODULE_IMPORT_NS("DMA_BUF");
