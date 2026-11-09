/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * gb10_dmab - user ABI for the GB10 (NVIDIA DGX Spark) CPU-core DMA bridge.
 *
 * The bridge dedicates a set of CPU cores as software DMA engines that move
 * data between three kinds of endpoints:
 *
 *   USER     pinned pages of the caller's address space (malloc, cudaHostAlloc,
 *            cudaMallocManaged, hugetlbfs, ...)
 *   DMABUF   an imported dma-buf, e.g. a cudaMalloc() allocation exported with
 *            cuMemGetHandleForAddressRange(CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD)
 *   PCI_BAR  an MMIO window of a PCI device BAR (CAP_SYS_RAWIO)
 *
 * All structures are 64-bit aligned and identical on 32/64-bit userspace.
 */
#ifndef _UAPI_GB10_DMAB_H
#define _UAPI_GB10_DMAB_H

#include <linux/types.h>
#include <linux/ioctl.h>

#define GB10_DMAB_ABI_VERSION		1
#define GB10_DMAB_DEV_NAME		"gb10_dmab"	/* /dev/gb10_dmab */

/* Endpoint types (gb10_dmab_buf_info.type) */
#define GB10_DMAB_BUF_USER		1
#define GB10_DMAB_BUF_DMABUF		2
#define GB10_DMAB_BUF_PCI_BAR		3

/* A PCI function address, used wherever a peer device is named. */
struct gb10_dmab_pci_addr {
	__u16 domain;
	__u8  bus;
	__u8  devfn;		/* PCI_DEVFN(slot, func) == (slot << 3) | func */
};

/* ---- GB10_DMAB_IOC_INFO ------------------------------------------------ */

#define GB10_DMAB_INFO_F_IO_COHERENT	(1u << 0) /* engines assume I/O-coherent peers (GB10 default) */
#define GB10_DMAB_INFO_F_RT		(1u << 1) /* engines run SCHED_FIFO */
#define GB10_DMAB_INFO_F_ISOLATED	(1u << 2) /* every engine CPU is in isolcpus=domain */
#define GB10_DMAB_INFO_F_NOHZ_FULL	(1u << 3) /* every engine CPU is in nohz_full= */

struct gb10_dmab_info {
	__u32 abi_version;
	__u32 nr_engines;
	__u64 engine_cpumask;	/* bit N set => an engine is bound to CPU N (N < 64) */
	__u32 flags;		/* GB10_DMAB_INFO_F_* */
	__u32 poll_us;		/* busy-poll window before an idle engine sleeps */
	__u64 bytes_copied;	/* lifetime totals over all engines */
	__u64 jobs_completed;
	struct gb10_dmab_pci_addr default_peer; /* importer used when import.peer is zero */
	__u32 chunk_bytes;	/* copy granularity / preemption point */
	__u64 reserved[3];
};

/* ---- GB10_DMAB_IOC_REG_USER -------------------------------------------- */

#define GB10_DMAB_REG_F_READONLY	(1u << 0) /* only ever a copy source */

struct gb10_dmab_reg_user {
	__u64 addr;		/* any alignment */
	__u64 len;		/* bytes, > 0 */
	__u32 flags;		/* GB10_DMAB_REG_F_* */
	__u32 handle;		/* out */
};

/* ---- GB10_DMAB_IOC_IMPORT_DMABUF --------------------------------------- */

struct gb10_dmab_import_dmabuf {
	__s32 fd;		/* dma-buf file descriptor */
	__u32 flags;		/* must be 0 */
	/*
	 * PCI function on whose behalf the buffer is attached. The exporter
	 * builds the scatter-list for this device; the NVIDIA exporter requires
	 * a real PCI peer here. All-zero selects the module's default peer
	 * (see GB10_DMAB_IOC_INFO.default_peer). Typically the device you are
	 * bridging the GPU to (ConnectX-7, NVMe, FPGA ...).
	 */
	struct gb10_dmab_pci_addr peer;
	__u32 handle;		/* out */
	__u64 size;		/* out: dma-buf size in bytes */
};

/* ---- GB10_DMAB_IOC_MAP_BAR (CAP_SYS_RAWIO) ----------------------------- */

#define GB10_DMAB_BAR_F_WC		(1u << 0) /* write-combining map (prefetchable BARs only) */

struct gb10_dmab_map_bar {
	struct gb10_dmab_pci_addr dev;
	__u8  bar;		/* 0..5 */
	__u8  pad[3];
	__u64 offset;		/* byte offset into the BAR */
	__u64 len;		/* 0 = to end of BAR */
	__u32 flags;		/* GB10_DMAB_BAR_F_* */
	__u32 handle;		/* out */
	__u64 bar_size;		/* out: full BAR size */
};

/* ---- GB10_DMAB_IOC_BUF_INFO -------------------------------------------- */

struct gb10_dmab_buf_info {
	__u32 handle;		/* in */
	__u32 type;		/* out: GB10_DMAB_BUF_* */
	__u64 size;		/* out */
	__u32 flags;		/* out: registration flags */
	__u32 nr_segments;	/* out: physically contiguous segments (0 for BAR) */
	__u64 reserved[2];
};

/* ---- GB10_DMAB_IOC_COPY ------------------------------------------------ */

#define GB10_DMAB_COPY_F_STRIPE		(1u << 0) /* split across all engines */
#define GB10_DMAB_COPY_F_FLUSH		(1u << 1) /* dma_sync src for CPU / dst for device (non-coherent peers) */
#define GB10_DMAB_COPY_F_WAIT		(1u << 2) /* block until complete; status/elapsed filled in */

struct gb10_dmab_copy {
	__u32 src_handle;
	__u32 dst_handle;
	__u64 src_off;
	__u64 dst_off;
	__u64 len;		/* bytes, > 0 */
	__u32 flags;		/* GB10_DMAB_COPY_F_* */
	__s32 engine;		/* -1 = auto (least loaded), else [0, nr_engines) */
	__u64 cookie;		/* out: completion identifier (never 0) */
	__u32 status;		/* out (F_WAIT): 0 or a positive errno */
	__u32 pad;
	__u64 elapsed_ns;	/* out (F_WAIT): submit -> completion */
};

/* ---- GB10_DMAB_IOC_WAIT ------------------------------------------------ */

struct gb10_dmab_wait {
	__u64 cookie;		/* in: 0 = any completed request; out: the request reaped */
	__s64 timeout_ns;	/* < 0 forever, 0 poll */
	__u32 status;		/* out: 0 or a positive errno */
	__u32 nr_inflight;	/* out: requests on this fd not yet complete */
	__u64 bytes;		/* out */
	__u64 elapsed_ns;	/* out: submit -> completion */
};

#define GB10_DMAB_IOC_MAGIC		0xB1

#define GB10_DMAB_IOC_INFO		_IOR(GB10_DMAB_IOC_MAGIC, 0x00, struct gb10_dmab_info)
#define GB10_DMAB_IOC_REG_USER		_IOWR(GB10_DMAB_IOC_MAGIC, 0x01, struct gb10_dmab_reg_user)
#define GB10_DMAB_IOC_IMPORT_DMABUF	_IOWR(GB10_DMAB_IOC_MAGIC, 0x02, struct gb10_dmab_import_dmabuf)
#define GB10_DMAB_IOC_MAP_BAR		_IOWR(GB10_DMAB_IOC_MAGIC, 0x03, struct gb10_dmab_map_bar)
#define GB10_DMAB_IOC_UNREG		_IOW(GB10_DMAB_IOC_MAGIC, 0x04, __u32)
#define GB10_DMAB_IOC_BUF_INFO		_IOWR(GB10_DMAB_IOC_MAGIC, 0x05, struct gb10_dmab_buf_info)
#define GB10_DMAB_IOC_COPY		_IOWR(GB10_DMAB_IOC_MAGIC, 0x06, struct gb10_dmab_copy)
#define GB10_DMAB_IOC_WAIT		_IOWR(GB10_DMAB_IOC_MAGIC, 0x07, struct gb10_dmab_wait)

#endif /* _UAPI_GB10_DMAB_H */
