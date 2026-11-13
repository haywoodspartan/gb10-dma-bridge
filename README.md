# gb10_dmab — CPU-core DMA bridge for NVIDIA GB10 (DGX Spark)

An out-of-tree Linux kernel module that dedicates a set of ARM cores on the
GB10 Grace Blackwell SoC as software DMA engines, moving data between GPU
memory (via dma-buf), pinned host memory and PCIe device BAR windows. It ships
with CPU-isolation tooling (`isolcpus`/`nohz_full`/`rcu_nocbs`), a Secure Boot
signing flow, and a test/benchmark tool.

Built and verified to compile against `7.0.0-1024-nvidia` (DGX OS, Ubuntu
24.04, gcc 13) on a DGX Spark. It has **not** been loaded on this machine yet:
that needs root and a Machine Owner Key because Secure Boot is enabled (see
*Install*).

---

## 1. What the hardware actually looks like

Measured on this DGX Spark (`lspci`, `nvidia-smi -q`, sysfs):

| Device | Address | Link | Notes |
|---|---|---|---|
| GB10 GPU (10de:2e12) | `000f:01:00.0` | PCIe **2.5 GT/s x1** | BAR0 64 MiB only, no BAR1. `Addressing Mode: ATS`, `FB Memory: N/A` |
| ConnectX-7, 2 ports / 4 PFs | `0000:01:00.{0,1}`, `0002:01:00.{0,1}` | 2 x PCIe Gen5 x4, each shared by both ports: 15.1 GB/s in and 14.0 GB/s out per link, 24.5 GB/s per port, 30.1 GB/s in and 27.7 GB/s out per node (measured, section 14) | One NIC in Socket Direct mode: its x8 is split into two x4 links to the SoC, and *each* link exposes *both* ports (`f0` = port 0, `f1` = port 1). Ports run 200GbE over 2 lanes; `mlx5_core`, 4 RoCE devices |
| NVMe PM9C1a | `0004:01:00.0` | PCIe Gen4 x4 | |
| SMMUv3 | 3 instances | | `iommu.passthrough=0`, translated DMA domains |
| CPU | 20 cores | | Cortex-A725: cpus 0-4, 10-14. Cortex-X925: cpus 5-9, 15-19. Two L3 clusters (0-9, 10-19) |

What that means:

* The GPU is **not** a PCIe data-path device. Its PCIe link is a 2.5 GT/s x1
  management link carrying config space and a 64 MiB register BAR. There is no
  framebuffer and no BAR1 aperture. The GPU reaches the unified 128 GB
  LPDDR5X pool over NVLink-C2C, translating addresses through the SMMU in ATS
  mode, cache-coherent with the CPUs.
* PCIe endpoints hang off root ports on the CPU die and DMA into that same
  pool. A ConnectX-7 writing a receive buffer and the GPU reading it are
  already "bridged" by the memory system. Measured with `tools/rdma_dmabuf_check`
  on this node: CUDA reports `DMA_BUF_SUPPORTED = 0` and
  `GPU_DIRECT_RDMA_SUPPORTED = 0`, `cudaMalloc()` memory cannot be registered
  with the NIC (no CPU page tables, `EFAULT`), but the NIC registers ordinary
  `malloc()` memory and the GPU writes every page of it through ATS. On GB10
  "GPUDirect" is simply: put the buffer in system memory, register it once,
  let both the NIC and the GPU use it.
* A kernel module therefore cannot create a PCIe root bridge, and does not
  need to. What software *can* add is a **DMA engine**: something that moves
  bytes between endpoints without the application cores doing the `memcpy`.
  GB10 has no general-purpose host DMA controller exposed to Linux, so the
  only way to get one is to spend CPU cores on it. That is what this driver
  does: "sacrifice cores to make a DMA bridge", implemented honestly as
  pinned, real-time kernel copy threads with a DMA-engine-style
  submit/complete interface.

Where it is useful on a Spark:

* Staging data between a PCIe device **BAR** (FPGA, driverless accelerator,
  SmartNIC doorbell/ring region) and GPU memory, which no other engine can do.
* Bouncing NIC/NVMe host buffers into `cudaMalloc()` memory without touching
  the application's cores or launching a GPU copy kernel.
* Any large host↔GPU or host↔host copy you want off the compute cores, with
  completion signalled through `poll()`.

---

## 2. Design

```
 user process                                 kernel                               dedicated cores
 ┌───────────────┐   ioctl(REG_USER)     ┌──────────────────────────┐          ┌─────────────────┐
 │ malloc/cuda   │ ───────────────────▶  │ pin_user_pages + vmap    │          │ gb10_dmab/11    │
 │ host buffer   │                       │  → handle A              │          │ (SCHED_FIFO,    │
 │               │   ioctl(IMPORT_DMABUF)│ dma_buf_attach(peer PCI) │  jobs    │  busy-poll then │
 │ cudaMalloc →  │ ───────────────────▶  │ sg pages → vmap          │ ───────▶ │  sleep)         │
 │ dma-buf fd    │                       │  → handle B              │          ├─────────────────┤
 │               │   ioctl(MAP_BAR)      │ pci_iomap_range          │ ───────▶ │ gb10_dmab/12    │
 │ FPGA BAR2     │ ───────────────────▶  │  → handle C              │          ├─────────────────┤
 │               │   ioctl(COPY A→B,     │ split into jobs, queue   │ ───────▶ │ gb10_dmab/13    │
 │               │      STRIPE)          │ per engine               │          ├─────────────────┤
 │               │ ◀─ poll()/WAIT ─────  │ completion list + cookie │ ◀─────── │ gb10_dmab/14    │
 └───────────────┘                       └──────────────────────────┘          └─────────────────┘
```

* **Engines**: one kernel thread per core in `cores=`, created with
  `kthread_run_on_cpu()` and `sched_set_fifo_low()`. Each has its own job
  queue. An idle engine busy-polls for `poll_us` (default 200 µs) so
  submit-to-start latency stays in the microseconds, then sleeps in
  `TASK_IDLE` so it costs nothing when the bridge is quiet.
* **Endpoints** (per-fd handles):
  * `USER`: `pin_user_pages_fast(FOLL_LONGTERM)` + `vmap()`. Charged against
    `RLIMIT_MEMLOCK` like vfio.
  * `DMABUF`: `dma_buf_attach()` on behalf of a *peer PCI function* (the
    NVIDIA exporter requires a real `pci_dev` and skips its topology check on
    coherent platforms), then the exporter's `vmap` if it has one, otherwise a
    `vmap()` of the `struct page`s in the scatter-list. The dma-buf CPU-access
    protocol (`begin/end_cpu_access`) is honoured. Note: with driver
    580.178.04 / CUDA 13.0 the GB10 refuses to export `cudaMalloc()` memory
    as a dma-buf, so on this platform the GPU-visible endpoint is a `USER`
    buffer in system memory (see section 11). The import path still serves
    other exporters (udmabuf, DMA heaps, future driver versions).
  * `PCI_BAR`: `pci_iomap_range()` / `pci_iomap_wc_range()`, `CAP_SYS_RAWIO`.
* **Copies**: `memcpy`, `memcpy_fromio`, `memcpy_toio`, or a bounce for
  MMIO→MMIO. Work is cut into `chunk_kb` pieces with `cond_resched()` between
  them. `STRIPE` splits one request across all engines (≥64 KiB per slice).
  `FLUSH` performs `dma_sync_*` for non-coherent peers (a no-op on GB10, where
  everything is I/O coherent).
* **Completions**: every request gets a cookie; reap with `WAIT` (any or a
  specific cookie, with timeout) or `COPY|WAIT` for synchronous use. `poll()`
  reports `EPOLLIN` when something is reapable. Closing the fd waits for
  in-flight work and releases every handle.
* **Isolation awareness**: at load the module checks the scheduler's
  housekeeping masks and reports whether its cores are in `isolcpus=domain`
  and `nohz_full=`, in `dmesg`, in `/sys/class/misc/gb10_dmab/isolation` and
  in the `INFO` ioctl. `require_isolation=1` refuses to load otherwise.

---

## 3. Which four cores

Measured answer (section 12): **spread the engines over both CPU clusters.**
On GB10 each cluster (cpus 0-9 and cpus 10-19) has its own ceiling on memory
traffic, about 7 to 8 GB/s for cluster 0-9 and 10 to 11 GB/s for cluster 10-19,
and a single core already saturates it. Four engines in one cluster copy at
5.3 GB/s; two per cluster copy at 7 to 9 GB/s. Big (X925) and little (A725)
pairs perform the same, so pick by what you would rather keep for compute:
`cores=8,9,18,19` (X925) or `cores=2,3,12,13` (A725). The isolation set in
`scripts/isolate-cores.sh --apply -c ...` must match. The shipped default is
still `16-19` because that is what the running nodes were set up with;
`scripts/isolate-cores.sh --status` prints the core type of each chosen CPU.

Note (2026-11-12): that advice comes from a boot in which the memory system
was in a slow operating point (section 13). On the current boot the four
engines on `16-19` alone copy 49 GB/s striped, so spreading across clusters
only matters in the slow state.

---

## 4. Build, sign, install

Secure Boot is enabled on this Spark and `CONFIG_MODULE_SIG` is on, so an
unsigned module will be rejected. One-time key setup, then build/sign/install:

```bash
sudo scripts/install.sh --mok-setup     # creates /var/lib/shim-signed/mok/MOK.{priv,der}, enrolls via mokutil
sudo reboot                              # MokManager: Enroll MOK -> Continue -> Yes -> password -> Reboot
sudo scripts/install.sh                  # make, sign, install to /lib/modules/.../extra, modprobe
```

Manual equivalents: `make`, `make sign`, `make install`, `make load`,
`make tools`. For a whole ring use `scripts/ring-mok.sh keys|build|enroll|load`
(section 11). DKMS users: `dkms add .` with the provided `dkms.conf` (DKMS
reuses the same MOK path on Ubuntu).

Files installed: `/etc/modprobe.d/gb10_dmab.conf` (options),
`/etc/modules-load.d/gb10_dmab.conf` (autoload),
`/etc/udev/rules.d/99-gb10-dmab.rules` (`/dev/gb10_dmab` 0660, group `render`).

---

## 5. CPU isolation (`isolcpus`)

Pinning kernel threads to cores does not stop the scheduler from placing other
tasks there. To hand the cores over completely:

```bash
scripts/isolate-cores.sh --status            # what is isolated now vs. the module's cores
sudo scripts/isolate-cores.sh --apply -c 16-19
sudo reboot
scripts/isolate-cores.sh --status            # expect "RESULT: OK"
```

`--apply` writes `/etc/default/grub.d/90-gb10-dmab.cfg` adding

```
isolcpus=domain,managed_irq,16-19 nohz_full=16-19 rcu_nocbs=16-19 irqaffinity=0-15
```

and runs `update-grub`. It also updates `cores=` in the modprobe config so the
two stay in sync. `--remove` undoes it. After reboot the module logs
`engine CPUs are isolated (isolcpus=domain, nohz_full)`.

If you want the module to insist: `options gb10_dmab require_isolation=1`.

---

## 6. Using it

### Tool

```bash
make tools
tools/dmabctl info                        # engines, isolation, default peer, totals
tools/dmabctl bench -s 1024 -n 10 -S      # 1 GiB host->host, striped over all engines, verified
tools/dmabctl bench -s 256 -e 0           # single engine 0
sudo tools/dmabctl udmabuf -s 64          # dma-buf import round trip through /dev/udmabuf
tools/dmabctl cuda -s 512 -n 10           # host <-> GPU-visible memory through the bridge, verified by the GPU
tools/rdma_dmabuf_check roceP2p1s0f0 256  # which GPU<->NIC registration paths work on this node
sudo tools/dmabctl bar -d 0004:01:00.0 -b 0 -l 64   # dump a BAR window (read side effects are yours)
```

`bench` and `cuda` print the kernel-measured GB/s per request plus a baseline
(`memcpy` on the caller's core, `cuMemcpyHtoD` from pinned memory). Large
`REG_USER` needs `ulimit -l unlimited` or `CAP_IPC_LOCK`.

### C API (see `include/uapi/gb10_dmab.h`)

```c
int fd = open("/dev/gb10_dmab", O_RDWR);

/* host buffer */
struct gb10_dmab_reg_user ru = { .addr = (uintptr_t)host, .len = len };
ioctl(fd, GB10_DMAB_IOC_REG_USER, &ru);

/* cudaMalloc() memory as a dma-buf */
int dmabuf_fd;
cuMemGetHandleForAddressRange(&dmabuf_fd, dptr, len, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
struct gb10_dmab_import_dmabuf im = { .fd = dmabuf_fd };          /* .peer = {0}: default peer */
ioctl(fd, GB10_DMAB_IOC_IMPORT_DMABUF, &im);

/* FPGA BAR2 (root) */
struct gb10_dmab_map_bar mb = { .dev = { .domain = 0x0007, .bus = 1, .devfn = 0 }, .bar = 2 };
ioctl(fd, GB10_DMAB_IOC_MAP_BAR, &mb);

/* async copy host -> GPU across all engines */
struct gb10_dmab_copy c = { .src_handle = ru.handle, .dst_handle = im.handle,
                            .len = len, .flags = GB10_DMAB_COPY_F_STRIPE, .engine = -1 };
ioctl(fd, GB10_DMAB_IOC_COPY, &c);                                  /* c.cookie */

struct pollfd pfd = { .fd = fd, .events = POLLIN };
poll(&pfd, 1, -1);
struct gb10_dmab_wait w = { .cookie = c.cookie, .timeout_ns = -1 };
ioctl(fd, GB10_DMAB_IOC_WAIT, &w);                                  /* w.status, w.elapsed_ns */
```

The `peer` in `IMPORT_DMABUF` is the PCI function the buffer is attached for.
Leave it zero for the module default (the GPU itself), or name the device you
are bridging to (for example ConnectX-7 `0000:01:00.0`) so the scatter-list is
also valid for that device's own DMA.

---

## 7. Module parameters

| Parameter | Default | Meaning |
|---|---|---|
| `cores` | `16-19` | cpulist of engine CPUs |
| `poll_us` | `200` | busy-poll window before an idle engine sleeps; `0` = never spin |
| `rt` | `Y` | `SCHED_FIFO` engines (priority 1) |
| `io_coherent` | `Y` | peers are I/O coherent; `FLUSH` is a no-op. Correct for GB10 |
| `require_isolation` | `N` | fail to load unless every engine CPU is in `isolcpus=domain` |
| `try_vmap` | `Y` | prefer the exporter's `vmap` for dma-bufs |
| `chunk_kb` | `2048` | bytes copied between preemption points |
| `peer_bdf` | *(first NVIDIA display-class function)* | default dma-buf importer |

sysfs: `/sys/class/misc/gb10_dmab/{engines,cores,isolation,default_peer}`.
`engines` gives per-engine `jobs bytes busy_ns queued sleeps`.

---

## 8. Limitations and safety

* **Software engine**: throughput is bounded by per-core `memcpy` bandwidth
  (expect tens of GB/s aggregate for four A725s against 273 GB/s LPDDR5X).
  That is well above any PCIe endpoint on the Spark, but a GPU copy kernel
  will still beat it for pure GPU↔GPU-memory moves.
* **BAR mapping** does not take ownership of the PCI region; it can alias a
  window owned by another driver. Only use it on devices you control, and
  remember that reads of device registers can have side effects.
* **dma-buf exporters** without `vmap` and without `struct page`s (PFN-only,
  e.g. BAR1 "FORCE_PCIE" mappings) are rejected with `EOPNOTSUPP`. On this
  GB10 the CUDA driver does not export `cudaMalloc()` memory at all
  (`DMA_BUF_SUPPORTED = 0`), so GPU-visible buffers for the bridge are
  system memory registered as `USER` endpoints; the GPU reaches them via ATS.
* **CPU hotplug**: do not offline engine CPUs while the module is loaded.
* **RT throttling**: engines run at `SCHED_FIFO` priority 1 and sleep when
  idle, so the default `sched_rt_runtime_us` (95%) is not a problem; a
  multi-second single copy on a non-isolated core may show throttling
  messages. Isolating the cores makes the question moot.
* Signals interrupting `COPY|WAIT` return `EINTR` with the cookie filled in;
  the request is still in flight and must be reaped with `WAIT`.

---

## 9. Files

```
gb10_dmab_main.c        module init, params, ioctls, sysfs, isolation check
gb10_dmab_buf.c         endpoints: USER / DMABUF / PCI_BAR
gb10_dmab_engine.c      per-core kthreads, queues, striping, completions
gb10_dmab.h             internal structures
include/uapi/gb10_dmab.h   user ABI
tools/dmabctl.c         info / bench / udmabuf / cuda / bar
tools/rdma_dmabuf_check.c  GPU<->NIC registration paths (dma-buf, direct, ATS system memory)
tools/ringbw.c          RDMA ring bandwidth test, GPU fill on the sender, GPU verify on the receiver
tools/ringgather.c      the head gathers GPU-produced slices from the other nodes over every NIC path at once
tools/membw.c           pinned-thread read/write/copy bandwidth per core, cluster or core set
tools/gpuhostbw.cu      GPU SM read/write/copy bandwidth on host memory (mmap/THP/4K, cudaHostAlloc, managed, cudaMalloc)
tools/memlat.c          dependent-load memory latency from one CPU (fingerprint of the memory operating point)
scripts/ring-test.sh    discovers the cables over SSH and runs ringbw on every cable of the ring
scripts/ring-gather.sh  run on the head: starts ringgather workers on the other nodes and gathers from all of them
scripts/ring-mtu.sh     MTU on every ring port of every node (passwordless sudo variant)
scripts/node-admin.sh   per-node root worker: MTU live + netplan drop-in, isolcpus staging, status
scripts/ring-admin.sh   runs node-admin.sh on every node (interactive sudo or passwordless)
scripts/cluster-bootstrap.sh  passwordless sudo + full SSH mesh + project sync across the nodes
scripts/ring-mok.sh     one MOK for the ring: create/copy, build+sign+install per node, enroll, load
scripts/isolate-cores.sh   isolcpus/nohz_full/rcu_nocbs/irqaffinity via GRUB drop-in
scripts/install.sh      MOK key setup, build, sign, install, load
etc/                    modprobe, modules-load, udev
Makefile, Kbuild, dkms.conf
```

## 10. Verification checklist after install

```bash
dmesg | grep gb10_dmab                       # "4 copy engine(s) on CPUs 16-19 ...", isolation line
ps -eo pid,psr,cls,rtprio,comm | grep gb10   # one FIFO thread per core, psr = its CPU
tools/dmabctl bench -s 512 -n 5 -S           # verify OK, GB/s
tools/dmabctl cuda -s 512 -n 5               # verify via cuMemcpyDtoH OK
scripts/isolate-cores.sh --status            # RESULT: OK once isolcpus is active
```

---

## 11. Three-node ring over ConnectX-7

Topology as configured by NVIDIA Sync on 2026-11-09 and read back from all
three nodes over SSH, corrected on 2026-11-12 after checking the hardware
(section 13). Each Spark has two QSFP ports, each running 200GbE over two
lanes (`mstlink`: `200G 2x`, RS-FEC; the DACs are 400G-capable QSFP112 parts
but the port only enables up to 200G). The ConnectX-7 is wired **Socket
Direct**: its PCIe x8 is split into two Gen5 x4 links to the SoC (domains
`0000` and `0002`), and each link exposes *both* ports as a PF. So every
physical port appears twice: `enp1s0f0np0` and `enP2p1s0f0np0` are the same
port 0 reached over different PCIe links (same cable serial in `ethtool -m`,
identical `rx/tx_bytes_phy` counters), and `enp1s0f1np1` / `enP2p1s0f1np1`
are port 1. Every pair of nodes is therefore joined by **one** cable that
carries two /24 subnets, one per PCIe link, and every node has four ring
netdevs on two cables:

| Link (/24) | Node A side | Node B side |
|---|---|---|
| 10.100.32 | DGX-1 `enp1s0f0np0` .2 | DGX-2 `enp1s0f1np1` .1 |
| 10.100.33 | DGX-1 `enP2p1s0f0np0` .2 | DGX-2 `enP2p1s0f1np1` .1 |
| 10.100.34 | DGX-2 `enp1s0f0np0` .1 | DGX-3 `enp1s0f1np1` .2 |
| 10.100.35 | DGX-2 `enP2p1s0f0np0` .1 | DGX-3 `enP2p1s0f1np1` .2 |
| 10.100.36 | DGX-3 `enp1s0f0np0` .2 | DGX-1 `enp1s0f1np1` .1 |
| 10.100.37 | DGX-3 `enP2p1s0f0np0` .2 | DGX-1 `enP2p1s0f1np1` .1 |

Rows 10.100.32/.33, .34/.35 and .36/.37 are the three physical cables. All
four netdevs per node report 200 Gb/s RoCE v2, ACTIVE, sub-millisecond ping,
but the two netdevs on one port share its 200 Gb/s. Each node's `enp1s0*`
netdevs share PCIe link `0000:01:00.x` and its `enP2p1s0*` netdevs share
`0002:01:00.x`, both Gen5 x4, so each PCIe link carries one half of the
traffic in and one half out of the ring direction. `rdma-core 50`, `perftest 24.01` (no CUDA),
no NCCL, `nvidia-peermem` unusable on this kernel. All twelve ring netdevs
run MTU 9000 (RoCE `active_mtu` 4096), set live and persisted on every node
by `scripts/ring-admin.sh --mtu 9000` in `/etc/netplan/99-zz-spark-ring-mtu.yaml`,
a drop-in that sorts after NVIDIA Sync's file and overrides only `mtu:`.

Three nodes in a ring is a full mesh: nobody forwards for anybody.

### How GPU data should move between nodes

`tools/rdma_dmabuf_check` result on this node:

```
dma-buf GPUDirect=no   direct cudaMalloc MR=no   unified (NIC MR + GPU via ATS)=yes
```

So the ring data path is:

1. Allocate communication buffers in **system memory** (`malloc`, `mmap`,
   hugetlbfs, or `cudaHostAlloc`), not `cudaMalloc`.
2. Register them once with the NIC (`ibv_reg_mr`), exchange rkeys, and RDMA
   WRITE/READ between nodes. The NIC DMAs straight into those pages.
3. The GPU reads and writes the same pages directly (`pageable memory
   access = 1`, host page tables via ATS). No copy, no staging, no engines.
4. Data that already lives in `cudaMalloc()` memory is moved into such a
   buffer by the GPU's own copy engine (`cudaMemcpyAsync`, measured with
   `dmabctl cuda` as the baseline), or kept in system memory from the start.

Where `gb10_dmab` fits in the ring: moving received data between NIC ring
buffers and other host/BAR endpoints without touching compute cores, and
BAR-to-memory staging for devices that have no DMA of their own. It is not
needed for NIC-to-GPU traffic, because on GB10 that traffic has no second hop.

### Bring-up commands

```bash
# raise RoCE MTU to 4096 on both ring links (both ends), persist in netplan
sudo ip link set enp1s0f0np0 mtu 9000
sudo ip link set enP2p1s0f0np0 mtu 9000

# bandwidth of the real path (RDMA into system memory the GPU will read):
#   on the neighbour:   ib_write_bw -d roceP2p1s0f0 -x 3 -a --report_gbits
#   on this node:       ib_write_bw -d roceP2p1s0f0 -x 3 -a --report_gbits <neighbour-ip>
# -x 3 selects the RoCE v2 IPv4 GID; check with: show_gids

# GPU<->NIC registration paths on each node
tools/rdma_dmabuf_check roceP2p1s0f0 256
```

### `tools/ringbw`: the ring test the GPU feeds and consumes

`ringbw` is one process per cable direction. `--listen` receives on one ring
link, `--connect` sends on another, so with both options a process is one
segment of the ring. `scripts/ring-test.sh DGX-1 DGX-2 DGX-3` discovers the
cables over SSH, works out the ring order, copies the binary to the remote
nodes, and runs one process per cable on every node at once (two per node
here); its retrying connect tolerates receivers that come up a little later. The sender's GPU fills a
NIC-registered system buffer (kernel built at runtime with NVRTC), the NIC
RDMA-WRITEs it to the neighbour in 1 MiB chunks with 32 in flight, the last
chunk carries the iteration number as an immediate, and the receiver's GPU
verifies every 64-bit word directly from the landed pages before acking.
`-C` does fill/verify on the CPU instead. Loopback on one node (`--listen`
and `--connect` to the same local address) checks the whole protocol without
touching the neighbours.

```bash
# the whole ring, every cable, all nodes at once (SSH to the others must work)
scripts/ring-test.sh -s 1024 -n 10 DGX-1 DGX-2 DGX-3
scripts/ring-test.sh --mtu 9000 DGX-1 DGX-2 DGX-3     # same, after setting jumbo MTU everywhere

# one cable by hand: receiver on DGX-3, sender on DGX-1 (cable 10.100.36)
tools/ringbw --listen 10.100.36.2                     # on DGX-3
tools/ringbw --connect 10.100.36.2 -s 1024 -n 20      # on DGX-1
```

Result of `scripts/ring-test.sh -s 512 -n 5 DGX-1 DGX-2 DGX-3` on 2026-11-09,
all six cables carrying traffic simultaneously, ring order DGX-1 -> DGX-3 -> DGX-2:

| Cable direction | Wire | Receiver GPU verify (512 MiB) |
|---|---|---|
| DGX-1 -> DGX-3, both cables | 13.61 / 13.61 GB/s | 3.16 / 3.23 ms |
| DGX-3 -> DGX-2, both cables | 13.56 / 13.62 GB/s | 3.11 / 3.12 ms |
| DGX-2 -> DGX-1, both cables | 13.48 / 13.43 GB/s | 3.63 / 3.47 ms |

Every iteration verified on the receiving GPU. Summed, that is about 81 GB/s
of GPU-produced, GPU-consumed data, 27 GB/s out and 27 GB/s in per node, with
each PCIe Gen5 x4 link at its practical limit in both directions.

**Correction (2026-11-12):** the two rows of each pair are the two PCIe halves
of *one* 200 Gb/s cable, and `ring-test.sh` starts cable 0 and cable 1 as
separate waves two seconds apart while each run lasts well under a second, so
the two halves of a port were not on the wire at the same time. 13.6 GB/s is
what one PCIe x4 half does alone; two halves driving the same port together
are capped by the 200G line rate (about 24 GB/s of RoCE payload), not 27.2.
Measured the same day with `ib_write_bw` (section 13): each half alone
111.8 Gb/s (14.0 GB/s), both halves of the port at once 98.0 Gb/s each,
196 Gb/s (24.5 GB/s) in total. A cable carries 24.5 GB/s per direction. A
node is not worth two cables, though: both ports share the same two PCIe
links, which together take 30.1 GB/s in and give 27.7 GB/s out (measured
2026-11-13, section 14). The 81 GB/s above is a sum of halves that never
overlapped; in ring order, where every node sends on one cable and receives
on the other, the three cables would carry about 73 GB/s (not measured).

Repeated after MTU 9000 (RoCE 4096) with `-s 1024 -n 10`: 13.37 to 13.44 GB/s
average per cable, best 13.69 GB/s, receiver GPU verify 6.0 to 7.8 ms per
1 GiB, all 60 iterations OK. The link is PCIe-bound, so the larger frames
change little; they mainly cut per-packet overhead on the NIC.

### Cluster administration scripts

| Script | What it does |
|---|---|
| `scripts/cluster-bootstrap.sh NODE...` | one-time: passwordless sudo for your user on every node (validated sudoers drop-in), an ed25519 key per node, every key in every `authorized_keys`, a managed `# >>> spark-ring` block in `~/.ssh/config` with direct-cable addresses, `known_hosts`, optional `--sync-project`, then a full ssh+sudo verification matrix. Asks for each node's password once. |
| `scripts/ring-admin.sh [--mtu N] [--isolcpus LIST] [--status] NODE...` | runs `node-admin.sh` as root on every node: live + persistent MTU, and/or `isolate-cores.sh --apply` to stage `isolcpus`/`nohz_full`/`rcu_nocbs`/`irqaffinity` in GRUB. Never reboots. |
| `scripts/node-admin.sh` | the per-node root worker behind `ring-admin.sh`; also usable directly with `sudo`. |
| `scripts/ring-mtu.sh` | non-interactive MTU variant for nodes that already have passwordless sudo. |
| `scripts/ring-test.sh` | the GPU-fed RDMA test on every cable (above). |
| `scripts/ring-gather.sh WORKER...` | run on the head: the other nodes' GPUs produce, all NIC paths carry, the head's GPU verifies (section 14). |
| `scripts/ring-mok.sh keys\|build\|enroll\|load\|status NODE...` | Secure Boot across the ring: one Machine Owner Key created on the first node and copied to the others, `make clean modules tools` + `sign` + `install` on every node against its own kernel, the interactive `mokutil --import` (one-time password per node, then reboot and confirm in MokManager), and `modprobe` afterwards. |

State on 2026-11-10 after running them: every node reaches every other with
key-based ssh and passwordless sudo; MTU 9000 everywhere; `isolcpus=domain,
managed_irq,16-19 nohz_full=16-19 rcu_nocbs=16-19 irqaffinity=0-15` active on
all three nodes; one MOK enrolled on all three; `gb10_dmab` signed per kernel,
installed and autoloaded on all three, each reporting four SCHED_FIFO engines
on the isolated cpus 16-19 and the GB10 as default dma-buf peer.

### First measurements of the engines (DGX-1 and DGX-2, `dmabctl` as root)

| Test | Result |
|---|---|
| pin 2 x 512 MiB | 44 to 52 ms (20 to 25 GB/s pin rate) |
| host -> host, one engine | 4.5 to 5.1 GB/s, verified |
| host -> host, striped over 4 engines | 5.0 to 5.6 GB/s, verified |
| host -> GPU-visible memory and back, GPU-verified | 5.1 to 5.6 GB/s each way, verified |
| caller core `memcpy` / GPU `cuMemcpy` on host memory | 3.8 to 4.9 GB/s |

Functionally everything checks out. The bandwidth is far below what the
GPU gets from the same pages (over 100 GB/s with `ringbw`), and striping over
four engines in one cluster barely beats one engine: the per-engine rate
collapses when all four run. The working hypothesis is a per-cluster ceiling
of roughly 10 to 12 GB/s of memory traffic on the CPU die's path to the
LPDDR5X (cpus 10-19 share one DSU), which a single core already saturates.
Next step: measure per-cluster CPU memory bandwidth and, if confirmed, spread
the engines across both clusters (for example `cores=8,9,18,19`, with the
isolation set changed to match).

Each sender iteration prints wire GB/s; each receiver iteration prints the
GPU verify time, which is also the GPU's read bandwidth from NIC-written
memory. Measured in loopback on this node (512 MiB x 5, GPU mode): 13.25 GB/s
(106 Gb/s) through the CX7, GPU fill 5 ms, GPU verify 3.3 ms = about 160 GB/s
read, every iteration OK. Across a real link expect the same order: the PCIe
Gen5 x4 endpoint, not the 200 Gb/s port, is the limit.

For collectives, NCCL is not installed here; when you add it, expect it to
use host (system) buffers (`NCCL_NET_GDR_LEVEL` will not apply, there is no
GDR on GB10) and both ring links (`NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0`).

---

## 12. CPU memory bandwidth on GB10 and engine placement

**Read section 13 first:** every number in this section was taken on the
boot of 2026-11-10. On the next boot (2026-11-11, same kernel, cmdline,
governor and binaries) the same `membw` read 7 to 8 times faster (78 GB/s
on cpus 16-19, 153 GB/s on all twenty cores) and the striped engine copy
went from 5.3 to 49 GB/s. The per-cluster "ceiling" below is an operating
point of the memory system, not a fabric limit, and nothing in Linux shows
which one a boot is in; `tools/memlat` and one `membw` line tell you.

The engines turned out to be bandwidth-bound by the CPU complex, not by the
driver. `tools/membw` pins one thread per CPU and measures streaming read,
write and `memcpy` on private THP-backed buffers (DGX-1, 2026-11-10, best of
5, GB/s; copy counts bytes copied):

| Threads on CPUs | Read | Write | Copy |
|---|---|---|---|
| 16 (one X925, cluster 10-19) | 8.2 | 6.0 | 4.95 |
| 5 (one X925, cluster 0-9) | 7.2 | 4.5 | 3.7 |
| 0 (one A725) | 4.7 | 2.6 | 4.4 |
| 16-19 (four X925, one cluster) | 10.3 | 8.5 | 5.6 |
| 5-8 (four X925, other cluster) | 7.7 | 4.0 | 3.8 |
| 6,7,16,17 (two per cluster, X925) | 16.7 | 11.8 | 7.7 to 8.5 |
| 2,3,12,13 (two per cluster, A725) | 16.7 | 10.6 | 10.3 to 10.6 |
| 5-9,15-19 (all ten X925) | 19.8 | 14.3 | 10.6 |
| 0-19 (all twenty cores) | 25.6 | 20.2 | 13.4 |
| 16 with 4 KiB pages instead of THP | 8.0 | 4.9 | 4.3 |

Three facts follow. Each cluster's path to the LPDDR5X carries roughly 10 GB/s
of traffic (cluster 0-9 somewhat less), and one core can saturate it. The
whole CPU complex tops out near 27 GB/s of reads or 13 GB/s of copy, while
GPU kernels read and write the same pages at 100 to 160 GB/s (`ringbw`) and
each ConnectX-7 cable DMAs at 24.5 GB/s (13.6 GB/s per PCIe half). Page
size does not matter.

The engines track the same curve (`dmabctl bench -s 1024 -n 6 -S` after
`modprobe gb10_dmab cores=...`, kernel-measured, per-engine rate from
`/sys/class/misc/gb10_dmab/engines`):

| `cores=` | Striped copy | Per engine while busy |
|---|---|---|
| 16-19 | 5.3 GB/s | 1.3 to 1.6 |
| 7,8,17,18 | 7.2 GB/s | 1.8 (cluster 0-9), 3.1 (cluster 10-19) |
| 2,3,12,13 | 7.3 to 8.7 GB/s | 1.9 to 2.2 and 2.4 to 2.6 |
| 1,2,11,12 | 6.9 GB/s | 1.7 and 2.5 |
| 1-4,11-14 (eight A725) | 7.0 GB/s | |

What this means for a CPU-core DMA bridge on GB10: four cores can move about
7 to 9 GB/s if split across the clusters, and no number of cores moves more
than about 13 GB/s. That is enough to feed or drain one 200 GbE cable, and it
is the right tool for BAR/MMIO staging and for keeping copies off application
cores, but bulk host-memory movement is an order of magnitude faster on the
GPU's SMs, and NIC traffic should land directly in GPU-visible memory as
section 11 describes. One more observation from `dmabctl cuda`: the GPU's
copy engines (`cuMemcpy` between host buffers) run at only 4 to 5 GB/s through
ATS, so "use the GPU" means a copy kernel, not `cudaMemcpy`.

---

## 13. The SoC block diagram against what Linux exposes (2026-11-12)

GB10 is two dies joined by NVLink-C2C. The CPU die (a MediaTek MT8901-class
design, which is why `mt8901-pinctrl`, `DRAM8901` and `MTKW*` ACPI IDs show
up) carries the HSIO block (PCIe root ports, USB), two CPU clusters, the
fabric, and two LPDDR5X memory subsystems with their PHYs. The GPU die
carries the SMs, the video engines and the other end of C2C. Reading the
ACPI tables of this node (`DSDT`, `IORT`, `PPTT`, decompiled with `iasl`)
together with sysfs gives this map:

| Block in the diagram | What Linux sees | Driver |
|---|---|---|
| HSIO | Root ports `10de:22ce` (Gen5 x4: CX7 link A, CX7 link B, NVMe), `10de:22d0` (Gen4 x1: 10GbE, Wi-Fi), `10de:22d1` (x16-capable, carries the GPU's 2.5 GT/s x1 management link); xHCI `NVDA8000/8001` x6; USB4 hosts `NVDA8100` x4 | `pcieport`, `xhci-hcd`; nothing binds USB4 |
| CPU clusters | `CLU0`/`CLU1` in PPTT = cpus 0-9 and 10-19, each 5 x A725 (512 KiB L2) + 5 x X925 (2 MiB L2), 8 MiB L3 per cluster; PMUs `armv8_pmuv3_0/1`, `arm_spe_0` (SPE works); CPPC through `cppc_cpufreq`, governor `performance`, 2.8 GHz (A725) / 3.9 GHz (X925) | yes |
| Fabric | no ACPI device, no uncore PMU (no `arm_cspmu`, CMN or DSU PMU node) | none |
| Memory subsystems | `DRAM8901` (`\_SB_.DRAM`: four 8 KiB register windows at 0x10200000/0x10206000/0x10208000/0x1C004000); `SSPM` (`MTKW8400`, MediaTek system power manager, 320 KiB at 0x1C8B0000, hidden when the OS claims to be Windows); `MTEL` (`NVDA8800`, read-only telemetry SRAM of the CPU DVFS microcontroller "CPUEB" at 0x05170000, with the CPPC registers, a perf-counter log and a "fastdvfs" log inside it, offsets published by its `_DSM`) | none, none, CPPC only. No `devfreq`, no NUMA/HMAT: one node, no bandwidth control |
| C2C | `nvidia-smi c2c -s`: 8 links x 43.125 GB/s = 345 GB/s per direction; GPU `Addressing Mode: ATS` with ATS and PASID enabled in its PCIe config, alone on SMMU 0x13000000 in a strict `DMA` domain; `/proc/driver/nvidia` says `DMA Size: 41 bits` | `nvidia` |
| GPU cores, video | 48 SMs, `sm_121`, graphics 2.4 GHz (max 3.0), video 2.1 GHz; HDA controller `NVDA2014` sits on the GPU's SMMU | `nvidia`, `snd_hda_acpi` |
| Peripheral DMA | `DMA0` (`NVDA2861`): 4 KiB of registers at 0x14200000, `_CCA = 0` (non-coherent), no interrupt, behind the third SMMU (0x14900000) with the three SPI controllers `NVDA0210` | **none** (no in-tree driver matches `NVDA2861`; SPI has none either) |
| NIC (second diagram) | Socket Direct: GB10 PCIe ports 6:9 and 10:13 feed `PEX1[3:0]` and `PEX1[7:4]` of the CX7; both halves see both QSFP ports (section 11). Firmware 28.50.1002, PSID `NVD0000000087`, `ATS_ENABLED = False`, no `sd_group` reported, so `mlx5_core`'s multi-PF netdev (Socket Direct merging, present in this kernel) stays off | `mlx5_core`, four PFs in `DMA-FQ` SMMU domains |

What this means for "experimental stuff":

* `DMA0` is the only hardware DMA engine the firmware admits to. Four KiB of
  registers, non-coherent, no IRQ, next to the SPI controllers: most likely
  MediaTek's CQDMA/HSDMA memory-copy block or the SPI DMA. Identifying it
  means reading its registers from a signed kernel module (lockdown blocks
  `/dev/mem`), and only after confirming its clock is on, because touching a
  gated MediaTek peripheral stalls the bus. Even if it works it will not beat
  the GPU numbers below. Low priority, noted for completeness.
* The fabric and memory DVFS are run by the SSPM/CPUEB firmware and Linux
  has no interface to them. The measurements below show that this is the
  single biggest factor in CPU-side copy bandwidth.
* The SMMU performance monitors (`smmuv3_pmcg_*`) work with `perf` and the
  GPU's ATS traffic lands on `smmuv3_pmcg_13002` alone.
* Socket Direct merging in `mlx5_core` needs firmware support this board
  does not advertise; `mstconfig` has no parameter for it.

### GPU SMs as the DMA engine (`tools/gpuhostbw`)

`gpuhostbw` hands a plain host pointer to GPU kernels (pageable memory
access through ATS, `usesHostPageTables = 1`) and measures streaming read,
write and copy with 192 blocks x 512 threads. 1 GiB buffers, best of 10,
this node, GB/s:

| Backing | Read | Write | Copy (bytes copied) |
|---|---|---|---|
| `mmap` + `MADV_HUGEPAGE` (THP, verified) | 162 | 113 | 77 |
| `mmap` + `MADV_NOHUGEPAGE` (4 KiB pages) | 169 | 109 | 77 |
| `cudaHostAlloc` (pinned) | 191 | 190 | 106 |
| `cudaMalloc` (reference, same LPDDR5X over C2C) | 249 | 197 | 109 |

So the GPU copies host memory at 77 to 106 GB/s, six to eight times what the
whole CPU complex manages in section 12 and ten times four `gb10_dmab`
engines. For host-to-host bulk moves the right "DMA engine" on GB10 is a
copy kernel on the SMs, and the right buffer for it is `cudaHostAlloc`
memory (which the NIC can also register, since it has CPU page tables).
Pageable memory loses about 15 % on reads and 40 % on writes against pinned;
the counters below show why it is not the translation.

### What the GPU's ATS path costs (SMMU PMCG counters)

`sudo perf stat -a -e smmuv3_pmcg_13002/{transaction,tlb_miss,trans_table_walk_access,pcie_ats_trans_rq}/`
around `gpuhostbw -m read -s 1024 -n 20` (21 launches = 22.5 GB read):

| Backing | ATS translation requests | SMMU TLB misses | Page-table walks | Read GB/s |
|---|---|---|---|---|
| THP | 1.72 M (13 KB per request) | 0.31 M | 0.32 M | 168 |
| 4 KiB pages | 9.87 M (2.3 KB per request) | 6.18 M | 6.38 M | 165 |
| `cudaHostAlloc` | 9.41 M | 6.24 M | 6.86 M | 194 |

Every GPU access to system memory is an ATS request; with 4 KiB pages the
SMMU walks page tables 6.4 million times for 22 GB and still sustains 165
GB/s, so page size does not matter for GPU reads of host memory, just as it
did not for the CPU. Pinned memory is translated at 4 KiB granularity too
and is nevertheless the fastest, so its advantage comes from how the driver
maps it (memory attributes, no coherence traffic with the CPU caches), not
from fewer translations.

### CPU memory bandwidth depends on a DVFS state the CPU does not control

The same `tools/membw` as section 12, same node, same binary, run right
after `gpuhostbw` had been streaming memory (GB/s, read / write / copy):

| CPUs | Section 12 (2026-11-10) | 2026-11-12, GPU just active | with GPU copying 83 GB/s | with GPU copying 189 GB/s |
|---|---|---|---|---|
| 16-19 (four X925, one cluster) | 10.3 / 8.5 / 5.6 | 77.7 / 68.4 / 49.2 | 64.3 / 67.1 / 41.5 | 24.9 / 66.5 / 13.8 |
| 2,3,12,13 (two A725 per cluster) | 16.7 / 10.6 / 10.3 | 124.9 / 89.4 / 70.3 | 96.3 / 89.5 / 55.1 | 40.4 / 88.4 / 24.3 |
| 0-19 (all twenty) | 25.6 / 20.2 / 13.4 | 153.1 / 130.8 / 89.5 | 126.9 / 140.3 / 75.0 | 56.7 / 123.3 / 33.6 |

Seven to eight times the section 12 numbers, from nothing but the state of
the memory system. One cluster of four X925 reads 78 GB/s, the whole CPU
complex 153 GB/s read and 131 GB/s write, and the GPU's traffic shares the
same 273 GB/s LPDDR5X with it (under the 189 GB/s GPU load, CPU reads fall
to the remainder while CPU writes are hardly touched). The engines follow:
`dmabctl bench -s 1024 -n 6 -S` with the stock `cores=16-19` copies at
**48.9 GB/s average, 49.5 best, verified**, against 5.3 GB/s on 2026-11-10,
the caller-core `memcpy` baseline went from 4 to 23.7 GB/s and the pin rate
from 22 to 68 GB/s.

What the state is and what sets it is not yet known. The slow numbers were
taken on the boot of 2026-11-10 01:12, the fast ones on the boot of
2026-11-11 03:42; kernel, cmdline, governor (`performance`, set by
`nv-cpu-governor.service`), THP and binaries are the same. It is not a
short-lived effect of GPU traffic: sampled every 30 s for six minutes after
the GPU went idle (P0, 2.4 GHz, 0 to 5 % utilisation, 10 W), cpus 16-19
stayed at 80 / 70 / 51 GB/s. The GPU reports no thermal or power slowdown.
Linux exposes no devfreq, no HMAT and no memory-controller driver, and the
CPUEB telemetry SRAM that would show the DSU/fabric/DRAM operating points is
behind lockdown (experiment 4). Fingerprint of the fast state, for telling
boots apart: `tools/memlat 16` 115 ns per dependent load (128 ns with 4 KiB pages),
`tools/memlat 0` 166 ns, and one core alone reads 33 GB/s (X925, cpu 16) or
39 GB/s (A725, cpu 0) in `membw`, against 8.2 and 4.7 GB/s in section 12. A
boot that shows 8 GB/s from one core is in the slow state; expect its
latency to be several hundred ns. The boot of 2026-11-13 was in the fast
state again: 116 ns on cpu 16, 76 / 63 / 48 GB/s on cpus 16-19 and
157 / 133 / 90 GB/s on all twenty. `memlat 0` read 240 ns on that boot with a
desktop session and an idle `llama-server` on the housekeeping cores, so use
an isolated core (16) as the reference. Two fast boots and one slow one so
far.

### Experiments: done, and proposed

Done on 2026-11-12 (above): GPU SM bandwidth on host memory by backing type,
SMMU PMCG counting of the GPU's ATS traffic, CPU bandwidth versus memory
DVFS state and versus GPU load, and the Socket Direct topology check
(`ethtool -m` serials, `*_phy` counters, `mstlink`).

Also done the same day: the shared-port ceiling, measured for real.
`ib_write_bw -s 1048576 -q 4 -D 10 -x 3 --report_gbits` from DGX-1's
`rocep1s0f0` (PCIe domain `0000`) and `roceP2p1s0f0` (domain `0002`) to
DGX-2's `rocep1s0f1` and `roceP2p1s0f1`, the two halves of cable
10.100.32/.33, servers started on DGX-2 over ssh:

| Run | Half A (`0000`) | Half B (`0002`) | Total |
|---|---|---|---|
| A alone | 111.8 Gb/s (14.0 GB/s) | | 14.0 GB/s |
| B alone | | 111.8 Gb/s (14.0 GB/s) | 14.0 GB/s |
| A and B at once | 98.0 Gb/s (12.3 GB/s) | 98.0 Gb/s (12.3 GB/s) | 196.1 Gb/s (24.5 GB/s) |

Alone, each half is PCIe-bound at the same 111.8 Gb/s to the fourth digit.
Together they share the 200G port evenly and the sum is its line rate:
`tx_bytes_phy` of `enp1s0f0np0` and `enP2p1s0f0np0` advanced by the same
250 GB during the run, 25.0 GB/s on the wire with headers. So a cable is
worth 24.5 GB/s per direction, and the per-half 13.6 GB/s figures in section
11 become 12.3 GB/s whenever both halves of a port are busy. A node is not
worth two cables: section 14 drives both ports at once and finds that the
two PCIe links, not the ports, set the node's limit at 30.1 GB/s in and
27.7 GB/s out. (An earlier version of this paragraph said 49 GB/s per node;
that was inferred from one cable and is wrong.) The engines' 49 GB/s striped
copy on this boot is more than the NIC can deliver or take; the 5.3 GB/s of
the slow state is a fifth of the NIC's limit.

Proposed, in order of expected payoff:

1. **Pin down the memory operating point.** Run `tools/memlat 16` and
   `tools/membw -s 256 -n 3 16-19` right after boot and again after the GPU
   has been used, across several boots, and log them next to the
   `nvidia-smi` pstate. If the slow state comes back, test what lifts it: a
   GPU copy loop (`gpuhostbw -L 30 -s 512 -P -b 8`), `nvidia-persistenced`
   off and on, a display attached or not, and whether the engines' own
   traffic ever lifts it. Until this is understood, measure before trusting
   any engine number, and read sections 3 and 12 as "the slow state".
2. **400G on the DACs.** The cables and the CX7 silicon are 400G parts and
   the port advertises `200G_2X,200G_4X,...` only. `mstlink -d <bdf>
   --speeds 400G_4X` on both ends of one cable (link drops while it
   retrains) would show whether the Spark's firmware/board allow it. The
   second diagram labels the ports 200G, so expect "no". It would lift one
   cable from 24.5 to at most 30 GB/s; a node fed by both neighbours is
   already PCIe-bound (section 14) and would gain nothing.
3. **GPU reads of a PCIe BAR through ATS.** `mmap` a BAR (`resource0` of a
   device whose registers are side-effect free to read, e.g. the NVMe
   controller's CAP/VS at offset 0), `cudaHostRegister(...,
   cudaHostRegisterIoMemory)`, read it from a kernel. If the SMMU forwards
   the Device-memory translation, the GPU can stage FPGA/BAR data itself at
   C2C speed and `gb10_dmab`'s BAR endpoint becomes a fallback. If not, the
   process gets an Xid; run it with nothing else on the GPU.
4. **CPUEB telemetry reader.** A small signed module that `ioremap`s the
   `MTEL` region (read-only SRAM) and exposes the perf-counter and fastdvfs
   logs in debugfs; the `_DSM` gives the offsets. This is the only visible
   window into the DVFS microcontroller and might show the DSU/fabric and
   DRAM operating points directly, which would turn experiment 1 from a
   hypothesis into a knob.
5. **Identify `DMA0`.** Same kind of module, reading the 4 KiB at
   0x14200000 and matching it against `mtk-cqdma`/`mtk-hsdma` register
   layouts. Only after the clock question is settled; see above.
6. **ATS on the NIC.** `mstconfig -d 0000:01:00.0 set ATS_ENABLED=1` (both
   halves, reboot) lets the CX7 cache SMMU translations and makes ODP
   (`IBV_ACCESS_ON_DEMAND`, implicit MRs) viable for "register nothing"
   RDMA into any process memory. Bandwidth will not change; registration
   cost and the NIC's behaviour under page migration will.

## 14. Into the head: per-node ceilings and `ringgather` (2026-11-13)

Section 13 measured one cable. This section asks what the two neighbours can
deliver into one node at the same time, the head (DGX-1 here), and adds the
function that does it.

### The two PCIe links, not the two ports, limit a node

Each PCIe x4 link of the ConnectX-7 carries *both* ports, so two 200G ports
(400 Gb/s of line rate) sit behind two Gen5 x4 links. On the head both links
train at 32 GT/s x4 with `MaxPayload 512` (the most either end supports),
`MaxReadReq 512` and no AER errors; `pci=pcie_bus_safe` is on the command
line.

`ib_write_bw -s 1048576 -q 4 -D 12 -x 3`, servers on the head, clients started
on the neighbours over ssh, every mix running at the same time. A and B are
the head's PCIe links (`0000`, `0002`); 0 and 1 are its ports (to DGX-2 and
to DGX-3). Gb/s of RDMA payload:

| Flows into the head | Per flow | Total | Bound by |
|---|---|---|---|
| A0 | 111.9 | 111.9 (14.0 GB/s) | the sender's PCIe link |
| A0 + B0, one cable, both halves | 98.0 each | 196.1 (24.5 GB/s) | the 200G port |
| A1 + B1, the other cable | 98.0 each | 196.1 (24.5 GB/s) | the 200G port |
| A0 + B1, two cables, one flow per link | 111.9 each | 223.7 (28.0 GB/s) | the senders' PCIe links |
| A0 + A1, two cables into the same link | 60.3 each | 120.6 (15.1 GB/s) | one PCIe link of the head |
| B0 + B1, same on the other link | 60.3 each | 120.6 (15.1 GB/s) | one PCIe link of the head |
| **A0 + B0 + A1 + B1** | 60.3 each | **241.2 (30.15 GB/s)** | both PCIe links of the head |

The head's `rx_bytes_phy` counters agree: 244.8 Gb/s on the wire for the
four-flow mix, headers included. The other direction, head to neighbours
(`--reversed`), is lower: 111.9 Gb/s from one link whether it feeds one port
or two, 222 Gb/s (27.7 GB/s) from both.

What this says:

- **Into a node, one x4 link takes 120.6 Gb/s (15.1 GB/s)** no matter which
  ports feed it. That is the link itself: Gen5 x4 carries 15.75 GB/s raw and
  about 15.1 GB/s once every 512 bytes of payload has paid for its TLP
  framing. Nothing in software is left to tune. Queue pairs (1, 4, 16) and
  message size (64 KiB, 1 MiB, 8 MiB) all give 241 Gb/s.
- **Out of a node, one link gives 111.9 Gb/s (14.0 GB/s).** The NIC has to
  read host memory with requests of at most `MaxReadReq` = 512 bytes. Raising
  that to 4096 is the one PCIe knob not yet tried; it could only help the
  sending side.
- **The most a node can receive is 30.15 GB/s, and only with four flows:**
  both neighbours sending, each on both of its PCIe halves. One neighbour
  alone delivers 24.5 GB/s (its cable). Two neighbours with one flow each
  deliver 28.0 GB/s, because then each sender's single link is the limit.
- 400G optics would not raise a node's total. They would only let one cable
  reach what two already do.

### `tools/ringgather`: the other two GB10s feeding the head

`ringgather` is that four-flow pattern as a function. The head opens one RC
connection per worker address, one address per link, and then per gather:

1. it splits its destination buffer into one contiguous slice per link and
   registers each slice on that link's RoCE device only;
2. **FILL**: it asks every worker to produce its slice. The worker's GPU
   writes the data into NIC-registered system memory and acks with the time
   it took;
3. **PUSH**: it tells every worker where its slice lives in the head's
   memory (address and rkey). The workers RDMA-WRITE in 1 MiB chunks, 32 in
   flight, the last one with an immediate;
4. it returns when every slice has landed. The head's GPU then reads the
   whole buffer straight from the NIC-written pages.

The slices carry one continuous pattern over the whole destination, so a
single verify kernel on the head proves that the GB10s together produced one
coherent buffer. The reusable part is four functions in `tools/ringgather.c`:

```c
struct ring_gather *rg_open(char **worker_ips, int n, int port, int wait_s);
int  rg_bind(struct ring_gather *g, void *dst, size_t len, size_t chunk);
int  rg_gather(struct ring_gather *g, uint64_t seed, uint32_t iter, struct rg_stats *st);
void rg_close(struct ring_gather *g);
```

On the worker, `produce()` is the one place where a real workload would put
its own output instead of the test pattern; it is handed the registered
staging buffer, so whatever the GPU writes there goes to the wire without a
copy.

```bash
# from the head: discovers the shared subnets, copies the binary, starts the workers, gathers
scripts/ring-gather.sh -s 4096 -n 5 DGX-2 DGX-3

# by hand
tools/ringgather --worker                                             # on DGX-2 and on DGX-3
tools/ringgather 10.100.32.1 10.100.33.1 10.100.36.2 10.100.37.2      # on DGX-1
```

Results on 2026-11-13, GPU produce on the workers and GPU verify on the head:

| Workers | Links | Gathered | Landed in the head | Head GPU verify |
|---|---|---|---|---|
| DGX-2 + DGX-3 | 4 | 5 x 4 GiB | 29.65 GB/s average (237 Gb/s), best 29.85 | 141 GB/s, all OK |
| DGX-2 + DGX-3 | 4 | 3 x 16 GiB | 29.73 GB/s average (238 Gb/s), best 29.77 | 132 GB/s, all OK |
| DGX-2 alone | 2 | 3 x 4 GiB | 24.45 GB/s (196 Gb/s) | 140 GB/s, all OK |
| DGX-3 alone | 2 | 3 x 4 GiB | 24.45 GB/s (196 Gb/s) | all OK |

With both neighbours every link runs at 59 to 62 Gb/s and the total is 98.5 %
of the `ib_write_bw` ceiling above. One 16 GiB gather takes 76 ms to produce
(each worker's GPU writes its 8 GiB at 114 GB/s), 578 ms on the wire and
130 ms to verify on the head: the GPUs on both ends are four to five times
faster than the NIC path, so the PCIe links are the only thing worth
optimising around. The first gather after a connect is slower to produce
(275 ms) because the workers allocate and register their staging memory
then.

Notes:

- The slices are equal because the links are. To leave a neighbour out, do
  not pass its addresses; the rest still gathers.
- A worker serves any number of connections and sleeps while the head is
  quiet. `-k N` makes it exit after N connections have closed and `-i secs`
  after that long without one, which is how `ring-gather.sh` leaves nothing
  running on the other nodes.
- There is no authentication. A worker stages whatever size it is asked for
  (up to 64 GiB) and writes to whatever address and rkey the head names, so
  run it on the private ring only.
- The opposite direction, the head scattering to both neighbours, is not
  built; the outbound numbers above say it would top out at 27.7 GB/s.
