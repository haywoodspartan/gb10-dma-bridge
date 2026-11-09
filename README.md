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
| ConnectX-7, 2 ports / 4 PFs | `0000:01:00.{0,1}`, `0002:01:00.{0,1}` | PCIe Gen5 x4 (~16 GB/s each) | `mlx5_core`, RoCE devices present |
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

Default `cores=16-19`: four Cortex-X925 cores of the second cluster (the
X925s are cpus 5-9 and 15-19; the A725s are 0-4 and 10-14). cpu 15 stays a
big core for housekeeping in that cluster, and cluster 0-9 is untouched. The
X925s give the highest per-core copy bandwidth; if you would rather keep the
big cores for compute, four A725s (`cores=11-14`) still outrun any PCIe
endpoint on the Spark. `scripts/isolate-cores.sh --status` prints the core
type of each chosen CPU so the choice is visible. Override with `cores=`; the
module refuses offline CPUs and refuses to take every CPU.

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
`make tools`. DKMS users: `dkms add .` with the provided `dkms.conf` (DKMS
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
scripts/ring-test.sh    discovers the cables over SSH and runs ringbw on every cable of the ring
scripts/ring-mtu.sh     MTU on every ring port of every node (passwordless sudo variant)
scripts/node-admin.sh   per-node root worker: MTU live + netplan drop-in, isolcpus staging, status
scripts/ring-admin.sh   runs node-admin.sh on every node (interactive sudo or passwordless)
scripts/cluster-bootstrap.sh  passwordless sudo + full SSH mesh + project sync across the nodes
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
three nodes over SSH. Each Spark's two 400G QSFP ports are split into two
200 GbE halves (`f0np0` and `f1np1` of each PCIe endpoint), and each half goes
to a different neighbour, so every pair of nodes is joined by **two** 200 GbE
cables and every node has four ring links:

| Cable (/24) | Node A side | Node B side |
|---|---|---|
| 10.100.32 | DGX-1 `enp1s0f0np0` .2 | DGX-2 `enp1s0f1np1` .1 |
| 10.100.33 | DGX-1 `enP2p1s0f0np0` .2 | DGX-2 `enP2p1s0f1np1` .1 |
| 10.100.34 | DGX-2 `enp1s0f0np0` .1 | DGX-3 `enp1s0f1np1` .2 |
| 10.100.35 | DGX-2 `enP2p1s0f0np0` .1 | DGX-3 `enP2p1s0f1np1` .2 |
| 10.100.36 | DGX-3 `enp1s0f0np0` .2 | DGX-1 `enp1s0f1np1` .1 |
| 10.100.37 | DGX-3 `enP2p1s0f0np0` .2 | DGX-1 `enP2p1s0f1np1` .1 |

All six links are 200 Gb/s RoCE v2, ACTIVE, sub-millisecond ping. Each node's
`enp1s0*` links share PCIe endpoint `0000:01:00.x` and its `enP2p1s0*` links
share `0002:01:00.x`, both Gen5 x4, so each endpoint carries one link in and
one link out of the ring direction. `rdma-core 50`, `perftest 24.01` (no CUDA),
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

Every iteration verified on the receiving GPU. That is about 81 GB/s of
GPU-produced, GPU-consumed data moving around the ring at once, 27 GB/s out
and 27 GB/s in per node, with each PCIe Gen5 x4 endpoint at its practical
limit in both directions.

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

State on 2026-11-09 after running them: every node reaches every other with
key-based ssh and passwordless sudo; MTU 9000 everywhere; `isolcpus=domain,
managed_irq,16-19 nohz_full=16-19 rcu_nocbs=16-19 irqaffinity=0-15` staged in
`/etc/default/grub.d/90-gb10-dmab.cfg` on all three nodes, **pending a reboot**
of each node; `/etc/modprobe.d/gb10_dmab.conf` carries `cores=16-19`.

Each sender iteration prints wire GB/s; each receiver iteration prints the
GPU verify time, which is also the GPU's read bandwidth from NIC-written
memory. Measured in loopback on this node (512 MiB x 5, GPU mode): 13.25 GB/s
(106 Gb/s) through the CX7, GPU fill 5 ms, GPU verify 3.3 ms = about 160 GB/s
read, every iteration OK. Across a real link expect the same order: the PCIe
Gen5 x4 endpoint, not the 200 Gb/s port, is the limit.

For collectives, NCCL is not installed here; when you add it, expect it to
use host (system) buffers (`NCCL_NET_GDR_LEVEL` will not apply, there is no
GDR on GB10) and both ring links (`NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0`).
