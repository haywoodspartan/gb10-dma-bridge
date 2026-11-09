#!/usr/bin/env bash
# Root-level steps for one node of the Spark ring. Run through scripts/ring-admin.sh
# (which copies it to each node and runs it under sudo), or directly:
#
#   sudo scripts/node-admin.sh [--mtu 9000] [--isolcpus 16-19] [-p prefix] [--status]
#
#   --mtu N        set N live on every ring port (IPv4 under PREFIX, default 10.100.)
#                  and persist it in /etc/netplan/99-zz-spark-ring-mtu.yaml, which sorts
#                  after NVIDIA Sync's file and merges only `mtu:` into those interfaces;
#                  validated with `netplan generate` (nothing is re-applied)
#   --isolcpus L   stage isolcpus/nohz_full/rcu_nocbs/irqaffinity for cpulist L in GRUB
#                  via isolate-cores.sh --apply (takes effect at the next reboot)
#   --status       report only
set -uo pipefail

MTU=""; ISO=""; PREFIX="10.100."; STATUS=0
while (($#)); do
	case $1 in
	--mtu) MTU=$2; shift ;;
	--isolcpus) ISO=$2; shift ;;
	-p|--prefix) PREFIX=$2; shift ;;
	--status) STATUS=1 ;;
	-h|--help) sed -n '2,14p' "$0"; exit 0 ;;
	*) echo "unknown option $1" >&2; exit 2 ;;
	esac
	shift
done
[[ $EUID -eq 0 ]] || { echo "node-admin.sh must run as root" >&2; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)
DROPIN=/etc/netplan/99-zz-spark-ring-mtu.yaml
H=$(hostname)

mapfile -t DEVS < <(ip -4 -o addr show scope global | awk -v p="$PREFIX" 'index($4, p) == 1 {print $2}' | sort -u)

if [[ -n $MTU ]]; then
	((${#DEVS[@]})) || { echo "[$H] no ring ports found under $PREFIX" >&2; exit 1; }
	for d in "${DEVS[@]}"; do
		ip link set dev "$d" mtu "$MTU" || { echo "[$H] cannot set mtu on $d" >&2; exit 1; }
	done
	{
		echo "# Written by gb10-dma-bridge/scripts/node-admin.sh: MTU for the Spark ring ports."
		echo "# Merges with 99-nvidia-sync-cluster.yaml (this file sorts later). Remove to undo."
		echo "network:"
		echo "  version: 2"
		echo "  ethernets:"
		for d in "${DEVS[@]}"; do echo "    $d:"; echo "      mtu: $MTU"; done
	} >"$DROPIN"
	chmod 600 "$DROPIN"
	if ! out=$(netplan generate 2>&1); then
		echo "[$H] netplan generate FAILED; removing $DROPIN:" >&2
		echo "$out" | sed 's/^/    /' >&2
		rm -f "$DROPIN"
		exit 1
	fi
	echo "[$H] MTU $MTU live on ${DEVS[*]}; persisted in $DROPIN"
fi

if [[ -n $ISO ]]; then
	if [[ -x $HERE/isolate-cores.sh ]]; then
		CORES=$ISO bash "$HERE/isolate-cores.sh" --apply -c "$ISO" || { echo "[$H] isolate-cores.sh --apply failed" >&2; exit 1; }
		echo "[$H] isolcpus=$ISO staged in GRUB: REBOOT REQUIRED to take effect"
	else
		echo "[$H] isolate-cores.sh not found next to this script" >&2
		exit 1
	fi
fi

echo "[$H] ring ports:"
for d in "${DEVS[@]}"; do
	ib=$(ls /sys/class/net/$d/device/infiniband 2>/dev/null | head -1)
	amtu=$(ibv_devinfo -d "$ib" 2>/dev/null | awk '/active_mtu/ {print $2; exit}')
	printf "  %-16s mtu %-5s rdma %-14s active_mtu %s\n" "$d" "$(cat /sys/class/net/$d/mtu)" "${ib:-?}" "${amtu:-?}"
done
if [[ -x $HERE/isolate-cores.sh ]]; then
	echo "[$H] isolation:"
	CORES=${ISO:-16-19} bash "$HERE/isolate-cores.sh" --status 2>/dev/null | sed 's/^/  /' || true
fi
