#!/usr/bin/env bash
# Set the MTU of every ring port on every node, live and persistently.
#
#   scripts/ring-mtu.sh [--mtu 9000] [-p prefix] [-u user] [--check] NODE NODE NODE...
#
# Ring ports are the netdevs carrying an IPv4 address under PREFIX (default
# 10.100.). For each node (over SSH, the local machine directly) the script:
#   1. sets the MTU live with `ip link set` (no link bounce),
#   2. writes /etc/netplan/99-zz-spark-ring-mtu.yaml, which sorts after
#      NVIDIA Sync's 99-nvidia-sync-cluster.yaml and merges only `mtu:` into
#      those interfaces, so it survives reboots and Sync rewriting its file,
#   3. runs `netplan generate` to validate the merged configuration (no apply),
#   4. reports netdev MTU and the RoCE active_mtu of every ring port.
# Needs passwordless sudo on every node. --check only reports.
# Undo: remove the drop-in and set the MTU back with `ip link set ... mtu 1500`.
set -euo pipefail

MTU=9000; PREFIX="10.100."; USER_=""; CHECK=0; NODES=()
while (($#)); do
	case $1 in
	--mtu) MTU=$2; shift ;;
	-p) PREFIX=$2; shift ;;
	-u) USER_=$2; shift ;;
	--check) CHECK=1 ;;
	-h|--help) sed -n '2,17p' "$0"; exit 0 ;;
	-*) echo "unknown option $1" >&2; exit 2 ;;
	*) NODES+=("$1") ;;
	esac
	shift
done
((${#NODES[@]})) || { echo "need at least one node" >&2; exit 2; }
DROPIN=/etc/netplan/99-zz-spark-ring-mtu.yaml
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new)

mapfile -t LOCAL_IPS < <(ip -4 -o addr show | awk '{print $4}' | cut -d/ -f1)
is_local() { local n=${1#*@} ip; for ip in "${LOCAL_IPS[@]}"; do [[ $n == "$ip" ]] && return 0; done; [[ $n == "$(hostname)" || $n == "$(hostname -s)" || $n == localhost ]]; }
target() { local n=$1; [[ -n $USER_ && $n != *@* ]] && n="$USER_@$n"; echo "$n"; }
run_on() { local n=$1; shift; if is_local "$n"; then bash -c "$*"; else ssh "${SSH_OPTS[@]}" "$(target "$n")" "$*"; fi; }

# the per-node worker, shipped over ssh as a single bash -c string
read -r -d '' WORKER <<'EOW' || true
set -u
MTU=$1; PREFIX=$2; CHECK=$3; DROPIN=$4
mapfile -t DEVS < <(ip -4 -o addr show scope global | awk -v p="$PREFIX" 'index($4, p) == 1 {print $2}' | sort -u)
((${#DEVS[@]})) || { echo "  no ring ports found under $PREFIX"; exit 1; }
if ((!CHECK)); then
	sudo -n true 2>/dev/null || { echo "  passwordless sudo not available on $(hostname); use scripts/ring-admin.sh --mtu $MTU (interactive)"; exit 3; }
	for d in "${DEVS[@]}"; do sudo -n ip link set dev "$d" mtu "$MTU" || exit 4; done
	{
		echo "# Written by gb10-dma-bridge/scripts/ring-mtu.sh: MTU for the Spark ring ports."
		echo "# Merges with 99-nvidia-sync-cluster.yaml (this file sorts later). Remove to undo."
		echo "network:"
		echo "  version: 2"
		echo "  ethernets:"
		for d in "${DEVS[@]}"; do echo "    $d:"; echo "      mtu: $MTU"; done
	} | sudo -n tee "$DROPIN" >/dev/null || exit 5
	sudo -n chmod 600 "$DROPIN"
	if ! out=$(sudo -n netplan generate 2>&1); then
		echo "  netplan generate FAILED, removing $DROPIN:"; echo "$out" | sed 's/^/    /'
		sudo -n rm -f "$DROPIN"; exit 6
	fi
fi
for d in "${DEVS[@]}"; do
	pci=$(basename "$(readlink /sys/class/net/$d/device)")
	ib=$(ls /sys/class/net/$d/device/infiniband 2>/dev/null | head -1)
	amtu=$(cat /sys/class/infiniband/$ib/ports/1/*mtu* 2>/dev/null | head -1)
	[[ -z $amtu ]] && amtu=$(ibv_devinfo -d "$ib" 2>/dev/null | awk '/active_mtu/ {print $2; exit}')
	printf "  %-16s %-14s mtu %-5s rdma %-14s active_mtu %s\n" "$d" "$pci" "$(cat /sys/class/net/$d/mtu)" "$ib" "${amtu:-?}"
done
((CHECK)) || echo "  persisted in $DROPIN ($(sudo -n netplan get ethernets 2>/dev/null | grep -c "mtu: $MTU") interfaces carry mtu: $MTU in the merged netplan)"
EOW

rc=0
for n in "${NODES[@]}"; do
	echo "== $n =="
	run_on "$n" "bash -c $(printf '%q' "$WORKER") -- $MTU $PREFIX $CHECK $DROPIN" || { rc=$?; echo "  FAILED on $n (rc $rc)"; }
done
exit $rc
