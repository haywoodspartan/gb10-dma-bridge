#!/usr/bin/env bash
# Interactive root changes across the Spark ring: copies node-admin.sh and
# isolate-cores.sh to each remote node and runs node-admin.sh under sudo with a
# TTY, so you type each node's sudo password as it comes up. The local node is
# run directly with sudo.
#
#   scripts/ring-admin.sh [--mtu 9000] [--isolcpus 16-19] [--status] NODE NODE NODE...
#
# Nothing reboots; --isolcpus only stages the GRUB change and says so.
set -uo pipefail

ARGS=(); NODES=()
while (($#)); do
	case $1 in
	--mtu|--isolcpus|-p|--prefix) ARGS+=("$1" "$2"); shift ;;
	--status) ARGS+=("$1") ;;
	-h|--help) sed -n '2,10p' "$0"; exit 0 ;;
	-*) echo "unknown option $1" >&2; exit 2 ;;
	*) NODES+=("$1") ;;
	esac
	shift
done
((${#NODES[@]})) || { echo "need at least one node" >&2; exit 2; }
((${#ARGS[@]})) || ARGS=(--status)

HERE=$(cd "$(dirname "$0")" && pwd)
REMOTE=/tmp/gb10-admin
mapfile -t LOCAL_IPS < <(ip -4 -o addr show | awk '{print $4}' | cut -d/ -f1)
is_local() { local n=${1#*@} ip; for ip in "${LOCAL_IPS[@]}"; do [[ $n == "$ip" ]] && return 0; done; [[ $n == "$(hostname)" || $n == "$(hostname -s)" || $n == localhost ]]; }

rc=0
for n in "${NODES[@]}"; do
	echo
	echo "================ $n ================"
	if is_local "$n"; then
		sudo bash "$HERE/node-admin.sh" "${ARGS[@]}" || { rc=1; echo "FAILED on $n"; }
	else
		ssh -o ConnectTimeout=5 "$n" "mkdir -p $REMOTE" &&
		scp -q "$HERE/node-admin.sh" "$HERE/isolate-cores.sh" "$n:$REMOTE/" &&
		ssh -t -o ConnectTimeout=5 "$n" "sudo bash $REMOTE/node-admin.sh ${ARGS[*]}" || { rc=1; echo "FAILED on $n"; }
	fi
done
echo
((rc == 0)) && echo "ring-admin: all nodes done" || echo "ring-admin: some nodes failed"
exit $rc
