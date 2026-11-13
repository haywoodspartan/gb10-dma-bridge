#!/usr/bin/env bash
# Gather GPU-produced data from the neighbouring DGX Spark nodes into this one
# (tools/ringgather): the other GB10s produce, every ConnectX-7 path carries,
# this node's GPU verifies the combined buffer.
#
#   scripts/ring-gather.sh [-s MiB] [-n iters] [-u user] [-p prefix] [-P port] [--cpu] WORKER [WORKER...]
#
# Run it on the head. WORKER is an SSH destination (hostname, IP, or user@host).
# The script asks each worker for its ring addresses (IPv4 /24 under PREFIX,
# default 10.100.), keeps the ones on a subnet this machine is on too, copies
# tools/ringgather to the worker, starts it there in --worker mode and runs
# the head against every such address at once. With Socket Direct a neighbour
# shares two subnets with the head, one per PCIe x4 half, so two neighbours
# give four links and fill both of the head's PCIe links (30 GB/s).
#
# Workers exit when the head closes its connections, or after 60 s without
# one, so nothing is left running on the other nodes.
#
#   --cpu   pass -C to ringgather (CPU fill/verify instead of the GPU)
set -euo pipefail

SIZE=4096; ITERS=5; USER_=""; PREFIX="10.100."; PORT=18520; CPU=""
WORKERS=()
while (($#)); do
	case $1 in
	-s) SIZE=$2; shift ;;
	-n) ITERS=$2; shift ;;
	-u) USER_=$2; shift ;;
	-p) PREFIX=$2; shift ;;
	-P) PORT=$2; shift ;;
	--cpu) CPU="-C" ;;
	-h|--help) sed -n '2,20p' "$0"; exit 0 ;;
	-*) echo "unknown option $1" >&2; exit 2 ;;
	*) WORKERS+=("$1") ;;
	esac
	shift
done
((${#WORKERS[@]} >= 1)) || { echo "need at least one worker node" >&2; exit 2; }

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=$ROOT/tools/ringgather
[[ -x $BIN ]] || { echo "build tools first: make -C $ROOT/tools" >&2; exit 1; }
REMOTE_DIR=/tmp/ringgather-$(id -un)
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new)
CUDA_LIB=${CUDA_HOME:-/usr/local/cuda}/lib64

target() { local n=$1; [[ -n $USER_ && $n != *@* ]] && n="$USER_@$n"; echo "$n"; }
# "dev ip" for every /24 ring address (the /32 router IDs on lo are not links)
RING_ADDRS="ip -4 -o addr show scope global | awk '\$4 ~ /^${PREFIX//./\\.}/ && \$4 ~ /\\/24\$/ {sub(/\\/24\$/, \"\", \$4); print \$2, \$4}'"

declare -A LOCAL_DEV
while read -r dev ip; do LOCAL_DEV[${ip%.*}]=$dev; done < <(bash -c "$RING_ADDRS")
((${#LOCAL_DEV[@]})) || { echo "this machine has no ring address under $PREFIX" >&2; exit 1; }

# 1. links: every subnet a worker shares with this machine
echo "== links into $(hostname) =="
declare -A NLINKS
IPS=()
for w in "${WORKERS[@]}"; do
	NLINKS[$w]=0
	while read -r dev ip; do
		subnet=${ip%.*}
		[[ -n ${LOCAL_DEV[$subnet]:-} ]] || continue
		IPS+=("$ip"); NLINKS[$w]=$((NLINKS[$w] + 1))
		printf "  %-14s %-16s %-14s -> %s\n" "$w" "$dev" "$ip" "${LOCAL_DEV[$subnet]}"
	done < <(ssh "${SSH_OPTS[@]}" "$(target "$w")" "$RING_ADDRS" | sort -k2,2V)
	((NLINKS[$w])) || { echo "$w shares no ring subnet with this machine" >&2; exit 1; }
done

# 2. distribute the binary (and the NVRTC libraries where the CUDA toolkit is absent)
echo "== copying ringgather =="
declare -A ENVPFX
for w in "${WORKERS[@]}"; do
	t=$(target "$w")
	ssh "${SSH_OPTS[@]}" "$t" "mkdir -p $REMOTE_DIR"
	scp -q "${SSH_OPTS[@]}" "$BIN" "$t:$REMOTE_DIR/ringgather"
	missing=$(ssh "${SSH_OPTS[@]}" "$t" "ldd $REMOTE_DIR/ringgather 2>/dev/null | awk '/not found/ {print \$1}'" || true)
	if [[ -n $missing ]]; then
		if [[ $missing == *libnvrtc* && -d $CUDA_LIB ]]; then
			scp -q "${SSH_OPTS[@]}" "$CUDA_LIB"/libnvrtc.so.* "$CUDA_LIB"/libnvrtc-builtins.so.* "$t:$REMOTE_DIR/"
			ENVPFX[$w]="LD_LIBRARY_PATH=$REMOTE_DIR "
			echo "  $w: $REMOTE_DIR (plus NVRTC libraries)"
		else
			echo "  $w: missing libraries: $missing" >&2; exit 1
		fi
	else
		echo "  $w: $REMOTE_DIR"
	fi
done

# 3. workers, then the head; the head retries its connects until they listen
LOG=$(mktemp -d "${TMPDIR:-/tmp}/ring-gather.XXXXXX")
pids=()
trap '((${#pids[@]})) && kill "${pids[@]}" 2>/dev/null || true' EXIT
for w in "${WORKERS[@]}"; do
	ssh "${SSH_OPTS[@]}" "$(target "$w")" \
		"${ENVPFX[$w]:-}$REMOTE_DIR/ringgather --worker -p $PORT -k ${NLINKS[$w]} -i 60 $CPU" >"$LOG/$w.log" 2>&1 &
	pids+=($!)
done
echo "== gathering $ITERS x $SIZE MiB over ${#IPS[@]} link(s) (worker logs in $LOG) =="
rc=0
"$BIN" -s "$SIZE" -n "$ITERS" -p "$PORT" $CPU "${IPS[@]}" || rc=$?
for i in "${!pids[@]}"; do wait "${pids[$i]}" || { rc=1; echo "  worker ${WORKERS[$i]} did not exit cleanly"; }; done
pids=()

echo
for w in "${WORKERS[@]}"; do
	echo "---- $w ----"
	grep -E "iter $((ITERS - 1)):|served|refused|failed|error|rror:" "$LOG/$w.log" || tail -n 5 "$LOG/$w.log"
done
((rc == 0)) && echo "RING GATHER PASSED: ${#WORKERS[@]} worker(s), ${#IPS[@]} link(s)" || echo "RING GATHER FAILED (see $LOG)"
exit $rc
