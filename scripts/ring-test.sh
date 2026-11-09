#!/usr/bin/env bash
# Run the GPU-fed RDMA ring test (tools/ringbw) on every cable of a DGX Spark ring.
#
#   scripts/ring-test.sh [-s MiB] [-n iters] [-u user] [-p prefix] [--mtu 9000] [--cpu] NODE NODE NODE...
#
# NODE is an SSH destination (hostname, IP, or user@host); the local machine is
# detected and run directly. The script asks each node for its ring addresses
# (IPv4 under PREFIX, default 10.100.), treats every /24 shared by exactly two
# nodes as one cable, groups cables by node pair, checks that the pairs form a
# ring, copies tools/ringbw to the remote nodes, and runs one ringbw per cable
# in the ring direction: on each cable the predecessor sends and the successor
# listens. With two cables per pair (split 400G ports) every node runs two
# processes, each listening on one cable from its predecessor and sending on
# one cable to its successor. Everything runs at once.
#
#   --mtu 9000   first set that MTU on every ring port of every node (sudo, not persistent)
#   --cpu        pass -C to ringbw (CPU fill/verify instead of the GPU)
set -euo pipefail

SIZE=1024; ITERS=10; USER_=""; PREFIX="10.100."; MTU=""; CPU=""
NODES=()
while (($#)); do
	case $1 in
	-s) SIZE=$2; shift ;;
	-n) ITERS=$2; shift ;;
	-u) USER_=$2; shift ;;
	-p) PREFIX=$2; shift ;;
	--mtu) MTU=$2; shift ;;
	--cpu) CPU="-C" ;;
	-h|--help) sed -n '2,19p' "$0"; exit 0 ;;
	-*) echo "unknown option $1" >&2; exit 2 ;;
	*) NODES+=("$1") ;;
	esac
	shift
done
((${#NODES[@]} >= 2)) || { echo "need at least two nodes" >&2; exit 2; }
N=${#NODES[@]}

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=$ROOT/tools/ringbw
[[ -x $BIN ]] || { echo "build tools first: make -C $ROOT tools" >&2; exit 1; }
REMOTE_DIR=/tmp/ringbw-$(id -un)
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new)
CUDA_LIB=${CUDA_HOME:-/usr/local/cuda}/lib64

mapfile -t LOCAL_IPS < <(ip -4 -o addr show | awk '{print $4}' | cut -d/ -f1)
is_local() { local n=${1#*@} ip; for ip in "${LOCAL_IPS[@]}"; do [[ $n == "$ip" ]] && return 0; done; [[ $n == "$(hostname)" || $n == "$(hostname -s)" || $n == localhost ]]; }
target() { local n=$1; [[ -n $USER_ && $n != *@* ]] && n="$USER_@$n"; echo "$n"; }
run_on() { local n=$1; shift; if is_local "$n"; then bash -c "$*"; else ssh "${SSH_OPTS[@]}" "$(target "$n")" "$*"; fi; }
pairkey() { if [[ $1 < $2 ]]; then echo "$1|$2"; else echo "$2|$1"; fi; }

# 1. ring addresses: ADDR["node|subnet"]=ip, DEV["node|subnet"]=dev
declare -A ADDR DEV SUBNET_NODES
echo "== discovering ring addresses (prefix $PREFIX) =="
for n in "${NODES[@]}"; do
	while read -r dev cidr; do
		ip=${cidr%/*}; subnet=${ip%.*}
		ADDR["$n|$subnet"]=$ip; DEV["$n|$subnet"]=$dev
		SUBNET_NODES[$subnet]="${SUBNET_NODES[$subnet]:-} $n"
		printf "  %-14s %-16s %s\n" "$n" "$dev" "$cidr"
	done < <(run_on "$n" "ip -4 -o addr show scope global | awk '\$4 ~ /^${PREFIX//./\\.}/ {print \$2, \$4}'")
done

# 2. cables (subnets shared by exactly two nodes) grouped by node pair
declare -A PAIR_SUBNETS NEIGHBOURS
for subnet in "${!SUBNET_NODES[@]}"; do
	read -ra m <<<"${SUBNET_NODES[$subnet]}"
	if ((${#m[@]} != 2)); then
		echo "  note: $subnet.0/24 has ${#m[@]} member(s) among the given nodes (${m[*]:-none}); ignored" >&2
		continue
	fi
	k=$(pairkey "${m[0]}" "${m[1]}")
	PAIR_SUBNETS[$k]="${PAIR_SUBNETS[$k]:-} $subnet"
	[[ " ${NEIGHBOURS[${m[0]}]:-} " == *" ${m[1]} "* ]] || NEIGHBOURS[${m[0]}]="${NEIGHBOURS[${m[0]}]:-} ${m[1]}"
	[[ " ${NEIGHBOURS[${m[1]}]:-} " == *" ${m[0]} "* ]] || NEIGHBOURS[${m[1]}]="${NEIGHBOURS[${m[1]}]:-} ${m[0]}"
done
echo "== cables =="
for k in "${!PAIR_SUBNETS[@]}"; do
	read -ra subs <<<"${PAIR_SUBNETS[$k]}"
	printf "  %-28s %d cable(s): %s\n" "${k/|/ <-> }" "${#subs[@]}" "${subs[*]/%/.0/24}"
done
for n in "${NODES[@]}"; do
	read -ra nb <<<"${NEIGHBOURS[$n]:-}"
	want=2; ((N == 2)) && want=1
	((${#nb[@]} == want)) || { echo "$n has ${#nb[@]} neighbour(s) (${nb[*]:-none}); a ring node needs $want" >&2; exit 1; }
done

# 3. walk the ring
ORDER=("${NODES[0]}"); cur=${NODES[0]}; prev=""
for ((i = 1; i < N; i++)); do
	read -ra nb <<<"${NEIGHBOURS[$cur]}"
	next=""; for c in "${nb[@]}"; do [[ $c != "$prev" ]] && { next=$c; break; }; done
	[[ -n $next ]] || { echo "cannot walk the ring from $cur" >&2; exit 1; }
	ORDER+=("$next"); prev=$cur; cur=$next
done
read -ra nb <<<"${NEIGHBOURS[$cur]}"
[[ " ${nb[*]} " == *" ${NODES[0]} "* ]] || { echo "the nodes do not close into a ring (${ORDER[*]} -> ?)" >&2; exit 1; }

# 4. per node: cables to the successor (send) and from the predecessor (listen)
declare -A SEND_TO LISTEN_ON SEND_DEV	# "node|j" -> address / dev
MAXJ=0
echo "== ring order: ${ORDER[*]} -> ${ORDER[0]} =="
for ((i = 0; i < N; i++)); do
	X=${ORDER[$i]}; Y=${ORDER[$(( (i + 1) % N ))]}; P=${ORDER[$(( (i + N - 1) % N ))]}
	read -ra succ <<<"${PAIR_SUBNETS[$(pairkey "$X" "$Y")]}"
	read -ra pred <<<"${PAIR_SUBNETS[$(pairkey "$P" "$X")]}"
	((N == 2)) && pred=("${succ[@]}")	# a pair: the same cables carry both directions
	j=0
	while ((j < ${#succ[@]} || j < ${#pred[@]})); do
		((j < ${#succ[@]})) && { SEND_TO["$X|$j"]=${ADDR["$Y|${succ[$j]}"]}; SEND_DEV["$X|$j"]=${DEV["$X|${succ[$j]}"]}; }
		((j < ${#pred[@]})) && LISTEN_ON["$X|$j"]=${ADDR["$X|${pred[$j]}"]}
		j=$((j + 1))
	done
	((j > MAXJ)) && MAXJ=$j
	for ((k = 0; k < j; k++)); do
		printf "  %-14s #%d listen %-14s send -> %-14s via %s\n" "$X" "$k" "${LISTEN_ON["$X|$k"]:--}" \
			"${SEND_TO["$X|$k"]:--}" "${SEND_DEV["$X|$k"]:--}"
	done
done

# 5. optional MTU
if [[ -n $MTU ]]; then
	echo "== setting MTU $MTU on all ring ports =="
	for n in "${NODES[@]}"; do
		for k in "${!DEV[@]}"; do
			[[ $k == "$n|"* ]] || continue
			if run_on "$n" "sudo -n ip link set dev ${DEV[$k]} mtu $MTU"; then
				echo "  $n ${DEV[$k]} mtu $MTU"
			else
				echo "  $n ${DEV[$k]}: could not set MTU (passwordless sudo needed)"
			fi
		done
	done
	sleep 2
fi

# 6. distribute the binary (and the NVRTC libraries where the CUDA toolkit is absent)
echo "== copying ringbw =="
declare -A ENVPFX
for n in "${NODES[@]}"; do
	is_local "$n" && continue
	run_on "$n" "mkdir -p $REMOTE_DIR"
	scp -q "${SSH_OPTS[@]}" "$BIN" "$(target "$n"):$REMOTE_DIR/ringbw"
	missing=$(run_on "$n" "ldd $REMOTE_DIR/ringbw 2>/dev/null | awk '/not found/ {print \$1}'" || true)
	if [[ -n $missing ]]; then
		if [[ $missing == *libnvrtc* && -d $CUDA_LIB ]]; then
			scp -q "${SSH_OPTS[@]}" "$CUDA_LIB"/libnvrtc.so.* "$CUDA_LIB"/libnvrtc-builtins.so.* "$(target "$n"):$REMOTE_DIR/"
			ENVPFX[$n]="LD_LIBRARY_PATH=$REMOTE_DIR "
			echo "  $n: $REMOTE_DIR (plus NVRTC libraries)"
		else
			echo "  $n: missing libraries: $missing" >&2; exit 1
		fi
	else
		echo "  $n: $REMOTE_DIR"
	fi
done

# 7. run everything at once: one process per (node, cable index)
LOG=$(mktemp -d "${TMPDIR:-/tmp}/ring-test.XXXXXX")
echo "== running $ITERS x ${SIZE} MiB on every cable (logs in $LOG) =="
pids=(); names=()
for ((j = 0; j < MAXJ; j++)); do
	for n in "${ORDER[@]}"; do
		l=${LISTEN_ON["$n|$j"]:-}; s=${SEND_TO["$n|$j"]:-}
		[[ -z $l && -z $s ]] && continue
		bin=$BIN; is_local "$n" || bin=$REMOTE_DIR/ringbw
		cmd="${ENVPFX[$n]:-}$bin ${l:+--listen $l} ${s:+--connect $s} -s $SIZE -n $ITERS -p $((18516 + j)) $CPU"
		run_on "$n" "$cmd" >"$LOG/$n-cable$j.log" 2>&1 &
		pids+=($!); names+=("$n cable $j")
	done
	sleep 2	# listeners up before the next wave of senders
done
rc=0
for i in "${!pids[@]}"; do wait "${pids[$i]}" || { rc=1; echo "  FAILED: ${names[$i]}"; }; done

echo
for f in "$LOG"/*.log; do
	echo "---- $(basename "$f" .log) ----"
	grep -E "GPU:|summary|done:|MISMATCH|error|failed|rdma_|expected|not found" "$f" || tail -n 5 "$f"
done
((rc == 0)) && echo "RING TEST PASSED: $N nodes, $((MAXJ * N)) cable-directions" || echo "RING TEST FAILED (see $LOG)"
exit $rc
