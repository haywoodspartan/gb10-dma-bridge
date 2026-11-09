#!/usr/bin/env bash
# One-time bootstrap of a DGX Spark ring: passwordless sudo and a full SSH mesh.
#
#   scripts/cluster-bootstrap.sh [-u user] [-p prefix] [--no-sudo] [--sync-project] NODE NODE NODE...
#
# Run it from a node that can already SSH to the others (NODE = name from your
# ~/.ssh/config, hostname or IP; the local machine is detected). You type each
# node's sudo password once; afterwards everything below is passwordless.
#
#   1. sudo     /etc/sudoers.d/90-<user>-nopasswd with "<user> ALL=(ALL) NOPASSWD:ALL",
#               validated with visudo before it is installed (skip with --no-sudo)
#   2. keys     an ed25519 key on every node that has none; every node's public key
#               appended to every node's ~/.ssh/authorized_keys (deduplicated)
#   3. hosts    a managed "# >>> spark-ring" block at the top of ~/.ssh/config on
#               every node: Host <hostname> -> HostName <its address on a cable it
#               shares with that node>, plus known_hosts entries via ssh-keyscan
#   4. verify   from every node to every other: ssh hostname and sudo -n true
#   --sync-project   rsync this project directory to the same path on every node
#
# Security note: step 1 makes <user>'s SSH keys worth root on every node.
set -uo pipefail

USER_=$(id -un); PREFIX="10.100."; DO_SUDO=1; SYNC=0; NODES=()
while (($#)); do
	case $1 in
	-u) USER_=$2; shift ;;
	-p|--prefix) PREFIX=$2; shift ;;
	--no-sudo) DO_SUDO=0 ;;
	--sync-project) SYNC=1 ;;
	-h|--help) sed -n '2,20p' "$0"; exit 0 ;;
	-*) echo "unknown option $1" >&2; exit 2 ;;
	*) NODES+=("$1") ;;
	esac
	shift
done
((${#NODES[@]} >= 2)) || { echo "need at least two nodes" >&2; exit 2; }
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SSH_OPTS=(-o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new)

mapfile -t LOCAL_IPS < <(ip -4 -o addr show | awk '{print $4}' | cut -d/ -f1)
is_local() { local n=${1#*@} ip; for ip in "${LOCAL_IPS[@]}"; do [[ $n == "$ip" ]] && return 0; done; [[ $n == "$(hostname)" || $n == "$(hostname -s)" || $n == localhost ]]; }
target() { local n=$1; [[ $n != *@* ]] && n="$USER_@$n"; echo "$n"; }
run_on() { local n=$1; shift; if is_local "$n"; then bash -c "$*"; else ssh "${SSH_OPTS[@]}" -o BatchMode=yes "$(target "$n")" "$*"; fi; }
run_tty() { local n=$1; shift; if is_local "$n"; then bash -c "$*"; else ssh -t "${SSH_OPTS[@]}" "$(target "$n")" "$*"; fi; }
step() { echo; echo "==== $* ===="; }

# ---- 0. reachability, hostnames, ring addresses -------------------------------
step "0. nodes"
declare -A HOST ADDR SUBNET_NODES
for n in "${NODES[@]}"; do
	h=$(run_on "$n" "hostname") || { echo "cannot reach $n over ssh without a password; fix that first (ssh-copy-id $(target "$n"))" >&2; exit 1; }
	HOST[$n]=$h
	while read -r cidr; do
		ip=${cidr%/*}; subnet=${ip%.*}
		ADDR["$n|$subnet"]=$ip
		SUBNET_NODES[$subnet]="${SUBNET_NODES[$subnet]:-} $n"
	done < <(run_on "$n" "ip -4 -o addr show scope global | awk -v p='$PREFIX' 'index(\$4, p) == 1 {print \$4}'")
	printf "  %-14s hostname %-10s ring addresses:" "$n" "$h"
	for k in "${!ADDR[@]}"; do [[ $k == "$n|"* ]] && printf " %s" "${ADDR[$k]}"; done
	echo
done
# direct address of Y as seen from X: Y's address on a subnet both share
direct() {	# direct X Y
	local x=$1 y=$2 k s
	for k in "${!ADDR[@]}"; do
		[[ $k == "$x|"* ]] || continue
		s=${k#*|}
		[[ -n ${ADDR["$y|$s"]:-} ]] && { echo "${ADDR["$y|$s"]}"; return 0; }
	done
	return 1
}

# ---- 1. passwordless sudo -----------------------------------------------------
if ((DO_SUDO)); then
	step "1. passwordless sudo for $USER_ (you will be asked for each node's password)"
	for n in "${NODES[@]}"; do
		echo "-- $n (${HOST[$n]})"
		if run_on "$n" "sudo -n true 2>/dev/null"; then
			echo "   already passwordless"
			continue
		fi
		run_tty "$n" "sudo bash -c 'f=/etc/sudoers.d/90-$USER_-nopasswd; t=\$(mktemp); echo \"$USER_ ALL=(ALL) NOPASSWD:ALL\" > \$t && visudo -cf \$t >/dev/null && install -m 440 \$t \$f && rm -f \$t && echo \"   installed \$f\"'" \
			|| { echo "   FAILED on $n" >&2; exit 1; }
	done
fi

# ---- 2. keys ------------------------------------------------------------------
step "2. ssh keys"
declare -A PUB
for n in "${NODES[@]}"; do
	PUB[$n]=$(run_on "$n" "umask 077; mkdir -p ~/.ssh; [ -f ~/.ssh/id_ed25519 ] || ssh-keygen -q -t ed25519 -N '' -f ~/.ssh/id_ed25519 -C \$(id -un)@\$(hostname); cat ~/.ssh/id_ed25519.pub") \
		|| { echo "cannot read/create a key on $n" >&2; exit 1; }
	echo "  ${HOST[$n]}: ${PUB[$n]##* }"
done
for n in "${NODES[@]}"; do
	for m in "${NODES[@]}"; do
		key=${PUB[$m]}
		run_on "$n" "umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys; grep -qxF '$key' ~/.ssh/authorized_keys || echo '$key' >> ~/.ssh/authorized_keys" \
			|| { echo "cannot update authorized_keys on $n" >&2; exit 1; }
	done
	echo "  ${HOST[$n]}: authorized_keys holds all ${#NODES[@]} keys"
done

# ---- 3. ssh config block + known_hosts ----------------------------------------
step "3. ~/.ssh/config and known_hosts on every node"
for x in "${NODES[@]}"; do
	block="# >>> spark-ring (managed by gb10-dma-bridge/scripts/cluster-bootstrap.sh) >>>"$'\n'
	targets=()
	for y in "${NODES[@]}"; do
		[[ $y == "$x" ]] && continue
		ip=$(direct "$x" "$y") || { echo "  $x and $y share no ring subnet; skipping that pair" >&2; continue; }
		aliases="${HOST[$y]}"; [[ $y != "${HOST[$y]}" && $y != "$ip" ]] && aliases+=" $y"
		block+="Host $aliases"$'\n'"    HostName $ip"$'\n'"    User $USER_"$'\n'"    StrictHostKeyChecking accept-new"$'\n'
		targets+=("$ip")	# ssh-keyscan cannot resolve config aliases; ssh checks known_hosts by HostName
	done
	block+="# <<< spark-ring <<<"$'\n'
	b64=$(printf '%s' "$block" | base64 -w0)
	run_on "$x" "umask 077; mkdir -p ~/.ssh && touch ~/.ssh/config && echo '$b64' | base64 -d > ~/.ssh/.spark-ring.block && awk '/^# >>> spark-ring/{skip=1} !skip{print} /^# <<< spark-ring/{skip=0}' ~/.ssh/config > ~/.ssh/.config.rest && cat ~/.ssh/.spark-ring.block ~/.ssh/.config.rest > ~/.ssh/config && rm -f ~/.ssh/.spark-ring.block ~/.ssh/.config.rest && chmod 600 ~/.ssh/config || exit 1; for t in ${targets[*]}; do ssh-keygen -F \$t -f ~/.ssh/known_hosts >/dev/null 2>&1 || ssh-keyscan -T 5 \$t >> ~/.ssh/known_hosts 2>/dev/null || true; done; exit 0" \
		|| { echo "cannot write ssh config on $x" >&2; exit 1; }
	echo "  ${HOST[$x]}: config block for $(printf '%s ' "${targets[@]}" | sed 's/ $//')"
done

# ---- 4. optional project sync -------------------------------------------------
if ((SYNC)); then
	step "4. syncing $ROOT to every node"
	for n in "${NODES[@]}"; do
		is_local "$n" && continue
		run_on "$n" "mkdir -p '$ROOT'"
		rsync -a --delete --exclude '*.o' --exclude '*.cmd' --exclude '.claude' "$ROOT/" "$(target "$n"):$ROOT/" && echo "  ${HOST[$n]}: synced"
	done
fi

# ---- 5. verify matrix ---------------------------------------------------------
step "5. verify: from -> to : ssh hostname / sudo -n"
rc=0
for x in "${NODES[@]}"; do
	for y in "${NODES[@]}"; do
		[[ $y == "$x" ]] && continue
		res=$(run_on "$x" "ssh -o BatchMode=yes -o ConnectTimeout=5 ${HOST[$y]} 'echo \$(hostname) \$(sudo -n true 2>/dev/null && echo sudo-ok || echo sudo-NO)'" 2>/dev/null) || res="ssh FAILED"
		printf "  %-8s -> %-8s : %s\n" "${HOST[$x]}" "${HOST[$y]}" "$res"
		[[ $res == *sudo-ok* ]] || rc=1
	done
done
echo
((rc == 0)) && echo "cluster-bootstrap: every node reaches every other node with key-based ssh and passwordless sudo" \
	    || echo "cluster-bootstrap: some pairs are not fully passwordless yet (see above)"
exit $rc
