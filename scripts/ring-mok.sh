#!/usr/bin/env bash
# Secure Boot module signing across the Spark ring: one Machine Owner Key for
# all nodes, per-node build/sign/install, key enrolment, loading.
#
#   scripts/ring-mok.sh keys   NODE...   create the MOK on the first node (if absent) and copy it to the others
#   scripts/ring-mok.sh build  NODE...   on every node: make clean modules tools, then sign + install (sudo)
#   scripts/ring-mok.sh enroll NODE...   INTERACTIVE: mokutil --import on every node (one-time password per node)
#   scripts/ring-mok.sh load   NODE...   after the enrolment reboot: modprobe gb10_dmab and show engines/isolation
#   scripts/ring-mok.sh status NODE...   enrolment state, pending imports, installed module signature, loaded?
#
# Options: -d DIR  project directory on every node (default: this checkout's path)
#          -u USER ssh user
# Requires passwordless sudo on every node (scripts/cluster-bootstrap.sh); the
# MokNew/MokList EFI variables are root-only, so even status needs it.
#
# The enrolment flow is: enroll (choose a one-time password) -> reboot the node ->
# blue MokManager screen: Enroll MOK -> Continue -> Yes -> that password -> Reboot
# -> load. MokManager needs a keyboard at the node's console (HDMI or serial).
set -uo pipefail

MOK_DIR=/var/lib/shim-signed/mok
CN="gb10-dma-bridge module signing"
USER_=""; DIR=""
CMD=${1:-}; shift || true
NODES=()
while (($#)); do
	case $1 in
	-d) DIR=$2; shift ;;
	-u) USER_=$2; shift ;;
	-*) echo "unknown option $1" >&2; exit 2 ;;
	*) NODES+=("$1") ;;
	esac
	shift
done
[[ -n $CMD && ${#NODES[@]} -gt 0 ]] || { sed -n '2,17p' "$0"; exit 2; }
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DIR=${DIR:-$ROOT}
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new)

mapfile -t LOCAL_IPS < <(ip -4 -o addr show | awk '{print $4}' | cut -d/ -f1)
is_local() { local n=${1#*@} ip; for ip in "${LOCAL_IPS[@]}"; do [[ $n == "$ip" ]] && return 0; done; [[ $n == "$(hostname)" || $n == "$(hostname -s)" || $n == localhost ]]; }
target() { local n=$1; [[ -n $USER_ && $n != *@* ]] && n="$USER_@$n"; echo "$n"; }
run_on() { local n=$1; shift; if is_local "$n"; then bash -c "$*"; else ssh "${SSH_OPTS[@]}" "$(target "$n")" "$*"; fi; }
run_on_stdin() { local n=$1; shift; if is_local "$n"; then bash -c "$*"; else ssh "${SSH_OPTS[@]}" "$(target "$n")" "$*"; fi; }
run_tty() { local n=$1; shift; if is_local "$n"; then bash -c "$*"; else ssh -t -o ConnectTimeout=8 "$(target "$n")" "$*"; fi; }

cmd_keys() {
	local first=${NODES[0]} n
	echo "== key on $first =="
	run_on "$first" "sudo -n mkdir -p $MOK_DIR && if sudo -n test -r $MOK_DIR/MOK.priv && sudo -n test -r $MOK_DIR/MOK.der; then echo '  existing MOK kept'; else sudo -n openssl req -new -x509 -newkey rsa:2048 -nodes -days 36500 -keyout $MOK_DIR/MOK.priv -outform DER -out $MOK_DIR/MOK.der -subj '/CN=$CN/' 2>/dev/null && sudo -n chmod 600 $MOK_DIR/MOK.priv && sudo -n chmod 644 $MOK_DIR/MOK.der && echo '  created MOK.priv / MOK.der'; fi; sudo -n openssl x509 -inform DER -in $MOK_DIR/MOK.der -noout -subject -fingerprint -sha256 | sed 's/^/  /'" || return 1
	for n in "${NODES[@]:1}"; do
		echo "== copy to $n =="
		run_on "$first" "sudo -n cat $MOK_DIR/MOK.priv" | run_on_stdin "$n" "sudo -n mkdir -p $MOK_DIR && sudo -n tee $MOK_DIR/MOK.priv >/dev/null && sudo -n chmod 600 $MOK_DIR/MOK.priv" || return 1
		run_on "$first" "sudo -n cat $MOK_DIR/MOK.der" | run_on_stdin "$n" "sudo -n tee $MOK_DIR/MOK.der >/dev/null && sudo -n chmod 644 $MOK_DIR/MOK.der" || return 1
		run_on "$n" "sudo -n openssl x509 -inform DER -in $MOK_DIR/MOK.der -noout -fingerprint -sha256 | sed 's/^/  /'"
	done
}

cmd_build() {
	local n rc=0
	for n in "${NODES[@]}"; do
		echo "== build on $n =="
		run_on "$n" "cd '$DIR' && echo \"  kernel \$(uname -r)\" && make clean >/dev/null 2>&1; make modules 2>&1 | grep -E 'error|warning: [^t]|LD \[M\]' | sed 's/^/  /'; test -f gb10_dmab.ko || { echo '  BUILD FAILED'; exit 1; }; make tools 2>&1 | grep -E 'error' | sed 's/^/  /'; sudo -n make sign 2>&1 | grep -vE '^make' | sed 's/^/  /'; sudo -n make install 2>&1 | grep -E 'error|install -D' | sed 's/^/  /'; echo \"  installed: \$(modinfo -F vermagic /lib/modules/\$(uname -r)/extra/gb10_dmab.ko | cut -d' ' -f1), signer: \$(modinfo -F signer /lib/modules/\$(uname -r)/extra/gb10_dmab.ko)\"" || { rc=1; echo "  FAILED on $n"; }
	done
	return $rc
}

cmd_enroll() {
	local n
	echo "Each node asks for a ONE-TIME password (twice). You will type it again at that node's"
	echo "MokManager screen on its next boot. Use the same password on every node to keep it simple."
	for n in "${NODES[@]}"; do
		echo; echo "== enroll on $n =="
		if run_on "$n" "sudo -n mokutil --list-enrolled 2>/dev/null | grep -q '$CN'"; then echo "  already enrolled"; continue; fi
		if run_on "$n" "sudo -n mokutil --list-new 2>/dev/null | grep -q '$CN'"; then echo "  import already pending; reboot and complete MokManager"; continue; fi
		run_tty "$n" "sudo mokutil --import $MOK_DIR/MOK.der" || echo "  FAILED on $n"
	done
	echo; echo "Now reboot each node and complete: Enroll MOK -> Continue -> Yes -> password -> Reboot."
	echo "Then: $0 load ${NODES[*]}"
}

cmd_load() {
	local n rc=0
	for n in "${NODES[@]}"; do
		echo "== load on $n =="
		run_on "$n" "if ! sudo -n mokutil --list-enrolled 2>/dev/null | grep -q '$CN'; then echo '  MOK not enrolled yet (reboot + MokManager first)'; exit 1; fi; sudo -n modprobe gb10_dmab && lsmod | grep -q '^gb10_dmab' && echo '  loaded' || { echo '  modprobe failed:'; sudo -n dmesg | grep -iE 'gb10_dmab|module verification' | tail -3 | sed 's/^/    /'; exit 1; }; sudo -n dmesg | grep gb10_dmab | tail -4 | sed 's/^/  /'; ls -l /dev/gb10_dmab | sed 's/^/  /'; cat /sys/class/misc/gb10_dmab/isolation | sed 's/^/  /'" || rc=1
	done
	return $rc
}

cmd_status() {
	local n
	for n in "${NODES[@]}"; do
		echo "== $n =="
		run_on "$n" "echo \"  secure boot: \$(mokutil --sb-state 2>/dev/null | head -1)\"; echo \"  MOK enrolled: \$(sudo -n mokutil --list-enrolled 2>/dev/null | grep -c '$CN')  pending import: \$(sudo -n mokutil --list-new 2>/dev/null | grep -c '$CN')\"; k=/lib/modules/\$(uname -r)/extra/gb10_dmab.ko; if [ -f \$k ]; then echo \"  installed module: \$(modinfo -F vermagic \$k | cut -d' ' -f1), signer: \$(modinfo -F signer \$k)\"; else echo '  installed module: none'; fi; echo \"  loaded: \$(lsmod | grep -q '^gb10_dmab' && echo yes || echo no)\""
	done
}

case $CMD in
keys) cmd_keys ;;
build) cmd_build ;;
enroll) cmd_enroll ;;
load) cmd_load ;;
status) cmd_status ;;
*) echo "unknown command $CMD" >&2; exit 2 ;;
esac
