#!/usr/bin/env bash
# Dedicate CPUs to the gb10_dmab copy engines at the kernel level.
#
#   isolate-cores.sh --status             current isolation vs. the module's cores
#   isolate-cores.sh --apply [-c 16-19]   GRUB drop-in + modprobe option, then update-grub
#   isolate-cores.sh --remove             drop the GRUB drop-in, then update-grub
#
# --apply adds to the kernel command line (takes effect after a reboot):
#   isolcpus=domain,managed_irq,<cores>   no scheduler load balancing, no managed IRQs
#   nohz_full=<cores>                     stop the tick while one task runs
#   rcu_nocbs=<cores>                     offload RCU callbacks to housekeeping CPUs
#   irqaffinity=<all other cpus>          default IRQ affinity away from the engines
set -euo pipefail

CORES="${CORES:-16-19}"
DROPIN=/etc/default/grub.d/90-gb10-dmab.cfg
MODCONF=/etc/modprobe.d/gb10_dmab.conf
SYSCPU=/sys/devices/system/cpu

expand() {	# cpulist -> newline-separated sorted cpu numbers
	local part a b i
	local -a parts
	IFS=, read -ra parts <<<"$1"
	for part in "${parts[@]}"; do
		[[ -z $part ]] && continue
		if [[ $part == *-* ]]; then
			a=${part%-*}; b=${part#*-}
			for ((i = a; i <= b; i++)); do echo "$i"; done
		else
			echo "$part"
		fi
	done | sort -n | uniq
}

compress() {	# newline/space-separated sorted cpus -> cpulist
	local prev=-2 start=-1 out="" c
	for c in $1; do
		if ((c != prev + 1)); then
			if ((start >= 0)); then
				((start == prev)) && out+="$start," || out+="$start-$prev,"
			fi
			start=$c
		fi
		prev=$c
	done
	if ((start >= 0)); then
		((start == prev)) && out+="$start" || out+="$start-$prev"
	fi
	echo "$out"
}

complement() {	# cpus not in $1, out of the online set (comm needs lexical order)
	comm -23 <(expand "$(cat $SYSCPU/online)" | sort) <(expand "$1" | sort) | sort -n
}

core_types() {	# "cpuN Cortex-X925" per cpu in $1 (MIDR part 0xd85 = X925, 0xd87 = A725)
	local c m part t
	for c in $(expand "$1"); do
		m=$(cat $SYSCPU/cpu$c/regs/identification/midr_el1 2>/dev/null || echo 0)
		part=$(( (m >> 4) & 0xfff ))
		case $part in
		3461) t=Cortex-X925 ;;
		3463) t=Cortex-A725 ;;
		*) t=$(printf "part-0x%x" $part) ;;
		esac
		echo "cpu$c $t"
	done
}

status() {
	local isolated nohz want missing=() c
	isolated=$(cat $SYSCPU/isolated)
	nohz=$(cat $SYSCPU/nohz_full 2>/dev/null || echo "")
	echo "online cpus      : $(cat $SYSCPU/online)"
	echo "isolcpus         : ${isolated:-(none)}"
	echo "nohz_full        : ${nohz:-(none)}"
	echo "cmdline          : $(grep -oE '(isolcpus|nohz_full|rcu_nocbs|irqaffinity)=[^ ]*' /proc/cmdline | tr '\n' ' ')"
	if [[ -r /sys/class/misc/gb10_dmab/cores ]]; then
		want=$(cat /sys/class/misc/gb10_dmab/cores)
		echo "gb10_dmab cores  : $want  ($(cat /sys/class/misc/gb10_dmab/isolation))"
	else
		want=$CORES
		echo "gb10_dmab cores  : $want  (module not loaded; from -c/CORES)"
	fi
	echo "core types       : $(core_types "$want" | awk '{print $2}' | sort | uniq -c | awk '{printf "%s x %s ", $1, $2}')"
	for c in $(expand "$want"); do
		expand "$isolated" | grep -qx "$c" || missing+=("$c")
	done
	if ((${#missing[@]})); then
		echo "RESULT           : NOT isolated: $(compress "${missing[*]}") still in the scheduler domain"
		echo "                   run: sudo $0 --apply -c $want   (then reboot)"
		return 1
	fi
	echo "RESULT           : OK, every engine core is isolated"
}

apply() {
	local hk
	[[ $EUID -eq 0 ]] || { echo "--apply needs root" >&2; exit 1; }
	expand "$CORES" >/dev/null
	echo "isolating: $(core_types "$CORES" | tr '\n' ' ')"
	if (( $(core_types "$CORES" | awk '{print $2}' | sort -u | wc -l) > 1 )); then
		echo "warning: $CORES mixes core types" >&2
	fi
	hk=$(compress "$(complement "$CORES")")
	[[ -n $hk ]] || { echo "refusing to isolate every CPU" >&2; exit 1; }
	command -v update-grub >/dev/null || { echo "update-grub not found; this script targets Ubuntu/GRUB" >&2; exit 1; }

	mkdir -p "$(dirname "$DROPIN")"
	cat >"$DROPIN" <<CFG
# Written by gb10-dma-bridge/scripts/isolate-cores.sh. Remove with --remove.
GRUB_CMDLINE_LINUX_DEFAULT="\$GRUB_CMDLINE_LINUX_DEFAULT isolcpus=domain,managed_irq,$CORES nohz_full=$CORES rcu_nocbs=$CORES irqaffinity=$hk"
CFG
	if [[ -f $MODCONF ]]; then
		sed -i -E "s/(^options gb10_dmab .*)cores=[^ ]*/\1cores=$CORES/" "$MODCONF"
	else
		echo "options gb10_dmab cores=$CORES" >"$MODCONF"
	fi
	update-grub
	echo
	echo "Isolation for CPUs $CORES written to $DROPIN (housekeeping: $hk)."
	echo "Reboot to activate, then verify with: $0 --status"
}

remove() {
	[[ $EUID -eq 0 ]] || { echo "--remove needs root" >&2; exit 1; }
	rm -f "$DROPIN"
	update-grub
	echo "Removed $DROPIN. Reboot to drop the isolation parameters."
}

ACTION=status
while (($#)); do
	case $1 in
	--status) ACTION=status ;;
	--apply) ACTION=apply ;;
	--remove) ACTION=remove ;;
	-c | --cores) CORES=$2; shift ;;
	-h | --help) sed -n '2,14p' "$0"; exit 0 ;;
	*) echo "unknown option $1" >&2; exit 2 ;;
	esac
	shift
done
$ACTION
