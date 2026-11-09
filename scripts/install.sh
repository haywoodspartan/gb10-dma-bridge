#!/usr/bin/env bash
# Build, sign (Secure Boot) and install gb10_dmab on this machine.
#
#   install.sh --mok-setup     one-time: create a Machine Owner Key and enroll it (reboot)
#   install.sh                 build, sign if Secure Boot is on, install, load
#   install.sh --uninstall     unload and remove module + config
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MOK_DIR=/var/lib/shim-signed/mok
MOK_KEY=$MOK_DIR/MOK.priv
MOK_CRT=$MOK_DIR/MOK.der

sb_enabled() { mokutil --sb-state 2>/dev/null | grep -q "SecureBoot enabled"; }

mok_setup() {
	[[ $EUID -eq 0 ]] || { echo "--mok-setup needs root" >&2; exit 1; }
	if [[ -r $MOK_KEY && -r $MOK_CRT ]]; then
		echo "MOK key already present at $MOK_DIR"
	else
		mkdir -p "$MOK_DIR"
		openssl req -new -x509 -newkey rsa:2048 -nodes -days 36500 \
			-keyout "$MOK_KEY" -outform DER -out "$MOK_CRT" \
			-subj "/CN=gb10-dma-bridge module signing/"
		chmod 600 "$MOK_KEY"
		echo "Created $MOK_KEY / $MOK_CRT"
	fi
	if mokutil --list-enrolled 2>/dev/null | grep -q "gb10-dma-bridge module signing"; then
		echo "Key is already enrolled."
		return
	fi
	echo
	echo "Enrolling the key with shim. Choose a one-time password; on the next boot"
	echo "the blue MokManager screen appears: Enroll MOK -> Continue -> Yes -> enter"
	echo "that password -> Reboot."
	mokutil --import "$MOK_CRT"
	echo "Now reboot, complete the enrollment, and run $0 again."
}

install_all() {
	[[ $EUID -eq 0 ]] || { echo "install needs root" >&2; exit 1; }
	make -C "$ROOT" modules
	if sb_enabled; then
		if [[ -r $MOK_KEY && -r $MOK_CRT ]]; then
			make -C "$ROOT" sign
		else
			echo "Secure Boot is enabled and no MOK key exists; run: $0 --mok-setup" >&2
			exit 1
		fi
	fi
	make -C "$ROOT" install
	make -C "$ROOT" tools || echo "tools build failed (non-fatal)"
	modprobe -r gb10_dmab 2>/dev/null || true
	modprobe gb10_dmab
	dmesg | grep gb10_dmab | tail -n 5
	ls -l /dev/gb10_dmab
	cat /sys/class/misc/gb10_dmab/engines
	"$ROOT/scripts/isolate-cores.sh" --status || true
}

uninstall_all() {
	[[ $EUID -eq 0 ]] || { echo "uninstall needs root" >&2; exit 1; }
	modprobe -r gb10_dmab 2>/dev/null || true
	make -C "$ROOT" uninstall
}

case ${1:-} in
--mok-setup) mok_setup ;;
--uninstall) uninstall_all ;;
"") install_all ;;
*) sed -n '2,7p' "$0"; exit 2 ;;
esac
