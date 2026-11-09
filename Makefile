# SPDX-License-Identifier: GPL-2.0
# Out-of-tree build for the GB10 CPU-core DMA bridge.
#
#   make                 build gb10_dmab.ko against the running kernel
#   make KDIR=...        build against another kernel tree
#   make tools           build tools/dmabctl (adds CUDA support if found)
#   make sign            sign gb10_dmab.ko with the MOK key (Secure Boot)
#   make install         install module + udev/modprobe config, run depmod
#   make load / unload   insmod / rmmod convenience targets
#
KVER  ?= $(shell uname -r)
KDIR  ?= /lib/modules/$(KVER)/build
PWD   := $(shell pwd)
MODULE = gb10_dmab
CORES ?= 16-19

MOK_DIR ?= /var/lib/shim-signed/mok
MOK_KEY ?= $(MOK_DIR)/MOK.priv
MOK_CRT ?= $(MOK_DIR)/MOK.der

.PHONY: all modules tools clean sign install uninstall load unload

all: modules

modules:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

tools:
	$(MAKE) -C tools

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	$(MAKE) -C tools clean

sign: modules
	@test -r $(MOK_KEY) || { echo "No MOK key at $(MOK_KEY); run scripts/install.sh --mok-setup"; exit 1; }
	$(KDIR)/scripts/sign-file sha256 $(MOK_KEY) $(MOK_CRT) $(MODULE).ko
	modinfo -F sig_key $(MODULE).ko

install: modules
	install -D -m 0644 $(MODULE).ko /lib/modules/$(KVER)/extra/$(MODULE).ko
	install -D -m 0644 etc/99-gb10-dmab.rules /etc/udev/rules.d/99-gb10-dmab.rules
	install -D -m 0644 etc/gb10_dmab.conf /etc/modprobe.d/gb10_dmab.conf
	install -D -m 0644 etc/gb10_dmab-load.conf /etc/modules-load.d/gb10_dmab.conf
	depmod -a $(KVER)
	udevadm control --reload-rules || true

uninstall:
	rm -f /lib/modules/$(KVER)/extra/$(MODULE).ko
	rm -f /etc/udev/rules.d/99-gb10-dmab.rules /etc/modprobe.d/gb10_dmab.conf /etc/modules-load.d/gb10_dmab.conf
	depmod -a $(KVER)

load: modules
	insmod ./$(MODULE).ko cores=$(CORES)
	@ls -l /dev/$(MODULE); cat /sys/class/misc/$(MODULE)/engines

unload:
	rmmod $(MODULE)
