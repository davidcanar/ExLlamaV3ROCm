#!/usr/bin/env bash
# Unload everything that depends on the thunderbolt core, then load the
# patched core from updates/ and the odinlink driver. Handles the Fedora
# detail that `typec` depends on `thunderbolt`, so rmmod thunderbolt alone
# silently fails and modprobe becomes a no-op (leaving the OLD core running).
set -u
echo "before: $(lsmod | grep -E '^(thunderbolt|typec|odl_tb5)' | awk '{print $1"("$3")"}' | tr '\n' ' ')"
systemctl stop odinlink 2>/dev/null

# Unload in dependency order. typec_* altmodes first, then typec, then tbnet.
for m in odl_tb5 thunderbolt_net typec_displayport typec_nvidia typec_thunderbolt typec_ucsi typec; do
    rmmod "$m" 2>/dev/null && echo "  unloaded $m"
done

for d in $(grep -l 0x0c0340 /sys/bus/pci/devices/*/class 2>/dev/null | sed 's|/class$||;s|.*/||'); do
    [ -e "/sys/bus/pci/drivers/thunderbolt/$d" ] && echo "$d" > /sys/bus/pci/drivers/thunderbolt/unbind && echo "  unbound $d"
done
sleep 1

if lsmod | grep -q '^thunderbolt '; then
    rmmod thunderbolt || { echo "!! cannot unload thunderbolt; still held by:"; lsmod | awk '$4 ~ /thunderbolt/ {print "   "$1}'; exit 1; }
fi
echo "after-unload: $(lsmod | grep -cE '^(thunderbolt|typec|odl_tb5)') tb-family modules"

modprobe thunderbolt || { echo "!! modprobe thunderbolt failed"; exit 1; }
sleep 1
echo "core: $(modinfo -F filename thunderbolt)"
echo "xd-api symbols: $(grep -c tb_register_protocol_handler_xd /proc/kallsyms)"
grep -q tb_register_protocol_handler_xd /proc/kallsyms || { echo "!! running core is not the xd-API build"; exit 1; }

insmod /usr/local/lib/odinlink/odl_tb5.ko odl_ring_size=1024 e2e=1 bind_any=0 || exit 1
echo "odl loaded: bind_any=$(cat /sys/module/odl_tb5/parameters/bind_any)"
dmesg | grep -E "odl_tb5: (source|OdinLink TB5)" | tail -2
