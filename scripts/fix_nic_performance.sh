#!/usr/bin/env bash
# fix_nic_performance.sh — Apply NIC and kernel tuning to eliminate periodic network stalls
# Target: Intel I219-LM (e1000e) on eno1
# Run as: sudo bash scripts/fix_nic_performance.sh
set -euo pipefail

NIC="eno1"

echo "=== NIC Performance Fix Script ==="
echo "Interface: $NIC"
echo ""

# 1. Disable Energy Efficient Ethernet (EEE)
echo "[1/5] Disabling EEE (Energy Efficient Ethernet)..."
ethtool --set-eee "$NIC" eee off
echo "      Done. Was: enabled/active → now: off"
echo ""

# 2. Increase ring buffers from 256 → 4096
echo "[2/5] Increasing ring buffers to 4096 (RX + TX)..."
ethtool -G "$NIC" rx 4096 tx 4096
echo "      Done. Was: 256/256 → now: 4096/4096"
echo ""

# 3. Disable s0ix (Modern Standby) on NIC
echo "[3/5] Disabling s0ix-enabled private flag..."
ethtool --set-priv-flags "$NIC" s0ix-enabled off
echo "      Done."
echo ""

# 4. Set CPU governor to performance (all cores)
echo "[4/5] Setting CPU frequency governor to 'performance' on all cores..."
if command -v cpupower &>/dev/null; then
    cpupower frequency-set -g performance
else
    for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo performance > "$gov"
    done
fi
echo "      Done. Was: powersave → now: performance"
echo ""

# 5. Increase kernel network buffer sizes and backlog
echo "[5/5] Tuning kernel network buffers..."
sysctl -w net.core.rmem_max=16777216
sysctl -w net.core.wmem_max=16777216
sysctl -w net.core.rmem_default=1048576
sysctl -w net.core.wmem_default=1048576
sysctl -w net.core.netdev_max_backlog=5000
sysctl -w net.ipv4.tcp_rmem="4096 131072 16777216"
sysctl -w net.ipv4.tcp_wmem="4096 65536 16777216"
echo "      Done."
echo ""

# Verify
echo "=== Verification ==="
echo ""
echo "EEE status:"
ethtool --show-eee "$NIC" | grep -E 'EEE status'
echo ""
echo "Ring buffers:"
ethtool -g "$NIC" 2>&1 | grep -A1 'Current hardware'
echo ""
echo "s0ix flag:"
ethtool --show-priv-flags "$NIC"
echo ""
echo "CPU governor (cpu0):"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo ""
echo "Key sysctl values:"
sysctl net.core.rmem_max net.core.wmem_max net.core.netdev_max_backlog
echo ""
echo "=== All fixes applied. These are runtime-only and will revert on reboot. ==="
echo "To make persistent, add sysctl lines to /etc/sysctl.d/99-nic-perf.conf"
echo "and ethtool commands to a NetworkManager dispatcher or systemd unit."
