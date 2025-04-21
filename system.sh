#!/usr/bin/env bash
# collect_system_info.sh
# Collect and display key info about a Linux machine.

set -euo pipefail

info() {
  printf "\e[1;34m%s\e[0m\n" "$1"
}

subinfo() {
  printf "  \e[1;32m%s\e[0m\n" "$1"
}

# Header
info "=== System Information Report ==="
date
echo

# OS and Kernel
info "Operating System"
subinfo "OS: $(. /etc/os-release && echo "$NAME $VERSION")"
subinfo "Kernel: $(uname -r)"
echo

# CPU
info "CPU"
if command -v lscpu &>/dev/null; then
  lscpu | sed -e 's/^/  /'
else
  subinfo "Model: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
  subinfo "Cores: $(grep -c '^processor' /proc/cpuinfo)"
fi
echo

# Memory
info "Memory"
subinfo "Total/Used/Free:"
free -h | sed -n '1,2p' | sed -e 's/^/  /'
echo

# Disk Usage
info "Disk Usage"
df -h --total | sed -e 's/^/  /'
echo

# GPU
info "GPU"
if command -v lspci &>/dev/null; then
  lspci | grep -i --color=never 'vga\|3d\|display' | sed -e 's/^/  /'
  if command -v nvidia-smi &>/dev/null; then
    subinfo "NVIDIA GPU details:"
    nvidia-smi | sed -e 's/^/    /'
  fi
else
  subinfo "lspci not found, cannot detect GPUs."
fi
echo

# Network
info "Network Interfaces"
if command -v ip &>/dev/null; then
  ip -brief addr show | sed -e 's/^/  /'
else
  subinfo "ip command not found."
fi
echo

# Top Processes by Memory and CPU
info "Top Processes"
subinfo "By CPU usage:"
ps -eo pid,comm,pcpu --sort=-pcpu | head -n 6 | sed -e 's/^/  /'
subinfo "By Memory usage:"
ps -eo pid,comm,pmem --sort=-pmem | head -n 6 | sed -e 's/^/  /'
echo

# Uptime and Load
info "Uptime and Load"
subinfo "$(uptime) "
echo

info "=== End of Report ==="
