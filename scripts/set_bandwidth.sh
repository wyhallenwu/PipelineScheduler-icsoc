#!/bin/bash

# Function to check if a command exists
command_exists() {
    type "$1" &> /dev/null
}

# Check if script is run as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root"
    exit 1
fi

# Check if TC is installed
if ! command_exists tc; then
    echo "TC (Traffic Control) is not installed. Please install it and try again."
    exit 1
fi

# Function to list available network interfaces
list_interfaces() {
    echo "Available network interfaces:"
    ip -o link show | awk -F': ' '{print $2}'
}

# List available interfaces
list_interfaces

# Prompt for network interface
read -p "Enter the network interface name: " INTERFACE

# Check if the interface exists
if ! ip link show "$INTERFACE" &> /dev/null; then
    echo "Interface $INTERFACE does not exist."
    exit 1
fi

# Prompt for bandwidth limit
read -p "Enter the desired bandwidth limit in Mbps: " BANDWIDTH_MBPS

# Convert Mbps to kbps
BANDWIDTH_KBPS=$((BANDWIDTH_MBPS * 1024))

echo "Removing existing qdisc..."
tc qdisc del dev "$INTERFACE" root 2> /dev/null
echo "Existing qdisc removed."

echo "Adding new qdisc with bandwidth limit of ${BANDWIDTH_MBPS} Mbps..."
TC_OUTPUT=$(sudo tc qdisc add dev "$INTERFACE" root tbf rate "${BANDWIDTH_KBPS}kbit" burst 1024kbit latency 50ms 2>&1)
TC_STATUS=$?

if [ $TC_STATUS -eq 0 ]; then
    echo "Bandwidth successfully limited to $BANDWIDTH_MBPS Mbps on interface $INTERFACE"
else
    echo "Failed to set bandwidth limit. Error output:"
    echo "$TC_OUTPUT"
    exit 1
fi

echo "Current qdisc settings:"
tc qdisc show dev "$INTERFACE"

echo "To remove this bandwidth limit, run: sudo tc qdisc del dev $INTERFACE root"

echo "Script completed. You can now test the bandwidth limit using iperf3 in another terminal."