#!/bin/bash
# SSH helper with Instance Connect key refresh
# Usage: ./ssh_l4.sh "command"
set -e

KEY="$HOME/.ssh/vla-temp"
HOST="ubuntu@44.234.88.211"
IID="i-0d86ee8a3dff3d6d1"
REGION="us-west-2"

# Send IC key (valid 60s)
aws ec2-instance-connect send-ssh-public-key \
  --region "$REGION" --instance-id "$IID" \
  --instance-os-user ubuntu \
  --ssh-public-key file://${KEY}.pub > /dev/null 2>&1

# SSH immediately
ssh -o StrictHostKeyChecking=no \
    -o ConnectTimeout=8 \
    -o ServerAliveInterval=5 \
    -o ServerAliveCountMax=2 \
    -i "$KEY" "$HOST" "$@"
