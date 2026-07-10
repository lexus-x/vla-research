#!/bin/bash
# Quick deploy and start diagnosis on L4
set -e

KEY="$HOME/.ssh/vla-temp"
HOST="ubuntu@44.234.88.211"
IID="i-0d86ee8a3dff3d6d1"

# Send IC key
aws ec2-instance-connect send-ssh-public-key \
  --region us-west-2 --instance-id "$IID" \
  --instance-os-user ubuntu \
  --ssh-public-key file://${KEY}.pub > /dev/null 2>&1

# All in one SSH session
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
    -o ServerAliveInterval=5 -o ServerAliveCountMax=2 \
    -i "$KEY" "$HOST" bash <<'REMOTE'
set -e
export PATH=$HOME/.local/bin:$PATH

# Kill old
pkill -f diagnosis_runner 2>/dev/null || true
sleep 1

# Clear cache
rm -rf ~/.cache/huggingface/modules/transformers_modules/openvla/

# Start
cd /tmp/vla_agents
nohup python3 -u scripts/diagnosis_runner.py > diagnosis.log 2>&1 &
echo "STARTED PID=$!"

# Wait a bit and show initial output
sleep 8
echo "=== INITIAL LOG ==="
head -20 diagnosis.log
REMOTE
