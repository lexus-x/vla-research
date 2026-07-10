#!/bin/bash
# deploy_l4.sh — Upload scripts to L4 and start diagnosis experiments
set -e

L4_IP="44.234.88.211"
L4_USER="ubuntu"
INSTANCE_ID="i-0d86ee8a3dff3d6d1"
SSH_KEY="$HOME/.ssh/vla-temp"
REMOTE_DIR="/tmp/vla_agents"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)/scripts"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== VLA Agent Deployment to L4 ==="
echo "Target: $L4_USER@$L4_IP"
echo "Scripts: $SCRIPT_DIR"

# Re-send Instance Connect key
echo "[1/5] Refreshing Instance Connect key..."
aws ec2-instance-connect send-ssh-public-key \
    --region us-west-2 \
    --instance-id "$INSTANCE_ID" \
    --instance-os-user "$L4_USER" \
    --ssh-public-key file://${SSH_KEY}.pub 2>&1 | grep -o '"Success": true'

# Create remote directory
echo "[2/5] Creating remote directory..."
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$SSH_KEY" "$L4_USER@$L4_IP" \
    "mkdir -p $REMOTE_DIR/results $REMOTE_DIR/scripts" 2>&1

# Upload diagnosis script
echo "[3/5] Uploading diagnosis runner..."
scp -o StrictHostKeyChecking=no -i "$SSH_KEY" \
    "$SCRIPT_DIR/diagnosis_runner.py" \
    "$L4_USER@$L4_IP:$REMOTE_DIR/scripts/" 2>&1

# Upload repo analysis files for context
echo "[4/5] Uploading analysis context..."
scp -o StrictHostKeyChecking=no -i "$SSH_KEY" \
    "$REPO_ROOT/code/perturb_vla/perturb_vla.py" \
    "$L4_USER@$L4_IP:$REMOTE_DIR/scripts/" 2>&1

# Start diagnosis experiments in background
echo "[5/5] Starting diagnosis experiments on L4..."
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$SSH_KEY" "$L4_USER@$L4_IP" "
    cd $REMOTE_DIR
    export PATH=\$HOME/.local/bin:\$PATH
    export PYTHONPATH=$REMOTE_DIR/scripts:\$PYTHONPATH
    
    # Start diagnosis in background with nohup
    nohup python3 -u scripts/diagnosis_runner.py > $REMOTE_DIR/diagnosis.log 2>&1 &
    DIAG_PID=\$!
    echo \"Diagnosis PID: \$DIAG_PID\"
    
    # Start GPU monitor
    nohup bash -c 'while true; do
        echo \"\$(date +%H:%M:%S) \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)\" >> $REMOTE_DIR/gpu_monitor.log
        sleep 10
    done' > /dev/null 2>&1 &
    MON_PID=\$!
    echo \"GPU Monitor PID: \$MON_PID\"
    
    echo \"Deployment complete. Logs at $REMOTE_DIR/diagnosis.log\"
" 2>&1

echo ""
echo "=== Deployment Complete ==="
echo "Monitor: ssh -i $SSH_KEY $L4_USER@$L4_IP 'tail -f $REMOTE_DIR/diagnosis.log'"
echo "GPU:     ssh -i $SSH_KEY $L4_USER@$L4_IP 'tail -f $REMOTE_DIR/gpu_monitor.log'"
