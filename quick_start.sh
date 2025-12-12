#!/bin/bash
# Quick start script for multi-GPU batch processing with tmux
# Usage: ./quick_start.sh YOUR_GEMINI_API_KEY

set -e

SESSION="eacps_batch"
TASK_IDS="71650389 71678832 71680285 71634650 71630498 71656881 71673477"

# Check if session exists
if tmux has-session -t $SESSION 2>/dev/null; then
    echo "Session '$SESSION' already exists. Attaching..."
    tmux attach-session -t $SESSION
    exit 0
fi

# Check API key argument
if [ -z "$1" ]; then
    echo "Error: Gemini API key required"
    echo "Usage: ./quick_start.sh YOUR_GEMINI_API_KEY"
    exit 1
fi

GEMINI_API_KEY="$1"

echo "Creating tmux session: $SESSION"
tmux new-session -d -s $SESSION -n "multigpu"

# Setup
tmux send-keys -t $SESSION "cd /mnt/data1/srini/qwen3-score" C-m
tmux send-keys -t $SESSION "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION "clear" C-m

# Show info
tmux send-keys -t $SESSION "echo '========================================'" C-m
tmux send-keys -t $SESSION "echo 'EACPS Multi-GPU Batch Processing'" C-m
tmux send-keys -t $SESSION "echo '========================================'" C-m
tmux send-keys -t $SESSION "echo 'Tasks: 7'" C-m
tmux send-keys -t $SESSION "echo 'GPUs: 1, 2, 3 (auto-distribute)'" C-m
tmux send-keys -t $SESSION "echo 'Expected time: ~15-20 min'" C-m
tmux send-keys -t $SESSION "echo 'Gemini calls: 280 (~\$1.05)'" C-m
tmux send-keys -t $SESSION "echo '========================================'" C-m
tmux send-keys -t $SESSION "echo ''" C-m

# Run
CMD="python3 inpaint_eacps/run_multigpu_batch.py \
  --task_ids $TASK_IDS \
  --gemini_api_key '$GEMINI_API_KEY' \
  --gpus 1 2 3"

tmux send-keys -t $SESSION "$CMD" C-m

# Create monitoring split
tmux split-window -t $SESSION -h -p 30
tmux send-keys -t $SESSION "cd /mnt/data1/srini/qwen3-score" C-m
tmux send-keys -t $SESSION "source .venv/bin/activate" C-m

# Monitor loop
MONITOR_CMD='while true; do 
  clear
  echo "=== GPU USAGE ==="
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | head -3
  echo ""
  echo "=== PROGRESS ==="
  COMPLETED=$(find outputs/multigpu_batch*/gpu*/task_*/result_*.png 2>/dev/null | wc -l)
  TOTAL=14
  echo "Outputs: $COMPLETED / $TOTAL"
  echo "Progress: $(echo "scale=1; $COMPLETED * 100 / $TOTAL" | bc)%"
  echo ""
  echo "=== LOGS (last update) ==="
  tail -n 3 outputs/multigpu_batch*/gpu*_log.txt 2>/dev/null | grep -E "VERSION|Task|✓" | tail -5
  sleep 10
done'

tmux send-keys -t $SESSION "$MONITOR_CMD" C-m

# Select main pane
tmux select-pane -t $SESSION:0.0

echo ""
echo "✓ Session created!"
echo ""
echo "Attaching to tmux session: $SESSION"
echo ""
echo "Controls:"
echo "  - Detach: Ctrl+b, then d"
echo "  - Reattach: tmux attach -t $SESSION"
echo "  - Kill: tmux kill-session -t $SESSION"
echo "  - Switch panes: Ctrl+b, then arrow keys"
echo ""
sleep 2
tmux attach-session -t $SESSION
