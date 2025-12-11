#!/bin/bash
# Run batch dual output processing in tmux with progress tracking
# Usage: ./run_batch_tmux.sh [single|multi]
# single: Run on single GPU (default)
# multi: Run on multiple GPUs (auto-detect and distribute)

set -e

# Mode selection
MODE="${1:-single}"

# Task IDs
TASK_IDS="71650389 71678832 71680285 71634650 71630498 71656881 71673477"

# Session name
SESSION="eacps_batch"

# Check if tmux session exists
if tmux has-session -t $SESSION 2>/dev/null; then
    echo "Session '$SESSION' already exists. Attaching..."
    tmux attach-session -t $SESSION
    exit 0
fi

# Check for Gemini API key
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY not set"
    echo "Export it first: export GEMINI_API_KEY='your-key'"
    exit 1
fi

# Create new tmux session
echo "Creating tmux session: $SESSION"
tmux new-session -d -s $SESSION -n "batch"

# Setup environment in tmux
tmux send-keys -t $SESSION "cd /Users/srini/Desktop/qwen3-score" C-m
tmux send-keys -t $SESSION "export GEMINI_API_KEY='$GEMINI_API_KEY'" C-m

# Activate venv if exists
if [ -d "venv" ]; then
    tmux send-keys -t $SESSION "source venv/bin/activate" C-m
fi

# Show configuration
tmux send-keys -t $SESSION "echo '========================================'" C-m
tmux send-keys -t $SESSION "echo 'EACPS Batch Dual Output Processing'" C-m
tmux send-keys -t $SESSION "echo 'Mode: $MODE'" C-m
tmux send-keys -t $SESSION "echo '========================================'" C-m
tmux send-keys -t $SESSION "echo 'Tasks: 7'" C-m
tmux send-keys -t $SESSION "echo 'Runs per task: 2 (init hair + char hair)'" C-m
tmux send-keys -t $SESSION "echo 'Total Gemini calls: 280 (~\$1.05)'" C-m
tmux send-keys -t $SESSION "echo '========================================'" C-m
tmux send-keys -t $SESSION "echo ''" C-m
tmux send-keys -t $SESSION "sleep 2" C-m

# Run the batch script
if [ "$MODE" = "multi" ]; then
    CMD="python3 inpaint_eacps/run_multigpu_batch.py \
      --task_ids $TASK_IDS \
      --gemini_api_key \$GEMINI_API_KEY \
      --output_dir outputs/multigpu_batch_$(date +%Y%m%d_%H%M%S)"
else
    CMD="python3 inpaint_eacps/run_batch_dual.py \
      --task_ids $TASK_IDS \
      --gemini_api_key \$GEMINI_API_KEY \
      --output_dir outputs/dual_hair_batch_$(date +%Y%m%d_%H%M%S)"
fi

tmux send-keys -t $SESSION "$CMD" C-m

# Create split for monitoring
tmux split-window -t $SESSION -h
tmux send-keys -t $SESSION "cd /Users/srini/Desktop/qwen3-score" C-m
tmux send-keys -t $SESSION "watch -n 5 'ls -lh outputs/dual_hair_batch_*/task_*/result_*.png 2>/dev/null | wc -l | xargs echo \"Completed outputs:\"'" C-m

# Select main pane
tmux select-pane -t $SESSION:0.0

# Attach to session
echo ""
echo "Attaching to tmux session: $SESSION"
echo "Detach with: Ctrl+b, then d"
echo "Kill session: tmux kill-session -t $SESSION"
echo ""
sleep 1
tmux attach-session -t $SESSION
