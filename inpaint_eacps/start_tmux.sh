#!/bin/bash
set -e

TASK_ID=${1:-""}
GEMINI_API_KEY="${GEMINI_API_KEY:-}"
DEVICE="${DEVICE:-cuda:0}"
SESSION_NAME="inpaint-eacps"

if [ -z "$TASK_ID" ]; then
    echo "Usage: $0 <task_id> [gemini_api_key]"
    echo "Example: $0 71651078 'your_gemini_api_key'"
    exit 1
fi

if [ -z "$GEMINI_API_KEY" ] && [ -n "$2" ]; then
    GEMINI_API_KEY="$2"
fi

if [ -z "$GEMINI_API_KEY" ]; then
    echo "Warning: GEMINI_API_KEY not set. Scoring will use defaults."
fi

# Check if session exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session $SESSION_NAME already exists. Attaching..."
    tmux attach -t "$SESSION_NAME"
    exit 0
fi

# Create new session
tmux new-session -d -s "$SESSION_NAME"

# Window 0: Main pipeline
tmux rename-window -t "$SESSION_NAME:0" "pipeline"
tmux send-keys -t "$SESSION_NAME:0" "cd /mnt/data1/srini/qwen3-score && git pull && source .venv/bin/activate && python3 inpaint_eacps/run.py --task_id $TASK_ID --gemini_api_key '$GEMINI_API_KEY' --device $DEVICE 2>&1 | tee outputs/inpaint_eacps/run_\$(date +%Y%m%d_%H%M%S).log" C-m

# Window 1: GPU monitoring
tmux new-window -t "$SESSION_NAME:1" -n "gpus"
tmux send-keys -t "$SESSION_NAME:1" "watch -n 1 nvidia-smi" C-m

# Window 2: Logs
tmux new-window -t "$SESSION_NAME:2" -n "logs"
tmux send-keys -t "$SESSION_NAME:2" "tail -f outputs/inpaint_eacps/*.log" C-m

# Select main window
tmux select-window -t "$SESSION_NAME:0"

# Attach to session
tmux attach -t "$SESSION_NAME"
