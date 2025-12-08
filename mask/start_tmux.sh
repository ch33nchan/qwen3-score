#!/bin/bash
# Start mask pipeline in tmux session on charizard
# Usage: ./mask/start_tmux.sh <project_id>

set -e

PROJECT_ID=${1:-""}
API_KEY="${LABEL_STUDIO_API_KEY:-c400214e3fa07bc2da35217ff6e42cc3e33c839f}"
LABEL_STUDIO_URL="${LABEL_STUDIO_URL:-http://localhost:8080}"

# Available GPUs (skip cuda:1 which is in use)
DEVICES="cuda:0,cuda:2,cuda:3"

# Session name
SESSION_NAME="mask-pipeline"

if [ -z "$PROJECT_ID" ]; then
    echo "Usage: ./mask/start_tmux.sh <project_id>"
    echo ""
    echo "Environment variables:"
    echo "  LABEL_STUDIO_API_KEY  - API key (default: provided)"
    echo "  LABEL_STUDIO_URL      - Label Studio URL (default: http://localhost:8080)"
    echo ""
    exit 1
fi

# Check if session exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists. Attaching..."
    tmux attach-session -t "$SESSION_NAME"
    exit 0
fi

# Create new session
echo "Creating tmux session: $SESSION_NAME"
echo "Project ID: $PROJECT_ID"
echo "Devices: $DEVICES"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Create tmux session with 3 windows
tmux new-session -d -s "$SESSION_NAME" -n "main" -c "$PROJECT_DIR"

# Window 0: Main pipeline
tmux send-keys -t "$SESSION_NAME:0" "cd $PROJECT_DIR" C-m
tmux send-keys -t "$SESSION_NAME:0" "source .venv/bin/activate 2>/dev/null || python3 -m venv .venv && source .venv/bin/activate" C-m
tmux send-keys -t "$SESSION_NAME:0" "pip install -q -r requirements.txt" C-m
tmux send-keys -t "$SESSION_NAME:0" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME:0" "echo '=== MASK PIPELINE READY ==='" C-m
tmux send-keys -t "$SESSION_NAME:0" "echo 'Run: python3 mask/run.py --project_id $PROJECT_ID --api_key \"$API_KEY\" --url \"$LABEL_STUDIO_URL\" --devices $DEVICES'" C-m
tmux send-keys -t "$SESSION_NAME:0" "echo ''" C-m

# Pre-fill the command (user can edit and run)
tmux send-keys -t "$SESSION_NAME:0" "python3 mask/run.py --project_id $PROJECT_ID --api_key \"$API_KEY\" --url \"$LABEL_STUDIO_URL\" --devices $DEVICES --debug"

# Window 1: GPU monitor
tmux new-window -t "$SESSION_NAME" -n "gpus"
tmux send-keys -t "$SESSION_NAME:1" "watch -n 2 nvidia-smi" C-m

# Window 2: Logs
tmux new-window -t "$SESSION_NAME" -n "logs" -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION_NAME:2" "cd $PROJECT_DIR/outputs/mask_results 2>/dev/null || mkdir -p $PROJECT_DIR/outputs/mask_results && cd $PROJECT_DIR/outputs/mask_results" C-m
tmux send-keys -t "$SESSION_NAME:2" "echo 'Watching for logs...'" C-m
tmux send-keys -t "$SESSION_NAME:2" "tail -f *.log 2>/dev/null || echo 'No logs yet. Will show when pipeline starts.'"

# Select main window
tmux select-window -t "$SESSION_NAME:0"

echo ""
echo "Tmux session '$SESSION_NAME' created with 3 windows:"
echo "  0: main  - Pipeline execution"
echo "  1: gpus  - GPU monitoring (nvidia-smi)"
echo "  2: logs  - Log file viewer"
echo ""
echo "Attaching to session..."
echo "Press Enter to run the pipeline command, or edit it first."
echo ""

tmux attach-session -t "$SESSION_NAME"
