#!/bin/bash
# Clean restart commands - copy/paste these one by one

# 1. Kill everything
tmux kill-session -t eacps_batch 2>/dev/null
pkill -f run_batch_dual
pkill -f run_multigpu_batch

# 2. Clean old outputs (optional)
# rm -rf outputs/multigpu_batch*

# 3. Start fresh tmux session
tmux new-session -d -s eacps_batch

# 4. Run the command in tmux
tmux send-keys -t eacps_batch "cd /mnt/data1/srini/qwen3-score" C-m
tmux send-keys -t eacps_batch "source .venv/bin/activate" C-m
tmux send-keys -t eacps_batch "python3 inpaint_eacps/run_multigpu_batch.py --task_ids 71650389 71678832 71680285 71634650 71630498 71656881 71673477 --gemini_api_key AIzaSyDg6uH8UcysiWVTLKMUgkkRCvAOOqWHyIc --gpus 1 2 3" C-m

# 5. Attach to watch
tmux attach -t eacps_batch
