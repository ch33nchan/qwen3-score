# Using Zellij with Inpaint EACPS

## Quick Start

The `start_zellij.sh` script runs the pipeline in the background and optionally creates a zellij session for monitoring.

## Run with Zellij

```bash
cd /mnt/data1/srini/qwen3-score
source .venv/bin/activate

./inpaint_eacps/start_zellij.sh \
  --task_id 71650389 71678832 71680285 71634650 71630498 71656881 71673477 \
  --gemini_api_key YOUR_KEY \
  --device cuda:0
```

When prompted, choose:
- `y` to create and attach to zellij session for monitoring
- `N` to run in background only (monitor via log file)

## Monitor Progress

### Option 1: Attach to Zellij Session
```bash
.venv/bin/zellij attach inpaint_eacps
```

### Option 2: View Log File
```bash
tail -f inpaint_eacps_inpaint_eacps.log
```

### Option 3: Check Outputs
```bash
# Count completed tasks
ls -d inpaint_eacps/task_*/ 2>/dev/null | wc -l

# View specific task
ls -lh inpaint_eacps/task_71680285/
```

## Direct Run (No Zellij)

If you prefer to run directly without zellij:

```bash
cd /mnt/data1/srini/qwen3-score
source .venv/bin/activate

python3 inpaint_eacps/run.py \
  --task_id 71650389 71678832 71680285 71634650 71630498 71656881 71673477 \
  --from_file project-label.json \
  --output_dir inpaint_eacps \
  --device cuda:0 \
  --gemini_api_key YOUR_KEY
```

This will show progress bars directly in your terminal.

## Zellij Session Management

```bash
# List sessions
.venv/bin/zellij list-sessions

# Attach to session
.venv/bin/zellij attach inpaint_eacps

# Kill session
.venv/bin/zellij kill-session inpaint_eacps

# Kill all sessions
.venv/bin/zellij kill-all-sessions
```

## Troubleshooting

### Zellij Not Installed
```bash
# Install zellij to .venv/bin
./inpaint_eacps/install_zellij.sh
```

### Session Already Exists
The script will prompt you to either attach or recreate the session.

### Terminal Issues
If zellij has terminal issues, just run the pipeline directly (see "Direct Run" above) - you'll still see progress bars.
