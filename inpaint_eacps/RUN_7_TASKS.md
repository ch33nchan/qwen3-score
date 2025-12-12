# Run 7 Label Studio Tasks with Zellij

## Task IDs
```
71650389 71678832 71680285 71634650 71630498 71656881 71673477
```

## Step 1: Install Zellij (Local in .venv)

```bash
# Navigate to project
cd /path/to/qwen3-score

# Run installation script (installs to .venv/bin/zellij)
./inpaint_eacps/install_zellij.sh
```

This will:
- Download zellij binary for your architecture
- Install it to `.venv/bin/zellij`
- Make it executable

### Verify Installation
```bash
.venv/bin/zellij --version
# Should show: zellij 0.x.x
```

**Note:** The `start_zellij.sh` script will automatically install zellij if it's not found, so you can skip this step if you prefer.

## Step 2: Setup Project Environment

```bash
# Navigate to project
cd /path/to/qwen3-score

# Pull latest changes
git pull

# Activate virtual environment
source .venv/bin/activate

# Install/update dependencies (if needed)
pip install -r requirements.txt

# Verify project-label.json exists
ls -lh project-label.json
```

## Step 3: Set Gemini API Key

```bash
# Option 1: Export as environment variable
export GEMINI_API_KEY="your_key_here"

# Option 2: Pass directly in command (see Step 4)
```

## Step 4: Run All 7 Tasks

### Command with Environment Variable
```bash
./inpaint_eacps/start_zellij.sh \
  --task_id 71650389 71678832 71680285 71634650 71630498 71656881 71673477 \
  --device cuda:0
```

### Command with API Key in Command
```bash
./inpaint_eacps/start_zellij.sh \
  --task_id 71650389 71678832 71680285 71634650 71630498 71656881 71673477 \
  --gemini_api_key "your_key_here" \
  --device cuda:0
```

### Full Command with All Options
```bash
./inpaint_eacps/start_zellij.sh \
  --task_id 71650389 71678832 71680285 71634650 71630498 71656881 71673477 \
  --from_file project-label.json \
  --output_dir inpaint_eacps \
  --device cuda:0 \
  --gemini_api_key "your_key_here" \
  --k_global 8 \
  --m_global 3 \
  --k_local 4
```

## Step 5: Monitor Progress

### Attach to Session
```bash
# Using zellij from .venv/bin
.venv/bin/zellij attach inpaint_eacps

# Or if .venv/bin is in PATH
zellij attach inpaint_eacps
```

### Check Progress (in another terminal)
```bash
# Count completed tasks
ls -d inpaint_eacps/task_*/ 2>/dev/null | wc -l

# List all task directories
ls -lh inpaint_eacps/task_*/

# Check specific task
ls -lh inpaint_eacps/task_71680285/

# Watch for new results
watch -n 5 'ls -lh inpaint_eacps/task_*/result.png 2>/dev/null | wc -l'
```

### View Logs
```bash
# Attach to session to see live progress
zellij attach inpaint_eacps
```

## Expected Output

All outputs will be in:
```
inpaint_eacps/
├── task_71650389/
│   ├── result.png
│   ├── init.png
│   ├── mask.png
│   ├── character.png
│   ├── faceswap_base.png
│   ├── candidate_*.png
│   └── metrics.json
├── task_71678832/
├── task_71680285/
├── task_71634650/
├── task_71630498/
├── task_71656881/
└── task_71673477/
```

## Quick One-Liner (Copy-Paste Ready)

```bash
# Make sure you're in project root and venv is activated
cd /path/to/qwen3-score && source .venv/bin/activate && \
./inpaint_eacps/start_zellij.sh \
  --task_id 71650389 71678832 71680285 71634650 71630498 71656881 71673477 \
  --gemini_api_key "$GEMINI_API_KEY" \
  --device cuda:0
```

## Troubleshooting

### Zellij Not Found
```bash
# Check if in PATH
which zellij

# If not, add to PATH
export PATH="$PATH:/usr/local/bin"
```

### Script Permission Denied
```bash
chmod +x inpaint_eacps/start_zellij.sh
```

### Session Already Exists
```bash
# Kill existing session
zellij kill-session inpaint_eacps

# Then run again
```

### Task Not Found
```bash
# Verify tasks exist in project-label.json
grep -E "71650389|71678832|71680285|71634650|71630498|71656881|71673477" project-label.json
```

## Session Management

```bash
# List all sessions
.venv/bin/zellij list-sessions

# Attach to session
.venv/bin/zellij attach inpaint_eacps

# Kill session
.venv/bin/zellij kill-session inpaint_eacps

# Kill all sessions
.venv/bin/zellij kill-all-sessions
```

**Tip:** Add `.venv/bin` to PATH for convenience:
```bash
export PATH="$PATH:$(pwd)/.venv/bin"
# Then you can use: zellij attach inpaint_eacps
```
