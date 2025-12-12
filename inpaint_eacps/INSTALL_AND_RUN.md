# Installation and Run Commands

## Step 1: Install Zellij

### On Linux/macOS (using cargo)
```bash
cargo install --locked zellij
```

### On Linux (using package manager)
```bash
# Ubuntu/Debian
curl -L https://github.com/zellij-org/zellij/releases/latest/download/zellij-x86_64-unknown-linux-musl.tar.gz | tar -xz
sudo mv zellij /usr/local/bin/

# Or using package manager
# For Ubuntu 22.04+
sudo apt update
sudo apt install zellij
```

### On macOS (using Homebrew)
```bash
brew install zellij
```

### Verify Installation
```bash
zellij --version
```

## Step 2: Setup Python Environment

```bash
# Navigate to project directory
cd /path/to/qwen3-score

# Create virtual environment (if not exists)
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or if using uv
uv pip install -r requirements.txt
```

## Step 3: Verify project-label.json

```bash
# Make sure project-label.json exists in project root
ls -lh project-label.json
```

## Step 4: Set Gemini API Key (Optional)

```bash
# Option 1: Set environment variable
export GEMINI_API_KEY="your_key_here"

# Option 2: Pass via command line (see Step 5)
```

## Step 5: Run All 7 Tasks

### With Environment Variable
```bash
./inpaint_eacps/start_zellij.sh \
  --task_id 71650389 71678832 71680285 71634650 71630498 71656881 71673477 \
  --device cuda:0
```

### With API Key in Command
```bash
./inpaint_eacps/start_zellij.sh \
  --task_id 71650389 71678832 71680285 71634650 71630498 71656881 71673477 \
  --gemini_api_key YOUR_KEY \
  --device cuda:0
```

### Full Command with All Options
```bash
./inpaint_eacps/start_zellij.sh \
  --task_id 71650389 71678832 71680285 71634650 71630498 71656881 71673477 \
  --from_file project-label.json \
  --output_dir inpaint_eacps \
  --device cuda:0 \
  --gemini_api_key YOUR_KEY \
  --k_global 8 \
  --m_global 3 \
  --k_local 4
```

## Step 6: Monitor Progress

### Attach to Session
```bash
zellij attach inpaint_eacps
```

### Check Outputs (in another terminal)
```bash
# Watch outputs being created
watch -n 5 'ls -lh inpaint_eacps/task_*/result.png 2>/dev/null | wc -l'

# Or check specific task
ls -lh inpaint_eacps/task_71680285/
```

### List All Sessions
```bash
zellij list-sessions
```

### Kill Session (if needed)
```bash
zellij kill-session inpaint_eacps
```

## Expected Output Structure

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
│   └── ...
├── task_71680285/
│   └── ...
├── task_71634650/
│   └── ...
├── task_71630498/
│   └── ...
├── task_71656881/
│   └── ...
└── task_71673477/
    └── ...
```

## Troubleshooting

### Zellij Not Found
```bash
# Add to PATH if installed manually
export PATH="$PATH:/usr/local/bin"
```

### Permission Denied
```bash
# Make script executable
chmod +x inpaint_eacps/start_zellij.sh
```

### Virtual Environment Not Found
```bash
# Create it
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Task Not Found in project-label.json
```bash
# Verify task IDs exist
grep -E "(71650389|71678832|71680285|71634650|71630498|71656881|71673477)" project-label.json
```

## Quick One-Liner (After Setup)

```bash
# If everything is set up, just run:
./inpaint_eacps/start_zellij.sh \
  --task_id 71650389 71678832 71680285 71634650 71630498 71656881 71673477 \
  --gemini_api_key "$GEMINI_API_KEY" \
  --device cuda:0
```
