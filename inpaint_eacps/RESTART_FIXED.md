# Fix and Restart Pipeline

## Step 1: Kill Current Process

```bash
# Find the process
ps aux | grep "inpaint_eacps/run.py" | grep -v grep

# Kill it (replace PID with actual process ID)
kill <PID>

# Or kill all Python processes running the pipeline
pkill -f "inpaint_eacps/run.py"
```

## Step 2: Pull Latest Changes

```bash
cd /mnt/data1/srini/qwen3-score
git pull
```

## Step 3: Verify Fix

```bash
# Check that tqdm import is in pipeline.py
grep "from tqdm import tqdm" inpaint_eacps/pipeline.py
```

## Step 4: Restart Pipeline

```bash
source .venv/bin/activate

python3 inpaint_eacps/run.py \
  --task_id 71650389 71678832 71680285 71634650 71630498 71656881 71673477 \
  --from_file project-label.json \
  --output_dir inpaint_eacps \
  --device cuda:0 \
  --gemini_api_key YOUR_NEW_API_KEY
```

## Quick One-Liner

```bash
cd /mnt/data1/srini/qwen3-score && \
pkill -f "inpaint_eacps/run.py" && \
git pull && \
source .venv/bin/activate && \
python3 inpaint_eacps/run.py \
  --task_id 71650389 71678832 71680285 71634650 71630498 71656881 71673477 \
  --from_file project-label.json \
  --output_dir inpaint_eacps \
  --device cuda:0 \
  --gemini_api_key YOUR_NEW_API_KEY
```
