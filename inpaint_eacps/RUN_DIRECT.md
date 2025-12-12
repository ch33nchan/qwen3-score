# Direct Run Commands (No Zellij)

## Simple Command

```bash
cd /mnt/data1/srini/qwen3-score
source .venv/bin/activate

python3 inpaint_eacps/run.py \
  --task_id 71650389 71678832 71680285 71634650 71630498 71656881 71673477 \
  --from_file project-label.json \
  --output_dir inpaint_eacps \
  --device cuda:0 \
  --gemini_api_key AIzaSyDg6uH8UcysiWVTLKMUgkkRCvAOOqWHyIc
```

## Run in Background with Logging

```bash
cd /mnt/data1/srini/qwen3-score
source .venv/bin/activate

nohup python3 inpaint_eacps/run.py \
  --task_id 71650389 71678832 71680285 71634650 71630498 71656881 71673477 \
  --from_file project-label.json \
  --output_dir inpaint_eacps \
  --device cuda:0 \
  --gemini_api_key AIzaSyDg6uH8UcysiWVTLKMUgkkRCvAOOqWHyIc \
  > inpaint_eacps.log 2>&1 &

# Check progress
tail -f inpaint_eacps.log
```

## Using the Wrapper Script (Direct)

```bash
cd /mnt/data1/srini/qwen3-score
source .venv/bin/activate

.venv/bin/run_inpaint_eacps.sh
```

## Monitor Progress

```bash
# Watch log file
tail -f inpaint_eacps.log

# Or watch output directory
watch -n 5 'ls -d inpaint_eacps/task_*/ 2>/dev/null | wc -l'

# Check specific task
ls -lh inpaint_eacps/task_71680285/
```

## Check Running Process

```bash
# Find the process
ps aux | grep "inpaint_eacps/run.py"

# Check GPU usage
nvidia-smi
```
