# Quick Start: Run Inpaint EACPS on Server

## Prerequisites

```bash
# On your server, ensure you have:
# 1. Virtual environment activated
# 2. project-label.json file in project root
# 3. GEMINI_API_KEY set (or pass via --gemini_api_key)
```

## Commands

### Single Task
```bash
./inpaint_eacps/start_zellij.sh \
  --task_id 71680285 \
  --gemini_api_key YOUR_KEY
```

### Multiple Tasks
```bash
./inpaint_eacps/start_zellij.sh \
  --task_id 71680285 71651078 71634264 \
  --gemini_api_key YOUR_KEY \
  --device cuda:0
```

### Full Example with All Options
```bash
./inpaint_eacps/start_zellij.sh \
  --task_id 71680285 71651078 71634264 \
  --from_file project-label.json \
  --output_dir inpaint_eacps \
  --device cuda:0 \
  --gemini_api_key YOUR_KEY \
  --k_global 8 \
  --m_global 3 \
  --k_local 4
```

## Output Location

All outputs are saved to:
```
inpaint_eacps/task_{ID}/
├── result.png              # Best result
├── init.png                # Original image
├── mask.png                # Mask
├── character.png           # Character reference
├── faceswap_base.png       # InsightFace swap
├── candidate_*.png         # Top candidates
└── metrics.json            # Scores
```

## Monitor Progress

### Attach to Session
```bash
zellij attach inpaint_eacps
```

### Check Outputs
```bash
ls -lh inpaint_eacps/task_*/
```

## Progress Bars

You'll see:
1. **Overall Progress**: `2/5 tasks [00:15<00:45]`
2. **Global Exploration**: `8/8 [00:12<00:00]`
3. **Local Refinement**: `12/12 [00:08<00:00]`

## Environment Variable (Optional)

```bash
export GEMINI_API_KEY="your_key_here"
# Then omit --gemini_api_key from command
```
