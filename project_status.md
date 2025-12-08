# Project Status: Qwen3-Score Inference Scaling

## Overview

This project compares inference scaling methods (TT-FLUX and EACPS) for image generation/editing using Qwen-Image-Edit as the base model. The main use case is character inpainting for Label Studio tasks.

## Repository Structure

```
qwen3-score/
├── src/
│   ├── ttflux/                    # TT-FLUX integration module
│   │   ├── __init__.py
│   │   ├── pipeline.py            # TTFluxPipeline with random_search, zero_order_search
│   │   └── verifiers.py           # LAIONAestheticVerifier, CLIPScoreVerifier
│   ├── eacps.py                   # EACPS implementation
│   ├── scorers.py                 # Scoring utilities
│   ├── labelstudio_inpaint.py     # Label Studio API integration (main script)
│   ├── character_inpaint.py       # Direct task processing with hardcoded data
│   ├── run_qwen_comparison.py     # Standalone comparison script
│   └── batch_compare.py           # Batch comparison utilities
├── experiments/
│   └── outputs/                   # Experiment results
├── data/                          # Dataset files
└── requirements.txt
```

## Key Components

### 1. TT-FLUX (Test-Time FLUX Scaling)
- Source: https://github.com/sayakpaul/tt-scale-flux
- Methods:
  - `random_search`: Sample 2^round candidates, pick best by verifier score
  - `zero_order_search`: Gradient-free optimization with noise perturbation
- Verifier: LAIONAestheticVerifier (CLIP ViT-L/14 + aesthetic MLP)

### 2. EACPS (Efficient Adaptive Candidate-based Prompt Search)
- Parameters: k_global=8, m_global=2, k_local=4
- Local refinement with noise injection

### 3. Qwen-Image-Edit
- Model: `Qwen/Qwen-Image-Edit`
- Pipeline: `QwenImageEditPipeline` from diffusers
- Key config: `true_cfg_scale=4.0`, `num_inference_steps=50`, `torch.bfloat16`

## Current Progress

### Completed
1. Virtual environment setup with uv
2. TT-FLUX module integration (`src/ttflux/`)
3. Label Studio API integration (`src/labelstudio_inpaint.py`)
4. Fixed task ID vs data.id lookup issue
5. Hardcoded test task script (`src/character_inpaint.py`)

### In Progress
- Running inference scaling comparison on GPU server

## Label Studio Integration

### Project Details
- URL: https://label.dashtoon.ai/projects/14514/data
- API: `https://label.dashtoon.ai/api/`

### Task Structure
Each task contains:
- `init_image_url`: Background/scene image
- `character_image_url`: Character reference to insert
- `mask_image_url`: Binary mask for inpainting region
- `character_name`: Name of character
- `prompt`: Edit instruction

### Important: Task ID vs Data ID
- Label Studio has two IDs:
  - **Task ID**: Internal Label Studio ID (e.g., `2476572`)
  - **Data ID**: Custom ID in `data.id` field (e.g., `71676648`)
- The script now handles both - searches by data.id if direct task lookup fails

## Commands

### 1. Environment Setup (Local)
```bash
cd /Users/ch33nchan/Desktop/qwen3-score
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Environment Setup (Server)
```bash
cd /mnt/data1/srini/qwen3-score
git pull origin main
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

### 3. Run Label Studio Inpainting (Single Task)
```bash
python src/labelstudio_inpaint.py \
    --api_key c400214e3fa07bc2da35217ff6e42cc3e33c839f \
    --project_id 14514 \
    --task_id 71671165 \
    --output_dir experiments/labelstudio_test \
    --device cuda:0
```

### 4. Run Label Studio Inpainting (Multiple Tasks)
```bash
python src/labelstudio_inpaint.py \
    --api_key c400214e3fa07bc2da35217ff6e42cc3e33c839f \
    --project_id 14514 \
    --task_ids 71676648 71634265 71698193 \
    --output_dir experiments/labelstudio_batch \
    --device cuda:0
```

### 5. Run with Hardcoded Test Task (No API needed)
```bash
python src/character_inpaint.py --device cuda:0
```

### 6. Git Sync
```bash
# Local (push changes)
git add -A && git commit -m "message" && git push origin main

# Server (pull changes)
git pull origin main
```

## Configuration Options

### labelstudio_inpaint.py Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--api_key` | required | Label Studio API token |
| `--project_id` | required | Label Studio project ID |
| `--task_id` | - | Single task ID or data.id to process |
| `--task_ids` | - | Multiple task IDs to process |
| `--output_dir` | `experiments/labelstudio_output` | Output directory |
| `--device` | `cuda:0` | Device (cuda:X or cpu) |
| `--ttflux_samples` | 8 | Number of TT-FLUX samples (2^n) |
| `--k_global` | 8 | EACPS global candidates |
| `--m_global` | 2 | EACPS global rounds |
| `--k_local` | 4 | EACPS local candidates |

## Output Structure

```
output_dir/
├── task_{id}/
│   ├── inputs/
│   │   ├── init_image.png
│   │   ├── character_image.png
│   │   └── mask_image.png
│   ├── baseline/
│   │   └── output.png
│   ├── ttflux/
│   │   ├── round_0/
│   │   ├── round_1/
│   │   └── best.png
│   ├── eacps/
│   │   └── best.png
│   └── comparison.png
└── batch_results.json
```

## Known Issues & Fixes

### 1. Model ID Error
- **Problem**: `Qwen/Qwen2.5-VL-7B-Instruct` returns 404
- **Solution**: Use `Qwen/Qwen-Image-Edit` with `QwenImageEditPipeline`

### 2. Task ID Not Found (404)
- **Problem**: `71671165` not found at `/api/tasks/71671165`
- **Cause**: This is a `data.id`, not a Label Studio task ID
- **Solution**: Script now searches project tasks by `data.id` field

### 3. Guidance Scale Parameter
- **Problem**: `guidance_scale` not valid for Qwen-Image-Edit
- **Solution**: Use `true_cfg_scale=4.0` instead

## Next Steps

1. Run comparison on GPU server with actual tasks
2. Evaluate results for character consistency
3. Compare TT-FLUX vs EACPS quality/speed tradeoffs
4. Potentially integrate results back to Label Studio

## Files Modified (Latest Session)

1. `src/ttflux/__init__.py` - Created
2. `src/ttflux/pipeline.py` - Created
3. `src/ttflux/verifiers.py` - Created
4. `src/labelstudio_inpaint.py` - Created, then fixed data.id lookup
5. `src/character_inpaint.py` - Created with hardcoded test task
6. `src/run_qwen_comparison.py` - Created for standalone testing

## Git History (Recent)

```
b257c4a Fix labelstudio_inpaint.py to search by data.id
d716c3f Add character_inpaint.py with hardcoded test task
[earlier] Add TT-FLUX integration module
[earlier] Add run_qwen_comparison.py for Qwen-Image-Edit testing
```
