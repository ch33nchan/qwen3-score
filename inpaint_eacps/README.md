# Inpaint EACPS: Multi-Model Inference Scaling

## Overview

This pipeline implements **EACPS (Efficient Adaptive Candidate-based Prompt Search)** for inpainting tasks with multi-model scoring using Gemini and Moondream.

## Key Features

### 1. EACPS Inference Scaling
- **Two-stage search**: Global exploration → Local refinement
- **Adaptive**: Focuses compute on promising candidates
- **Efficient**: Better quality with same compute budget

### 2. Multi-Model Scoring
- **Gemini 1.5 Flash**: Scores identity, quality, and consistency
- **Moondream V3**: Optional identity verification
- **Combined potential**: Weighted scoring function

### 3. Automated Workflow
- **tmux integration**: Background execution with monitoring
- **Auto git push**: Results automatically committed and pushed
- **Caching**: Images cached locally for faster reruns

## Architecture

```
Input: (init_image, mask, character_image, character_name)
         │
         ▼
┌─────────────────────────┐
│   Qwen-Image-Edit       │  Generate K candidates
│   (50 inference steps)  │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Gemini Scorer         │  Score each candidate:
│   - Identity (0-10)     │  - Face match quality
│   - Quality (0-10)      │  - Image realism
│   - Consistency (0-10)  │  - Scene blend
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   EACPS Selection       │
│   potential =           │
│     α×consistency +     │
│     β×quality +         │
│     γ×identity          │
└─────────────────────────┘
         │
         ▼
    Best Result
```

## Potential Function

```
potential = α×consistency + β×quality + γ×identity

Where:
  α = 1.0  (consistency weight)
  β = 1.0  (quality weight)
  γ = 2.0  (identity weight - higher priority)
```

## Usage

### Basic Run

```bash
python3 inpaint_eacps/run.py \
    --task_id 71651078 \
    --gemini_api_key "YOUR_GEMINI_API_KEY" \
    --device cuda:0
```

### With tmux (Recommended)

```bash
# One-liner
cd /mnt/data1/srini/qwen3-score && \
GEMINI_API_KEY="your_key" \
./inpaint_eacps/start_tmux.sh 71651078

# Or with API key as argument
./inpaint_eacps/start_tmux.sh 71651078 "your_gemini_api_key"
```

**tmux windows:**
- `pipeline`: Main execution
- `gpus`: GPU monitoring (nvidia-smi)
- `logs`: Live log tailing

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task_id` | Required | Task ID from project-label.json |
| `--gemini_api_key` | Env var | Gemini API key |
| `--device` | `cuda:0` | CUDA device |
| `--k_global` | `4` | Global exploration candidates |
| `--m_global` | `2` | Top candidates for refinement |
| `--k_local` | `2` | Local refinements per candidate |
| `--output_dir` | `outputs/inpaint_eacps` | Output directory |
| `--cache_dir` | `data/cache` | Image cache directory |

## Output Structure

```
outputs/inpaint_eacps/task_71651078/
├── init.png                    # Original image
├── mask.png                    # Mask region
├── character.png               # Character reference
├── result.png                  # Best result
├── candidate_1_seed5000.png   # Top candidates
├── candidate_2_seed5031.png
├── candidate_3_seed5062.png
└── metrics.json               # Scores and metadata
```

## Example Output

```json
{
  "task_id": "71651078",
  "character_name": "Suraj",
  "best_seed": 5031,
  "best_phase": "local",
  "best_potential": 24.5,
  "best_scores": {
    "gemini_identity": 8.5,
    "gemini_quality": 7.8,
    "gemini_consistency": 8.2
  },
  "total_candidates": 8
}
```

## Dependencies

```bash
# Core
pip install torch transformers diffusers accelerate Pillow

# Gemini
pip install google-generativeai

# Optional: Moondream (disabled by default)
pip install transformers
```

## Auto Git Push

After completion, the pipeline automatically:
1. Stages output directory
2. Commits with message: `"Inpaint EACPS result: task {id} ({character})"`
3. Pushes to `origin/main`

**Note**: Git push failures are non-critical and won't stop execution.

## Comparison with Other Pipelines

| Pipeline | Method | Scoring | Use Case |
|----------|--------|---------|----------|
| `inpaint_eacps/` | EACPS + Gemini | Multi-model | General inpainting |
| `labelstudio_inpaint/` | InsightFace | Direct swap | Face identity transfer |
| `src/` | EACPS vs TTFLUX | CLIP/LPIPS | General edits |

## Troubleshooting

**Gemini API errors:**
- Check API key is valid
- Verify quota/rate limits
- Pipeline will use default scores if Gemini fails

**CUDA OOM:**
- Reduce `--k_global` or `--k_local`
- Use smaller batch size
- Try `--device cpu` (slow)

**Git push fails:**
- Check git credentials
- Verify remote is configured
- Non-critical - results still saved locally

## Performance

- **Per candidate**: ~30-60 seconds (Qwen-Edit generation)
- **Gemini scoring**: ~2-5 seconds per candidate
- **Total time**: ~5-10 minutes for full EACPS (8 candidates)

## Future Improvements

- [ ] Parallel candidate generation
- [ ] Moondream integration (currently disabled)
- [ ] Batch processing multiple tasks
- [ ] Multi-GPU support
- [ ] Real-time progress tracking
