# EACPS: Efficient Adaptive Candidate-based Prompt Search for Image Editing

## Overview

**EACPS (Efficient Adaptive Candidate-based Prompt Search)** is a multi-stage search algorithm that improves image editing quality by intelligently exploring and refining candidate edits. This project compares EACPS against **TT-FLUX (Test-Time FLUX)**, using the actual TT-FLUX implementation from [sayakpaul/tt-scale-flux](https://github.com/sayakpaul/tt-scale-flux).

## What is EACPS?

EACPS is a two-stage approach to finding better image edits:

1. **Global Exploration**: Generate multiple candidates with diverse seeds
2. **Scoring & Selection**: Evaluate candidates using LAION aesthetic score (same as TT-FLUX)
3. **Local Refinement**: Generate refined versions of the top candidates
4. **Final Selection**: Pick the best overall candidate

## How It Differs from TT-FLUX

**TT-FLUX** (from the paper "Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps") uses random search - it generates N candidates exponentially (2^round) and picks the best one based on a verifier score.

**EACPS** improves upon this by:
- **Adaptively refining** the most promising candidates instead of pure random search
- More **compute-efficient** by focusing refinement on top candidates
- Same scoring metrics (LAION aesthetic) for fair comparison

## Project Structure

```
qwen3-score/
├── src/
│   ├── ttflux/             # TT-FLUX integration (based on sayakpaul/tt-scale-flux)
│   │   ├── pipeline.py     # TTFluxPipeline with random/zero-order search
│   │   ├── verifiers.py    # LAION aesthetic, CLIP score verifiers
│   │   └── verifier_prompt.txt
│   ├── run_comparison.py   # Main comparison script (EACPS vs TT-FLUX)
│   ├── eacps.py            # Core EACPS algorithm
│   ├── scorers.py          # Multi-metric scoring
│   └── compare.py          # Legacy comparison script
├── data/                   # Input images and prompts
├── experiments/            # Experimental results
└── requirements.txt
```

## Quick Start

### Installation

```bash
# Create venv with uv (recommended)
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Or with standard pip
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

### Run EACPS vs TT-FLUX Comparison (Image Generation)

```bash
# Generate images with FLUX.1-dev and compare methods
python src/run_comparison.py \
  --prompt "a tiny astronaut hatching from an egg on the moon" \
  --output_dir experiments/results/comparison \
  --device cuda:0 \
  --search_rounds 4 \
  --k_global 8 \
  --m_global 2 \
  --k_local 4
```

### Run EACPS vs TT-FLUX Comparison (Image Editing with Qwen)

```bash
# Edit images with Qwen-Image-Edit and compare methods
python src/run_comparison.py \
  --image data/bear.png \
  --prompt "Add a colorful art board and paintbrush in the bear's hands" \
  --output_dir experiments/results/edit_comparison \
  --device cuda:0 \
  --mode edit \
  --num_samples 8 \
  --k_global 8 \
  --m_global 2 \
  --k_local 4
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--prompt` | required | Generation/edit prompt |
| `--image` | None | Input image (enables edit mode) |
| `--output_dir` | experiments/results/comparison | Output directory |
| `--device` | cuda:0 | GPU device |
| `--mode` | auto | generate or edit (auto-detected from --image) |
| `--search_rounds` | 4 | TT-FLUX search rounds (2^round samples per round) |
| `--num_samples` | 8 | TT-FLUX samples for edit mode |
| `--k_global` | 8 | EACPS global exploration candidates |
| `--m_global` | 2 | EACPS top candidates to refine |
| `--k_local` | 4 | EACPS refinements per candidate |
| `--steps` | 50 | Inference steps |
| `--cfg` | 3.5/5.0 | Guidance scale (3.5 for FLUX, 5.0 for Qwen) |

## TT-FLUX Integration

This project integrates the TT-FLUX algorithm from [sayakpaul/tt-scale-flux](https://github.com/sayakpaul/tt-scale-flux):

- **Random Search**: Exponentially scales noise pool (2, 4, 8, 16...) per round
- **Zero-Order Search**: Uses gradient-free optimization with neighbor generation
- **Verifiers**: LAION Aesthetic Score (default), CLIP Score

## References

- TT-FLUX Paper: [Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps](https://arxiv.org/abs/2501.09732) (Ma et al., 2025)
- TT-FLUX Implementation: https://github.com/sayakpaul/tt-scale-flux
- Qwen-Image-Edit: https://huggingface.co/Qwen/Qwen-Image-Edit

## License

This project is for research purposes.
