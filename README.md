# EACPS: Efficient Adaptive Candidate-based Prompt Search for Image Editing

## Overview

I've developed **EACPS (Efficient Adaptive Candidate-based Prompt Search)**, a multi-stage search algorithm that improves image editing quality by intelligently exploring and refining candidate edits. This project compares EACPS against **TT-FLUX (Test-Time FLUX)**, a state-of-the-art inference-time scaling method for diffusion models.

## What is EACPS?

EACPS stands for **Efficient Adaptive Candidate-based Prompt Search**. It's a two-stage approach to finding better image edits:

1. **Global Exploration**: Generate multiple candidates with diverse seeds
2. **Scoring & Selection**: Evaluate candidates using multiple metrics (CLIP prompt following, consistency, LPIPS)
3. **Local Refinement**: Generate refined versions of the top candidates
4. **Final Selection**: Pick the best overall candidate

## How It Differs from TT-FLUX

**TT-FLUX** (from the paper "Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps") uses random search - it generates N candidates and picks the best one based on a single metric. While effective, it doesn't refine promising candidates.

**EACPS** improves upon this by:
- Using **multi-metric scoring** (not just CLIP score)
- **Adaptively refining** the most promising candidates
- Balancing **prompt following**, **consistency**, and **perceptual quality**

## Project Structure

```
qwen3-score/
├── src/                    # Core source code
│   ├── compare.py          # Head-to-head comparison script
│   ├── batch_compare.py    # Batch evaluation across prompts
│   ├── eacps.py            # Core EACPS algorithm
│   ├── scorers.py          # Multi-metric scoring (CLIP, LPIPS, Aesthetic)
│   └── setup_labelstudio.py # Label Studio task generation
├── character-annon/        # Character inpainting pipeline
│   ├── run_inpaint.py      # Main inpainting script
│   └── README.md           # Character pipeline docs
├── data/                   # Input images and prompts
├── docs/                   # Documentation
│   ├── eacps_blog.html     # Detailed blog-style writeup
│   └── ttflux_reference.pdf # TT-FLUX paper reference
├── experiments/            # Experimental results
└── requirements.txt
```

## Quick Start

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

### Run a Single Comparison

```bash
python src/compare.py \
  --image data/bear.png \
  --prompt "Add a colorful art board and paintbrush in the bear's hands" \
  --output_dir experiments/comparison \
  --device cuda:0 \
  --num_samples 4 \
  --k_global 4 \
  --m_global 2 \
  --k_local 2 \
  --steps 50 \
  --cfg 5.0
```

### Run Batch Evaluation

```bash
python src/batch_compare.py \
  --prompts data/eval_prompts.jsonl \
  --output_dir experiments/batch_eval \
  --devices cuda:0,cuda:1 \
  --num_samples 4 \
  --k_global 4 \
  --m_global 2 \
  --k_local 2
```

### Character Inpainting (Label Studio Integration)

```bash
cd character-annon
python run_inpaint.py \
  --input project-label.json \
  --output_dir outputs \
  --device cuda:0 \
  --method both \
  --all
```

See [character-annon/README.md](character-annon/README.md) for full documentation.

## Results

Based on my experiments across multiple image editing tasks, EACPS consistently outperforms TT-FLUX on:
- **CLIP Score** (prompt following)
- **Aesthetic Score** (visual quality)
- **LPIPS** (perceptual similarity)

See [`docs/eacps_blog.html`](docs/eacps_blog.html) for detailed analysis, explanations, and examples.

## References

- TT-FLUX Paper: "Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps" (Ma et al., 2024)
- Qwen-Image-Edit: https://huggingface.co/Qwen/Qwen-Image-Edit

## License

This project is for research purposes.
