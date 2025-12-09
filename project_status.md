# Project Status: Inference-Time Scaling for Image Editing

## Project Structure

```
qwen3-score/
├── src/                        # CORE: EACPS vs TTFLUX comparison
│   ├── eacps.py               # EACPS algorithm implementation
│   ├── scorers.py             # Multi-metric scoring (CLIP, LPIPS, etc.)
│   ├── compare.py             # Single comparison runner
│   ├── batch_compare.py       # Batch comparison across datasets
│   ├── benchmark.py           # Benchmarking utilities
│   └── ttflux/                # TTFLUX baseline
│       ├── pipeline.py        # TTFLUX random search
│       └── verifiers.py       # Score verification
│
├── labelstudio_inpaint/        # LABEL STUDIO: Face swap pipeline
│   ├── config.py              # Pipeline configuration
│   ├── labelstudio.py         # Label Studio API client
│   ├── pipeline.py            # EACPS-based inpainting (deprecated)
│   ├── faceswap.py            # InsightFace face swap
│   ├── run.py                 # Main runner (EACPS approach)
│   ├── run_faceswap.py        # Main runner (InsightFace approach)
│   └── worker.py              # Multi-GPU worker pool
│
├── data/                       # Test images and datasets
├── experiments/                # Experiment outputs
├── outputs/                    # Pipeline outputs
├── docs/                       # Documentation
├── writeup.md                  # Technical presentation doc
└── project_status.md           # This file
```

---

## Part 1: Core Pipeline (EACPS vs TTFLUX)

### Purpose
Compare inference-time scaling methods for general image editing.

### Methods

| Method | Approach | Selection |
|--------|----------|-----------|
| TTFLUX | Random seed search | Single metric (CLIP) |
| EACPS | Adaptive 2-stage search | Multi-metric potential |

### EACPS Algorithm
```
potential = PF + (α × CONS) - (β × LPIPS)
α = 10.0, β = 3.0
```

### Results
- **EACPS wins 70.8%** of comparisons vs TTFLUX
- Better quality with same compute budget

### How to Run
```bash
# Single comparison
python3 src/compare.py --image data/bear.png --prompt "make the bear wear a hat"

# Batch comparison
python3 src/batch_compare.py --dataset data/edits_v1.jsonl --output experiments/results
```

### Status: WORKING

---

## Part 2: Label Studio Inpaint (Face Swap)

### Purpose
Replace faces in images using character references from Label Studio.

### Data Flow
```
Label Studio API → Task (init_image, mask, character) → Face Swap → Result
```

### Approaches Tried

| Approach | Method | Result |
|----------|--------|--------|
| Side-by-side + EACPS | Diffusion model | FAILED - white/washed faces |
| Composite + EACPS | Paste + refine | FAILED - wrong tool |
| InsightFace | Direct face swap | READY TO TEST |

### Why Diffusion Failed
- Diffusion models cannot do identity transfer
- "Copy face from left to right" is not a learnable prompt
- This requires specialized face swap models

### How to Run
```bash
# InsightFace approach (recommended)
python3 labelstudio_inpaint/run_faceswap.py \
    --from_file project-label.json \
    --task_ids 71651078 \
    --output_dir outputs/faceswap_results

# EACPS approach (deprecated - doesn't work for face swap)
python3 labelstudio_inpaint/run.py \
    --from_file project-label.json \
    --task_ids 71651078 \
    --devices cuda:0
```

### Status: IN PROGRESS
- InsightFace pipeline implemented
- Needs model download and testing

---

## Quick Reference

### Core Pipeline (General Edits)
```bash
cd /mnt/data1/srini/qwen3-score
python3 src/compare.py --image <image> --prompt "<edit>"
```

### Label Studio Face Swap
```bash
cd /mnt/data1/srini/qwen3-score

# Setup InsightFace (one-time)
pip install insightface onnxruntime-gpu
mkdir -p ~/.insightface/models
# Download inswapper_128.onnx to ~/.insightface/models/

# Run
python3 labelstudio_inpaint/run_faceswap.py --from_file project-label.json --task_ids <id>
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/eacps.py` | Core EACPS algorithm |
| `src/scorers.py` | Multi-metric scoring |
| `src/ttflux/pipeline.py` | TTFLUX baseline |
| `labelstudio_inpaint/faceswap.py` | InsightFace face swap |
| `labelstudio_inpaint/labelstudio.py` | Label Studio API client |
| `writeup.md` | Technical presentation |

---

## TODO

- [ ] Test InsightFace on Label Studio tasks
- [ ] Download inswapper_128.onnx model
- [ ] Run face swap on full project
- [ ] Evaluate quality with human review
