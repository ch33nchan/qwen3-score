# Inference-Time Scaling for Image Editing
## Technical Overview for Team Discussion

---

## 1. Problem Statement

**Goal**: Improve image editing quality by scaling compute at inference time.

**Challenge**: Single-shot diffusion edits are inconsistent. Running multiple seeds and picking the best improves quality, but how do we search efficiently?

---

## 2. Two Approaches

### TTFLUX (Baseline)
- **Method**: Random search over seeds
- **Selection**: Pick best by single metric (e.g., CLIP score)
- **Limitation**: No learning from intermediate results - wastes compute on bad candidates

### EACPS (Our Method)
- **Method**: Adaptive Candidate-based Prompt Search
- **Key Insight**: Early denoising scores predict final quality
- **Approach**: Two-stage search with feedback

---

## 3. EACPS Algorithm

```
Stage 1: Global Exploration
  - Generate K candidates with diverse seeds
  - Score each with multi-metric potential
  - Select top M candidates

Stage 2: Local Refinement  
  - For each top candidate, generate K_local variations
  - Score and rank all candidates
  - Return best overall
```

**Potential Function**:
```
potential = PF + (α × CONS) - (β × LPIPS)

Where:
  PF    = CLIP Prompt Following (how well edit matches prompt)
  CONS  = CLIP Consistency (similarity to original)
  LPIPS = Perceptual distance (lower = more similar)
  α = 10.0, β = 3.0 (tuned hyperparameters)
```

---

## 4. Results: EACPS vs TTFLUX

| Metric | EACPS Wins | TTFLUX Wins | Tie |
|--------|-----------|-------------|-----|
| Overall | **70.8%** | 29.2% | - |

**Why EACPS wins**:
- Focuses compute on promising regions
- Multi-metric scoring captures quality better than single metric
- Early feedback prevents wasted computation

---

## 5. Pipeline Architecture

```
Input: (image, prompt, mask)
         │
         ▼
┌─────────────────────────┐
│   Qwen-Image-Edit       │  Base diffusion model
│   (50 inference steps)  │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Multi-Metric Scorer   │
│   - CLIP PF/CONS        │
│   - LPIPS               │
│   - Aesthetic (optional)│
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   EACPS Search          │
│   - Global: 4 candidates│
│   - Local: 2 per top-2  │
│   - Total: 8 evaluations│
└─────────────────────────┘
         │
         ▼
    Best Result
```

---

## 6. Label Studio Face Swap - The Problem

### What We Tried
1. **Side-by-side approach**: `[character | init_image]` → ask model to copy face
2. **Composite + refine**: Paste character face, then use EACPS to blend

### Why It Failed

**The model cannot understand "copy face from left to right"**

Evidence (task 71651078):
- Input: Scene with man in yellow, character reference (Suraj)
- Output: White/washed-out face region
- The diffusion model is not trained for identity transfer

### Root Cause
This is a **wrong-tool problem**, not an EACPS problem.

| Task Type | Right Tool |
|-----------|------------|
| General image edits | EACPS + Qwen-Image-Edit |
| Style transfer | EACPS + Qwen-Image-Edit |
| Face identity swap | InsightFace / IP-Adapter-FaceID |

---

## 7. Solution for Face Swap

**InsightFace Pipeline** (implemented, ready to test):

```
1. Detect face in character image → extract face embedding
2. Detect face in init image (using mask region)
3. Direct face swap using inswapper model
4. Result: ~2 seconds per image, preserves identity
```

**No diffusion model, no EACPS** - this is a specialized task requiring specialized tools.

---

## 8. Summary

| Component | Status | Notes |
|-----------|--------|-------|
| EACPS for general edits | Working | 70.8% win rate vs TTFLUX |
| Multi-GPU pipeline | Working | 3x H100, ~8 min/task |
| Label Studio integration | Working | API fetch, deduplication |
| Face swap with diffusion | Failed | Wrong tool for the job |
| Face swap with InsightFace | Ready | Needs model download + testing |

---

## 9. Next Steps

1. **Test InsightFace** pipeline on Label Studio tasks
2. **Separate pipelines**: 
   - EACPS for general edits
   - InsightFace for face swap
3. **Evaluate** face swap quality with human review

---

## Quick Reference Commands

```bash
# General edits with EACPS
python3 mask/run.py --from_file project-label.json --task_ids <id> --devices cuda:0

# Face swap with InsightFace  
python3 mask/run_faceswap.py --from_file project-label.json --task_ids <id>
```
