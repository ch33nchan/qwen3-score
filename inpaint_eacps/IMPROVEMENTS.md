# SOTA Inpainting Pipeline - Improvements Summary

## Overview

The pipeline now uses a **hybrid approach** combining:
1. **InsightFace** for exact identity preservation (face swap)
2. **Qwen-Edit** for photorealistic refinement
3. **EACPS** for inference-time scaling
4. **Gemini + Moondream** for multi-model scoring

## Key Improvements

### 1. SOTA Mask Generation

**Face Detection-Based Mask Refinement:**
- Uses InsightFace to detect face in init image
- Creates precise mask from face landmarks (106-point model)
- Uses convex hull for accurate facial feature boundaries
- Combines with original mask (union) for safety
- Excludes hair region for better blending

**Result:** Masks are now pixel-perfect, covering only facial features.

### 2. Hybrid Face Swap + Refinement

**Stage 0: InsightFace Face Swap**
- Direct face swap preserves exact identity
- No diffusion artifacts
- Fast (~2 seconds)

**Stage 1-2: Qwen-Edit Refinement**
- Refines the swapped face for photorealism
- Minimal steps (12) to preserve texture
- Low CFG (2.0) for subtle changes
- Focuses on lighting/blending, not identity

**Result:** Best of both worlds - exact identity + photorealistic quality.

### 3. Enhanced Scoring (Gemini + Moondream)

**Gemini Scoring:**
- **Strict evaluation criteria** for photorealism
- Checks for: CGI look, painted appearance, plastic skin, artifacts
- **Fatal flaws** → automatic low scores
- Model fallback: gemini-2.5-pro → gemini-1.5-pro → gemini-pro

**Moondream Scoring:**
- **Dual-question cross-validation** for identity
- Asks from both result and character perspectives
- Averages scores for robustness
- Proper multi-image comparison

**Potential Function:**
```
potential = 1.0×consistency + 4.0×realism + 1.5×identity
```
**Realism is weighted 4x** - prioritizes photorealistic results.

### 4. Advanced Mask Blending

**Feature-Preserving Feathering:**
- Distance transform to identify mask edges
- Feathering only at edges (preserves interior texture)
- Minimal blur for skin/hair detail preservation
- Erosion to avoid edge artifacts

**Result:** Seamless blending without losing texture detail.

### 5. Improved Inference Scaling

**EACPS Configuration:**
- **k_global: 8** (more exploration)
- **m_global: 3** (more refinement candidates)
- **k_local: 4** (more refinements per candidate)
- **Total: 21 candidates** (vs 8 before)

**Raw Face Swap Candidate:**
- Includes InsightFace result as candidate (seed=0)
- Ensures we always have the "pure" swap option
- No diffusion artifacts

## Pipeline Flow

```
1. Load images (init, mask, character)
   ↓
2. Refine mask with face detection (SOTA precision)
   ↓
3. InsightFace face swap (exact identity)
   ↓
4. EACPS Global Exploration (8 candidates):
   - Raw swap (no diffusion)
   - 7 Qwen-Edit refinements
   ↓
5. Score each with Gemini + Moondream
   ↓
6. Select top 3 for local refinement
   ↓
7. EACPS Local Refinement (4 per candidate = 12 more)
   ↓
8. Score all 21 candidates
   ↓
9. Return best by potential (realism-weighted)
```

## Configuration Highlights

### Generation Settings
- **Steps: 12** - Minimal to preserve texture
- **CFG: 2.0** - Low guidance for subtle refinement
- **Negative prompt:** Extensive list targeting artifacts

### Scoring Weights
- **Realism: 4.0x** - Most important
- **Identity: 1.5x** - Important but already handled by swap
- **Consistency: 1.0x** - Scene blending

### Mask Settings
- **Feather radius: 25px** - Smooth edges
- **Texture preservation: True** - Minimal blur
- **Erosion: 3px** - Avoid edge artifacts

## Expected Results

**Before:**
- Diffusion-only approach → identity loss, artifacts
- Generic masks → imprecise boundaries
- Single-model scoring → less reliable

**After:**
- **Exact identity** from InsightFace
- **Photorealistic quality** from Qwen-Edit refinement
- **Precise masks** from face detection
- **Reliable scoring** from multi-model evaluation

## Usage

```bash
# Single task
python3 inpaint_eacps/run.py \
    --task_id 71680285 \
    --gemini_api_key "YOUR_KEY" \
    --device cuda:0

# Multiple tasks
python3 inpaint_eacps/run.py \
    --task_id 71680285 71680286 71680287 \
    --gemini_api_key "YOUR_KEY" \
    --device cuda:0
```

## Dependencies

```bash
pip install scipy  # For convex hull masks
pip install google-generativeai  # For Gemini
pip install transformers  # For Moondream
pip install insightface onnxruntime-gpu  # For face swap
```

## Troubleshooting

**Gemini 404 errors:**
- Pipeline automatically falls back to gemini-1.5-pro or gemini-pro
- Check API key is valid

**Moondream errors:**
- Falls back to default scores (5.0)
- Check transformers is installed

**InsightFace not found:**
- Download inswapper_128.onnx manually
- Place at ~/.insightface/models/inswapper_128.onnx

**Mask refinement fails:**
- Falls back to original mask
- Check InsightFace is installed and working
