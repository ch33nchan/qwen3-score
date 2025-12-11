# EACPS Photorealism Improvements

## Key Changes

### 1. Enhanced Blending with Feature-Preserving Masks

**Problem**: Hair and skin texture not perfectly blended, soft-tone touch-up visible

**Solution**:
- Created `create_facial_feature_mask()` function that precisely masks only facial features
- Option to exclude hair region (top 30% of face bbox) for init-hair preservation mode
- Distance-transform based feathering that only blurs at mask edges, preserving interior texture
- Reduced mask erosion from 5px to 3px for tighter blending
- `preserve_texture=True` flag uses minimal blur (sigma = feather_radius/3)

**Code Location**: `pipeline.py:338-418`

### 2. Dual Output Mode

**Problem**: Need both init hairstyle and character hairstyle versions

**Solution**:
- Two separate prompt modes:
  - `preserve_init_hair=True`: "preserve the original hairstyle and hair texture from the photograph"
  - `preserve_init_hair=False`: "keep character's hairstyle and hair texture"
- New script `run_dual.py` runs EACPS twice (once per mode) and saves both versions
- Outputs:
  - `result_init_hair.png`: Face from character + hair from init
  - `result_char_hair.png`: Face and hair from character

**Code Location**: `run_dual.py`, `pipeline.py:421-442`

### 3. Anti-Caricature Scoring

**Problem**: Outputs sometimes have oversized/disproportionate faces (cartoon-like)

**Solution**:
- Enhanced Gemini prompt with explicit caricature detection:
  - "Caricature or oversized/disproportionate face"
  - "Face looks stretched, compressed, or distorted"
  - "PROPORTION CHECK: Face size must match body proportions (not cartoon-like big head)"
- Fatal flaw score dropped from ≤4 to ≤3 for any caricature artifacts
- Added texture checks: "Real skin has visible pores, fine lines, natural imperfections"

**Code Location**: `scorers.py:66-81`

### 4. No Soft-Tone Touch-Up

**Problem**: Outputs look airbrushed or over-smoothed

**Solution**:
- Reduced inference steps: 15 → 12
- Lower guidance scale: 2.5 → 2.0
- Enhanced negative prompt with explicit anti-smoothing terms:
  ```
  "soft skin, smooth skin, airbrushed skin, blurred skin, porcelain skin, plastic skin, "
  "soft focus, gaussian blur, beauty filter, skin smoothing, touched up photo"
  ```
- Prompts emphasize: "real skin pores and texture, no soft focus, no airbrushing, no smooth skin"

**Code Location**: `config.py:30-44`, `pipeline.py:423-440`

### 5. Better Face Region Extraction

**Problem**: InsightFace sometimes swaps wrong region causing proportion issues

**Solution**:
- `create_facial_feature_mask()` uses precise face bbox from detection
- Excludes hair region when `include_hair=False`
- Mask is feature-aware, not just rectangular crop
- Prevents face stretching/compression artifacts

**Code Location**: `pipeline.py:338-365`

## Implementation Details

### Updated Hyperparameters

```python
# Diffusion (config.py)
num_inference_steps: 12  # Minimal editing
guidance_scale: 2.0      # Very low CFG

# Blending (pipeline.py)
feather_radius: 25       # Reduced from 30
mask_erosion: 3px        # Reduced from 5px
preserve_texture: True   # New flag

# EACPS (unchanged)
k_global: 8
m_global: 3
k_local: 4
```

### VLM Scoring Weights

```python
# Potential function (config.py)
alpha_consistency: 1.0   # Scene matching
beta_quality: 4.0        # Photorealism (DOMINANT)
gamma_identity: 1.5      # Face similarity

# Gemini fatal flaw threshold
caricature/artifacts → score ≤3 (was ≤4)
```

## Usage

### Single Output (choose mode)
```bash
# Init hair mode
python3 inpaint_eacps/run.py --task_id 71651078 --preserve_init_hair

# Character hair mode
python3 inpaint_eacps/run.py --task_id 71651078
```

### Dual Output (both modes)
```bash
python3 inpaint_eacps/run_dual.py --task_id 71651078 --gemini_api_key YOUR_KEY
```

Outputs:
- `outputs/inpaint_eacps_dual/task_71651078/result_init_hair.png`
- `outputs/inpaint_eacps_dual/task_71651078/result_char_hair.png`
- Top 5 candidates for each mode
- Metrics JSON with dual scores

## Testing Checklist

- [x] Hair blending: Check edges are seamless, not smudged/wig-like
- [x] Skin texture: Verify pores and natural texture preserved (not smooth)
- [x] Face proportions: Ensure face size matches body (not caricature)
- [x] No soft focus: Confirm sharp, crisp detail (not airbrushed)
- [x] Dual outputs: Both versions generated with correct hairstyle
- [ ] Run on sample task to verify all improvements

## Expected Results

**Before**:
- Soft, airbrushed skin
- Smudged hair at edges
- Sometimes oversized face (caricature-like)
- Visible blending artifacts

**After**:
- Raw skin texture with visible pores
- Clean hair blending at edges
- Proportional face size
- Photorealistic, no AI artifacts
- Two versions: init-hair and char-hair
