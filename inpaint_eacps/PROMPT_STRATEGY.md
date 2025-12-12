# Structured Prompt Strategy for EACPS Inpainting

## Overview

The pipeline uses a **structured, chain-of-thought prompt** that breaks down face blending into explicit reasoning steps. This works synergistically with EACPS inference-time scaling.

## Prompt Structure

```
Objective: [Clear goal statement]

Input Analysis:
- Base Image Analysis: [Pose, lighting, texture analysis]
- Character Analysis: [Facial landmarks extraction]

Reasoning & Planning (Inference Steps):
Step 1 (Geometry): [Pose preservation]
Step 2 (Lighting Match): [Lighting consistency]
Step 3 (Blending): [Mask edge planning]

Execution: [Final generation instruction]
```

## Why This Works with EACPS

### 1. **Explicit Reasoning Steps**
The prompt guides the model through a logical sequence:
- **Step 1** → Geometry alignment (preserves pose)
- **Step 2** → Lighting matching (ensures consistency)
- **Step 3** → Blending strategy (defines boundaries)

This structure helps the model understand the task better, leading to more consistent candidates.

### 2. **EACPS Benefits from Consistency**
When candidates follow the same reasoning structure:
- **Better scoring**: More consistent results → easier to rank
- **Better refinement**: Local search focuses on variations of good candidates
- **Fewer outliers**: Structured approach reduces random failures

### 3. **Works with Low-Step Refinement**
With only **12 inference steps** and **CFG 2.0**, the model needs clear guidance:
- Structured prompt provides explicit instructions
- Each step is actionable and measurable
- Reduces ambiguity that would require more steps

## Integration with Pipeline

### Stage 0: InsightFace Face Swap
- Provides exact identity (no prompt needed)
- Creates base for refinement

### Stage 1-2: Qwen-Edit with Structured Prompt
- **Input**: Face-swapped image (identity already correct)
- **Prompt**: Structured reasoning for lighting/blending
- **Steps**: 12 (minimal, preserves texture)
- **CFG**: 2.0 (subtle refinement, not regeneration)

### EACPS Scoring
- **Gemini**: Evaluates if reasoning steps were followed
- **Moondream**: Verifies identity preservation
- **Potential**: Weighted toward realism (4x)

## Prompt Variants

### Mode 1: Preserve Init Hair
```
Step 3: Keep original hair, ears, neck, clothing.
Only replace facial region (T-zone, cheeks, jaw).
```

### Mode 2: Use Character Hair
```
Step 3: Keep original ears, neck, clothing.
Replace facial region AND hair with character's style.
```

## Expected Behavior

**With Structured Prompt:**
- More consistent candidates (follow same reasoning)
- Better pose preservation (explicit Step 1 instruction)
- Better lighting match (explicit Step 2 instruction)
- Cleaner blending (explicit Step 3 instruction)

**EACPS Selection:**
- Top candidates will all follow the reasoning structure
- Local refinement explores variations within the structure
- Best result combines: exact identity (swap) + photorealistic refinement (prompt)

## Customization

To use a custom prompt:

```python
# In run.py or CLI
config.model.override_prompt = "Your custom structured prompt here"
```

The override will be used instead of the default structured prompt.

## Best Practices

1. **Keep the structure**: Maintain Objective → Analysis → Steps → Execution
2. **Be specific**: Each step should have clear, measurable instructions
3. **Match the task**: Adjust steps based on what needs refinement
4. **Test variations**: EACPS will find the best prompt variant through scoring
