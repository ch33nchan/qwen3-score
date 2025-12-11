#!/usr/bin/env python3
"""
Enhanced analysis: Show where the differences are and what changed.
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np
import json

def analyze_differences(task_dir: Path):
    """Detailed analysis of what changed between versions."""
    
    init_img = Image.open(task_dir / "init.png").convert('RGB')
    char_img = Image.open(task_dir / "character.png").convert('RGB')
    mask_img = Image.open(task_dir / "mask.png").convert('L')
    result_init = Image.open(task_dir / "result_init_hair.png").convert('RGB')
    result_char = Image.open(task_dir / "result_char_hair.png").convert('RGB')
    
    # Load metrics
    with open(task_dir / "metrics.json") as f:
        metrics = json.load(f)
    
    # Convert to numpy
    init_arr = np.array(init_img)
    char_arr = np.array(char_img)
    mask_arr = np.array(mask_img)
    result_init_arr = np.array(result_init)
    result_char_arr = np.array(result_char)
    
    # Calculate differences
    diff_init_char = np.abs(result_init_arr.astype(int) - result_char_arr.astype(int))
    diff_magnitude = np.sum(diff_init_char, axis=2)
    
    # Where are the differences?
    significant_diff = diff_magnitude > 30
    
    # Create heatmap
    heatmap = np.zeros_like(init_arr)
    for i in range(3):
        heatmap[:, :, i] = np.clip(diff_magnitude * 3, 0, 255)
    
    # Overlay on original
    overlay_init = result_init.copy()
    overlay_char = result_char.copy()
    heatmap_img = Image.fromarray(heatmap.astype(np.uint8))
    
    # Blend
    overlay_init = Image.blend(overlay_init, heatmap_img, 0.5)
    overlay_char = Image.blend(overlay_char, heatmap_img, 0.5)
    
    # Analysis by region
    mask_binary = mask_arr > 128
    
    # Hair region detection (top 40% of image, outside mask)
    h, w = mask_arr.shape
    hair_region = np.zeros_like(mask_binary)
    hair_region[:int(h*0.6), :] = True  # Top 60% of image
    hair_region = hair_region & ~mask_binary  # Outside face mask
    
    # Calculate stats
    total_pixels = h * w
    diff_pixels = np.sum(significant_diff)
    diff_in_mask = np.sum(significant_diff & mask_binary)
    diff_in_hair = np.sum(significant_diff & hair_region)
    diff_outside = np.sum(significant_diff & ~mask_binary & ~hair_region)
    
    print(f"\n{'='*70}")
    print(f"DETAILED DIFFERENCE ANALYSIS")
    print(f"{'='*70}")
    print(f"Character: {metrics['character_name']}")
    print(f"Task ID: {metrics['task_id']}")
    print(f"\n{'SCORING':-^70}")
    print(f"  Init Hair Version:  {metrics['init_hair']['potential']:.2f} (seed {metrics['init_hair']['seed']})")
    print(f"  Char Hair Version:  {metrics['char_hair']['potential']:.2f} (seed {metrics['char_hair']['seed']})")
    print(f"  Score Difference:   {abs(metrics['char_hair']['potential'] - metrics['init_hair']['potential']):.2f}")
    
    print(f"\n{'PIXEL DIFFERENCES':-^70}")
    print(f"  Total different pixels:    {diff_pixels:,} ({diff_pixels/total_pixels*100:.2f}%)")
    print(f"  In face region (mask):     {diff_in_mask:,} ({diff_in_mask/diff_pixels*100:.1f}% of changes)")
    print(f"  In hair region (top):      {diff_in_hair:,} ({diff_in_hair/diff_pixels*100:.1f}% of changes)")
    print(f"  Other areas:               {diff_outside:,} ({diff_outside/diff_pixels*100:.1f}% of changes)")
    
    print(f"\n{'WHAT CHANGED?':-^70}")
    if diff_in_hair > diff_in_mask:
        print(f"  ✓ PRIMARY: Hair styling/texture differences")
        print(f"  → Char hair version uses character's hairstyle more")
    elif diff_in_mask > diff_in_hair:
        print(f"  ✓ PRIMARY: Facial features/skin tone differences")
        print(f"  → Changes are mostly in the face region")
    else:
        print(f"  ✓ MIXED: Changes across both hair and face")
    
    if diff_pixels / total_pixels < 0.05:
        print(f"\n  ⚠️  Very subtle differences (<5% of image)")
        print(f"     This suggests the character hair is already similar to init")
    elif diff_pixels / total_pixels > 0.15:
        print(f"\n  ✓ Significant visual differences (>15% of image)")
        print(f"     The two modes produce meaningfully different results")
    
    # Scores analysis
    scores_init = metrics['init_hair']['scores']
    scores_char = metrics['char_hair']['scores']
    
    print(f"\n{'SCORE BREAKDOWN':-^70}")
    print(f"  Metric                   Init Hair    Char Hair    Difference")
    print(f"  {'-'*66}")
    
    for key in scores_init:
        if isinstance(scores_init[key], (int, float)) and isinstance(scores_char[key], (int, float)):
            init_val = scores_init[key]
            char_val = scores_char[key]
            diff_val = char_val - init_val
            arrow = "→" if abs(diff_val) < 0.5 else ("↑" if diff_val > 0 else "↓")
            print(f"  {key:24} {init_val:8.2f}     {char_val:8.2f}     {arrow} {diff_val:+.2f}")
    
    print(f"\n{'RECOMMENDATION':-^70}")
    score_diff = metrics['char_hair']['potential'] - metrics['init_hair']['potential']
    if score_diff > 2:
        print(f"  ✓ Character hair version is clearly better (+{score_diff:.1f} points)")
        print(f"    Use: result_char_hair.png")
    elif score_diff < -2:
        print(f"  ✓ Init hair version is clearly better (+{abs(score_diff):.1f} points)")
        print(f"    Use: result_init_hair.png")
    else:
        print(f"  → Scores are very close (±{abs(score_diff):.1f} points)")
        print(f"    Choose based on visual preference:")
        print(f"    - result_init_hair.png: More faithful to original hairstyle")
        print(f"    - result_char_hair.png: Uses character reference hair")
    
    # Create enhanced visualization
    target_size = (512, 512)
    
    imgs_to_show = [
        (result_init.resize(target_size), "Init Hair Result"),
        (result_char.resize(target_size), "Char Hair Result"),
        (heatmap_img.resize(target_size), "Difference Heatmap"),
    ]
    
    grid_w = 512 * 3 + 40
    grid_h = 512 + 80
    grid = Image.new('RGB', (grid_w, grid_h), 'white')
    draw = ImageDraw.Draw(grid)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    for i, (img, label) in enumerate(imgs_to_show):
        x = i * (512 + 20)
        grid.paste(img, (x, 60))
        draw.text((x + 10, 10), label, fill='black', font=font)
        
        if i == 0:
            draw.text((x + 10, 35), f"Score: {metrics['init_hair']['potential']:.1f}", fill='green', font=font)
        elif i == 1:
            draw.text((x + 10, 35), f"Score: {metrics['char_hair']['potential']:.1f}", fill='blue', font=font)
        elif i == 2:
            draw.text((x + 10, 35), f"{diff_pixels/total_pixels*100:.1f}% different", fill='red', font=font)
    
    output_path = task_dir / "analysis_detailed.png"
    grid.save(output_path)
    print(f"\n{'='*70}")
    print(f"Saved detailed visualization: {output_path}")
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_results.py <task_dir>")
        sys.exit(1)
    
    task_dir = Path(sys.argv[1])
    if not task_dir.exists():
        print(f"Directory not found: {task_dir}")
        sys.exit(1)
    
    analyze_differences(task_dir)
