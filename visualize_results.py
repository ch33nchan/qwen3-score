#!/usr/bin/env python3
"""
Visualize dual output results to understand the differences.
Creates side-by-side comparison grids.

Usage:
    python3 visualize_results.py outputs/test_single/task_71650389
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json

def create_comparison_grid(task_dir: Path, output_path: Path):
    """Create a comparison grid for dual output results."""
    
    # Load images
    try:
        init_img = Image.open(task_dir / "init.png")
        character_img = Image.open(task_dir / "character.png")
        mask_img = Image.open(task_dir / "mask.png")
        result_init_hair = Image.open(task_dir / "result_init_hair.png")
        result_char_hair = Image.open(task_dir / "result_char_hair.png")
    except FileNotFoundError as e:
        print(f"Missing file: {e}")
        return None
    
    # Load metrics
    metrics_file = task_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
    else:
        metrics = {}
    
    # Resize all to same height
    target_h = 512
    target_w = int(init_img.width * target_h / init_img.height)
    
    imgs = [init_img, character_img, mask_img, result_init_hair, result_char_hair]
    resized = []
    for img in imgs:
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        resized.append(img.resize((target_w, target_h), Image.Resampling.LANCZOS))
    
    # Create grid: 2 rows x 3 cols
    # Row 1: Init | Character | Mask
    # Row 2: Result (Init Hair) | Result (Char Hair) | Difference
    
    grid_w = target_w * 3 + 40  # 20px padding between
    grid_h = target_h * 2 + 100  # 100px for labels
    
    grid = Image.new('RGB', (grid_w, grid_h), 'white')
    draw = ImageDraw.Draw(grid)
    
    # Try to load a font
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Row 1
    labels_row1 = ["Init Image", "Character Reference", "Mask"]
    for i, (img, label) in enumerate(zip(resized[:3], labels_row1)):
        x = i * (target_w + 20)
        grid.paste(img, (x, 50))
        draw.text((x + 10, 10), label, fill='black', font=font_large)
    
    # Row 2
    y_offset = target_h + 100
    
    # Init hair result
    grid.paste(resized[3], (0, y_offset))
    init_seed = metrics.get('init_hair', {}).get('seed', '?')
    init_potential = metrics.get('init_hair', {}).get('potential', 0)
    draw.text((10, y_offset - 40), f"Init Hair Preserved", fill='black', font=font_large)
    draw.text((10, y_offset - 20), f"Seed: {init_seed} | Score: {init_potential:.2f}", fill='gray', font=font_small)
    
    # Char hair result
    grid.paste(resized[4], (target_w + 20, y_offset))
    char_seed = metrics.get('char_hair', {}).get('seed', '?')
    char_potential = metrics.get('char_hair', {}).get('potential', 0)
    draw.text((target_w + 30, y_offset - 40), f"Character Hair", fill='black', font=font_large)
    draw.text((target_w + 30, y_offset - 20), f"Seed: {char_seed} | Score: {char_potential:.2f}", fill='gray', font=font_small)
    
    # Difference map
    diff_map = Image.new('RGB', (target_w, target_h), 'black')
    pixels_init = resized[3].load()
    pixels_char = resized[4].load()
    pixels_diff = diff_map.load()
    
    diff_count = 0
    for y in range(target_h):
        for x in range(target_w):
            r1, g1, b1 = pixels_init[x, y]
            r2, g2, b2 = pixels_char[x, y]
            diff = abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)
            if diff > 30:  # Significant difference
                pixels_diff[x, y] = (255, 0, 0)  # Red for differences
                diff_count += 1
            else:
                pixels_diff[x, y] = (int((r1+r2)/2), int((g1+g2)/2), int((b1+b2)/2))
    
    grid.paste(diff_map, (target_w * 2 + 40, y_offset))
    
    diff_pct = (diff_count / (target_w * target_h)) * 100
    draw.text((target_w * 2 + 50, y_offset - 40), f"Difference Map", fill='black', font=font_large)
    draw.text((target_w * 2 + 50, y_offset - 20), f"{diff_pct:.1f}% pixels differ", fill='red', font=font_small)
    
    # Title
    char_name = metrics.get('character_name', 'Unknown')
    task_id = metrics.get('task_id', '?')
    draw.text((grid_w // 2 - 150, grid_h - 30), 
              f"Task {task_id}: {char_name}", 
              fill='black', font=font_large)
    
    grid.save(output_path)
    print(f"✓ Saved comparison: {output_path}")
    print(f"  Difference: {diff_pct:.1f}% of pixels differ")
    print(f"  Init hair score: {init_potential:.2f}")
    print(f"  Char hair score: {char_potential:.2f}")
    
    return grid


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 visualize_results.py <task_dir>")
        print("Example: python3 visualize_results.py outputs/test_single/task_71650389")
        sys.exit(1)
    
    task_dir = Path(sys.argv[1])
    
    if not task_dir.exists():
        print(f"Directory not found: {task_dir}")
        sys.exit(1)
    
    output_path = task_dir / "comparison_grid.png"
    create_comparison_grid(task_dir, output_path)
    
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print('='*60)
    print(f"If the difference is <5%, the two outputs are nearly identical.")
    print(f"This means:")
    print(f"  1. The character already has similar hair to init image")
    print(f"  2. The mask may not include hair region")
    print(f"  3. EACPS is choosing similar seeds for both modes")
    print(f"\nCheck the mask - does it cover the hair region?")


if __name__ == "__main__":
    main()
