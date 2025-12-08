"""
Face compositing into mask region.
Handles cropping, color matching, scaling, and blending.
"""
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from typing import Tuple, Optional


def crop_to_content(image: Image.Image, padding: int = 2) -> Image.Image:
    """Crop image to its non-transparent/non-white content."""
    img_array = np.array(image.convert("RGBA"))
    
    # Check alpha channel if exists, otherwise check for non-white pixels
    if img_array.shape[2] == 4:
        alpha = img_array[:, :, 3]
        non_empty = alpha > 10
    else:
        rgb = img_array[:, :, :3]
        non_empty = np.any(rgb < 250, axis=2)
    
    rows = np.any(non_empty, axis=1)
    cols = np.any(non_empty, axis=0)
    
    if not rows.any() or not cols.any():
        return image
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Add padding
    y_min = max(0, y_min - padding)
    y_max = min(image.height - 1, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(image.width - 1, x_max + padding)
    
    return image.crop((x_min, y_min, x_max + 1, y_max + 1))


def match_color_grading(source: Image.Image, reference: Image.Image) -> Image.Image:
    """
    Match the color grading of source image to reference image.
    Uses simple histogram matching in RGB color space.
    """
    src_array = np.array(source.convert("RGB")).astype(np.float32)
    ref_array = np.array(reference.convert("RGB")).astype(np.float32)
    
    # Calculate mean and std for each channel
    src_mean = src_array.mean(axis=(0, 1))
    src_std = src_array.std(axis=(0, 1)) + 1e-6
    ref_mean = ref_array.mean(axis=(0, 1))
    ref_std = ref_array.std(axis=(0, 1)) + 1e-6
    
    # Apply color transfer: normalize source, then scale to reference stats
    result = (src_array - src_mean) * (ref_std / src_std) + ref_mean
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result)


def feather_mask(
    mask: Image.Image, 
    feather_radius: int = 5, 
    threshold: int = 32
) -> Image.Image:
    """Normalize mask and apply feathering for smooth blending."""
    mask_norm = ImageOps.autocontrast(mask)
    mask_binary = mask_norm.point(lambda p: 255 if p > threshold else 0)
    if feather_radius > 0:
        return mask_binary.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    return mask_binary


def get_mask_bbox(mask: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """Get bounding box of mask region (white area)."""
    mask_array = np.array(mask.convert("L"))
    rows = np.any(mask_array > 10, axis=1)
    cols = np.any(mask_array > 10, axis=0)
    
    if not rows.any() or not cols.any():
        return None
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return (x_min, y_min, x_max, y_max)


def composite_face_to_mask(
    init_image: Image.Image,
    character_image: Image.Image,
    mask_image: Image.Image,
    feather_radius: int = 10,
    threshold: int = 20,
) -> Image.Image:
    """
    Composite the character face into the mask region of init_image.
    
    Strategy:
    1. Crop character to content (remove transparent borders)
    2. Scale character to COVER the mask region (aspect fill, no distortion)
    3. Center the scaled character over the mask center
    4. Match color grading to init_image
    5. Apply feathered mask for smooth blending
    
    Args:
        init_image: Base image (RGB)
        character_image: Character/face to insert (RGBA)
        mask_image: Mask showing where to place character (L/grayscale)
        feather_radius: Blur radius for mask edge smoothing
        threshold: Threshold for mask binarization
    
    Returns:
        Composited image (RGB)
    """
    init_rgb = init_image.convert("RGB")
    init_rgba = init_image.convert("RGBA")
    char_rgba = character_image.convert("RGBA")
    
    # Crop character to actual content
    char_cropped = crop_to_content(char_rgba)
    
    # Resize mask to match init image
    mask_resized = mask_image.convert("L").resize(init_rgb.size, Image.Resampling.LANCZOS)
    
    # Get mask bounding box
    bbox = get_mask_bbox(mask_resized)
    if bbox is None:
        print("Warning: Empty mask, returning init_image")
        return init_rgb
    
    x_min, y_min, x_max, y_max = bbox
    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1
    mask_center_x = (x_min + x_max) // 2
    mask_center_y = (y_min + y_max) // 2
    
    # Match color grading BEFORE resizing
    char_rgb = char_cropped.convert("RGB")
    char_color_matched = match_color_grading(char_rgb, init_rgb)
    
    # Preserve alpha channel
    if char_cropped.mode == "RGBA":
        char_color_matched = char_color_matched.convert("RGBA")
        char_color_matched.putalpha(char_cropped.split()[3])
    
    # Scale character to COVER the mask region (aspect fill)
    char_w, char_h = char_color_matched.size
    scale_w = bbox_width / char_w
    scale_h = bbox_height / char_h
    scale = max(scale_w, scale_h)  # Use max to ensure full coverage
    
    new_w = int(char_w * scale)
    new_h = int(char_h * scale)
    char_scaled = char_color_matched.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Center the scaled character over the mask center
    paste_x = mask_center_x - new_w // 2
    paste_y = mask_center_y - new_h // 2
    
    # Create a layer for the character
    char_layer = Image.new("RGBA", init_rgb.size, (0, 0, 0, 0))
    char_layer.paste(char_scaled, (paste_x, paste_y))
    
    # Create feathered mask for smooth edge blending
    mask_feathered = feather_mask(mask_resized, feather_radius=feather_radius, threshold=threshold)
    
    # Composite: character appears ONLY in masked region
    composite = Image.composite(char_layer, init_rgba, mask_feathered)
    
    return composite.convert("RGB")


def blend_with_mask(
    output_img: Image.Image,
    original_img: Image.Image,
    mask_img: Image.Image,
    feather: bool = True,
    feather_radius: int = 6,
    threshold: int = 30,
) -> Image.Image:
    """
    Blend output with original using mask.
    Keep output ONLY in masked area, keep original OUTSIDE mask.
    
    Args:
        output_img: The edited/generated image
        original_img: The original base image
        mask_img: Mask defining the edit region
        feather: Whether to feather the mask edges
        feather_radius: Blur radius for feathering
        threshold: Threshold for mask binarization
    
    Returns:
        Blended image (RGB)
    """
    # Resize output to match original if needed
    if output_img.size != original_img.size:
        output_img = output_img.resize(original_img.size, Image.Resampling.LANCZOS)
    
    # Resize mask to match
    mask_resized = mask_img.convert("L").resize(original_img.size, Image.Resampling.LANCZOS)
    
    # Prepare mask
    if feather:
        mask_processed = feather_mask(mask_resized, feather_radius=feather_radius, threshold=threshold)
    else:
        mask_processed = mask_resized
    
    output_rgb = output_img.convert("RGB")
    original_rgb = original_img.convert("RGB")
    
    # Blend: masked area from output, rest from original
    result = Image.composite(output_rgb, original_rgb, mask_processed)
    return result
