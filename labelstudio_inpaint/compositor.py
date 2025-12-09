"""
Face compositing into mask region.
Handles face detection, extraction, color matching, scaling, and blending.
"""
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

# Try to import face detection libraries
FACE_DETECTOR = None

def _init_face_detector():
    """Initialize face detector (mediapipe or opencv)."""
    global FACE_DETECTOR
    
    if FACE_DETECTOR is not None:
        return FACE_DETECTOR
    
    # Try mediapipe first
    try:
        import mediapipe as mp
        mp_face = mp.solutions.face_detection
        FACE_DETECTOR = ("mediapipe", mp_face.FaceDetection(
            model_selection=1,  # Full range model
            min_detection_confidence=0.5
        ))
        logger.info("Using MediaPipe face detection")
        return FACE_DETECTOR
    except ImportError:
        pass
    
    # Fallback to OpenCV
    try:
        import cv2
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        FACE_DETECTOR = ("opencv", face_cascade)
        logger.info("Using OpenCV face detection")
        return FACE_DETECTOR
    except Exception:
        pass
    
    logger.warning("No face detection available, will use full image")
    FACE_DETECTOR = ("none", None)
    return FACE_DETECTOR


def detect_face_bbox(image: Image.Image, padding_ratio: float = 0.3) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect face in image and return bounding box with padding.
    
    Args:
        image: PIL Image (RGB)
        padding_ratio: Extra padding around face (0.3 = 30% extra on each side)
    
    Returns:
        (x_min, y_min, x_max, y_max) or None if no face detected
    """
    detector_type, detector = _init_face_detector()
    
    if detector_type == "none" or detector is None:
        return None
    
    img_array = np.array(image.convert("RGB"))
    h, w = img_array.shape[:2]
    
    if detector_type == "mediapipe":
        results = detector.process(img_array)
        if results.detections:
            # Get first (largest) face
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            x_min = int(bbox.xmin * w)
            y_min = int(bbox.ymin * h)
            box_w = int(bbox.width * w)
            box_h = int(bbox.height * h)
            
            # Add padding
            pad_x = int(box_w * padding_ratio)
            pad_y = int(box_h * padding_ratio)
            
            x_min = max(0, x_min - pad_x)
            y_min = max(0, y_min - pad_y)
            x_max = min(w, x_min + box_w + 2 * pad_x)
            y_max = min(h, y_min + box_h + 2 * pad_y)
            
            return (x_min, y_min, x_max, y_max)
    
    elif detector_type == "opencv":
        import cv2
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        
        if len(faces) > 0:
            # Get largest face
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            
            # Add padding
            pad_x = int(fw * padding_ratio)
            pad_y = int(fh * padding_ratio)
            
            x_min = max(0, x - pad_x)
            y_min = max(0, y - pad_y)
            x_max = min(w, x + fw + pad_x)
            y_max = min(h, y + fh + pad_y)
            
            return (x_min, y_min, x_max, y_max)
    
    return None


def extract_face(image: Image.Image, padding_ratio: float = 0.3) -> Image.Image:
    """
    Extract face region from image.
    Falls back to cropping to content if no face detected.
    
    Args:
        image: PIL Image (RGB or RGBA)
        padding_ratio: Extra padding around face
    
    Returns:
        Cropped face image
    """
    # Try face detection
    face_bbox = detect_face_bbox(image, padding_ratio)
    
    if face_bbox is not None:
        x_min, y_min, x_max, y_max = face_bbox
        logger.info(f"Face detected at ({x_min}, {y_min}, {x_max}, {y_max})")
        return image.crop((x_min, y_min, x_max, y_max))
    
    # Fallback: crop to content (non-transparent/non-white areas)
    logger.info("No face detected, cropping to content")
    return crop_to_content(image)


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


def get_mask_region_stats(init_image: Image.Image, mask_image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get color statistics from the region around the mask (for better color matching).
    """
    init_array = np.array(init_image.convert("RGB")).astype(np.float32)
    mask_array = np.array(mask_image.convert("L").resize(init_image.size, Image.Resampling.LANCZOS))
    
    # Dilate mask to get surrounding region
    from PIL import ImageFilter
    mask_dilated = mask_image.convert("L").resize(init_image.size, Image.Resampling.LANCZOS)
    mask_dilated = mask_dilated.filter(ImageFilter.MaxFilter(size=51))
    mask_dilated_array = np.array(mask_dilated)
    
    # Get pixels in dilated region but NOT in original mask (surrounding pixels)
    surround_mask = (mask_dilated_array > 128) & (mask_array < 128)
    
    if surround_mask.sum() < 100:
        # Not enough surrounding pixels, use whole image
        mean = init_array.mean(axis=(0, 1))
        std = init_array.std(axis=(0, 1)) + 1e-6
    else:
        surround_pixels = init_array[surround_mask]
        mean = surround_pixels.mean(axis=0)
        std = surround_pixels.std(axis=0) + 1e-6
    
    return mean, std


def match_color_to_region(
    source: Image.Image, 
    init_image: Image.Image,
    mask_image: Image.Image
) -> Image.Image:
    """
    Match the color grading of source image to the region around the mask.
    This gives better blending than matching to the whole image.
    """
    src_array = np.array(source.convert("RGB")).astype(np.float32)
    
    # Get source stats
    src_mean = src_array.mean(axis=(0, 1))
    src_std = src_array.std(axis=(0, 1)) + 1e-6
    
    # Get target region stats
    ref_mean, ref_std = get_mask_region_stats(init_image, mask_image)
    
    # Apply color transfer
    result = (src_array - src_mean) * (ref_std / src_std) + ref_mean
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result)


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
    
    # Apply color transfer
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
    feather_radius: int = 15,
    threshold: int = 20,
    extract_face_only: bool = True,
) -> Image.Image:
    """
    Composite the character FACE into the mask region of init_image.
    
    Strategy:
    1. Detect and extract face from character image
    2. Scale face to FIT the mask region (aspect fit, no distortion)
    3. Center the scaled face over the mask center
    4. Match color grading to the region AROUND the mask
    5. Apply feathered mask for smooth blending
    
    Args:
        init_image: Base image (RGB)
        character_image: Character image to extract face from (RGBA)
        mask_image: Mask showing where to place face (L/grayscale)
        feather_radius: Blur radius for mask edge smoothing
        threshold: Threshold for mask binarization
        extract_face_only: If True, detect and extract just the face
    
    Returns:
        Composited image (RGB)
    """
    init_rgb = init_image.convert("RGB")
    init_rgba = init_image.convert("RGBA")
    char_rgba = character_image.convert("RGBA")
    
    # Step 1: Extract face from character image
    if extract_face_only:
        face_image = extract_face(char_rgba, padding_ratio=0.4)
        logger.info(f"Extracted face: {face_image.size}")
    else:
        face_image = crop_to_content(char_rgba)
    
    # Resize mask to match init image
    mask_resized = mask_image.convert("L").resize(init_rgb.size, Image.Resampling.LANCZOS)
    
    # Get mask bounding box
    bbox = get_mask_bbox(mask_resized)
    if bbox is None:
        logger.warning("Empty mask, returning init_image")
        return init_rgb
    
    x_min, y_min, x_max, y_max = bbox
    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1
    mask_center_x = (x_min + x_max) // 2
    mask_center_y = (y_min + y_max) // 2
    
    logger.info(f"Mask bbox: ({x_min}, {y_min}) to ({x_max}, {y_max}), size: {bbox_width}x{bbox_height}")
    
    # Step 2: Match color grading to region around mask
    face_rgb = face_image.convert("RGB")
    face_color_matched = match_color_to_region(face_rgb, init_rgb, mask_resized)
    
    # Preserve alpha channel if present
    if face_image.mode == "RGBA":
        face_color_matched = face_color_matched.convert("RGBA")
        face_color_matched.putalpha(face_image.split()[3])
    
    # Step 3: Scale face to FIT the mask region (aspect fit - no overflow)
    face_w, face_h = face_color_matched.size
    scale_w = bbox_width / face_w
    scale_h = bbox_height / face_h
    scale = min(scale_w, scale_h)  # Use min to fit within mask
    
    new_w = int(face_w * scale)
    new_h = int(face_h * scale)
    face_scaled = face_color_matched.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    logger.info(f"Scaled face from {face_w}x{face_h} to {new_w}x{new_h}")
    
    # Step 4: Center the scaled face over the mask center
    paste_x = mask_center_x - new_w // 2
    paste_y = mask_center_y - new_h // 2
    
    # Create a layer for the face
    face_layer = Image.new("RGBA", init_rgb.size, (0, 0, 0, 0))
    face_layer.paste(face_scaled, (paste_x, paste_y))
    
    # Step 5: Create feathered mask for smooth edge blending
    mask_feathered = feather_mask(mask_resized, feather_radius=feather_radius, threshold=threshold)
    
    # Composite: face appears ONLY in masked region
    composite = Image.composite(face_layer, init_rgba, mask_feathered)
    
    return composite.convert("RGB")


def blend_with_mask(
    output_img: Image.Image,
    original_img: Image.Image,
    mask_img: Image.Image,
    feather: bool = True,
    feather_radius: int = 10,
    threshold: int = 30,
) -> Image.Image:
    """
    Blend output with original using mask.
    Keep output ONLY in masked area, keep original OUTSIDE mask.
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
