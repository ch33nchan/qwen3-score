"""
Face swap using InsightFace.
Swaps the face from character image into the init image at the mask region.
"""
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple
import logging
import cv2

logger = logging.getLogger(__name__)

# Global model cache
_FACE_ANALYZER = None
_FACE_SWAPPER = None


def _init_insightface():
    """Initialize InsightFace models."""
    global _FACE_ANALYZER, _FACE_SWAPPER
    
    if _FACE_ANALYZER is not None:
        return _FACE_ANALYZER, _FACE_SWAPPER
    
    try:
        import insightface
        from insightface.app import FaceAnalysis
        
        # Initialize face analyzer
        logger.info("Loading InsightFace models...")
        _FACE_ANALYZER = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        _FACE_ANALYZER.prepare(ctx_id=0, det_size=(640, 640))
        
        # Load face swapper model
        import os
        model_path = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")
        
        if not os.path.exists(model_path):
            logger.warning(f"Face swapper model not found at {model_path}")
            logger.info("Download from: https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx")
            logger.info(f"Place at: {model_path}")
            _FACE_SWAPPER = None
        else:
            _FACE_SWAPPER = insightface.model_zoo.get_model(model_path)
        
        logger.info("InsightFace models loaded")
        return _FACE_ANALYZER, _FACE_SWAPPER
        
    except ImportError:
        logger.error("InsightFace not installed. Run: pip install insightface onnxruntime-gpu")
        return None, None


def detect_faces(image: Image.Image):
    """Detect faces in image, return list of face objects."""
    analyzer, _ = _init_insightface()
    if analyzer is None:
        return []
    
    img_array = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    faces = analyzer.get(img_bgr)
    return faces


def get_face_in_mask_region(
    image: Image.Image,
    mask: Image.Image,
) -> Optional[object]:
    """Get the face that overlaps with the mask region."""
    faces = detect_faces(image)
    
    if not faces:
        logger.warning("No faces detected in image")
        return None
    
    # Get mask bbox
    mask_array = np.array(mask.convert("L").resize(image.size, Image.Resampling.LANCZOS))
    mask_points = np.where(mask_array > 128)
    
    if len(mask_points[0]) == 0:
        logger.warning("Empty mask")
        return faces[0] if faces else None
    
    mask_center_y = mask_points[0].mean()
    mask_center_x = mask_points[1].mean()
    
    # Find face closest to mask center
    best_face = None
    best_dist = float('inf')
    
    for face in faces:
        bbox = face.bbox
        face_center_x = (bbox[0] + bbox[2]) / 2
        face_center_y = (bbox[1] + bbox[3]) / 2
        
        dist = np.sqrt((face_center_x - mask_center_x)**2 + (face_center_y - mask_center_y)**2)
        if dist < best_dist:
            best_dist = dist
            best_face = face
    
    return best_face


def swap_face(
    init_image: Image.Image,
    character_image: Image.Image,
    mask_image: Optional[Image.Image] = None,
) -> Optional[Image.Image]:
    """
    Swap the face in init_image with the face from character_image.
    
    Args:
        init_image: Target image (where face will be replaced)
        character_image: Source image (face to use)
        mask_image: Optional mask to specify which face to replace
    
    Returns:
        Image with swapped face, or None if failed
    """
    analyzer, swapper = _init_insightface()
    
    if analyzer is None:
        logger.error("Face analyzer not available")
        return None
    
    if swapper is None:
        logger.error("Face swapper model not available")
        return None
    
    # Convert to BGR for InsightFace
    init_array = np.array(init_image.convert("RGB"))
    init_bgr = cv2.cvtColor(init_array, cv2.COLOR_RGB2BGR)
    
    char_array = np.array(character_image.convert("RGB"))
    char_bgr = cv2.cvtColor(char_array, cv2.COLOR_RGB2BGR)
    
    # Detect faces
    init_faces = analyzer.get(init_bgr)
    char_faces = analyzer.get(char_bgr)
    
    if not init_faces:
        logger.error("No face detected in init image")
        return None
    
    if not char_faces:
        logger.error("No face detected in character image")
        return None
    
    # Get source face (from character)
    source_face = char_faces[0]  # Use first/largest face
    
    # Get target face (in init image)
    if mask_image is not None:
        target_face = get_face_in_mask_region(init_image, mask_image)
        if target_face is None:
            target_face = init_faces[0]
    else:
        target_face = init_faces[0]
    
    logger.info(f"Swapping face: source bbox={source_face.bbox}, target bbox={target_face.bbox}")
    
    # Perform swap
    result_bgr = swapper.get(init_bgr, target_face, source_face, paste_back=True)
    
    # Convert back to RGB
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    result_image = Image.fromarray(result_rgb)
    
    return result_image


def swap_face_with_blend(
    init_image: Image.Image,
    character_image: Image.Image,
    mask_image: Image.Image,
    feather_radius: int = 5,
) -> Optional[Image.Image]:
    """
    Swap face and blend with original using mask.
    """
    # First do the swap
    swapped = swap_face(init_image, character_image, mask_image)
    
    if swapped is None:
        return None
    
    # Blend with original using mask
    from PIL import ImageFilter
    
    mask_resized = mask_image.convert("L").resize(init_image.size, Image.Resampling.LANCZOS)
    
    # Feather the mask
    if feather_radius > 0:
        mask_feathered = mask_resized.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    else:
        mask_feathered = mask_resized
    
    # Composite: swapped in mask area, original outside
    result = Image.composite(swapped, init_image.convert("RGB"), mask_feathered)
    
    return result


def process_faceswap_task(
    init_image: Image.Image,
    mask_image: Image.Image,
    character_image: Image.Image,
    output_dir: Optional[Path] = None,
    task_id: str = "unknown",
) -> dict:
    """
    Process a single face swap task.
    
    Returns dict with result and metadata.
    """
    logger.info(f"Processing face swap task {task_id}")
    
    # Perform face swap
    result = swap_face(init_image, character_image, mask_image)
    
    if result is None:
        return {
            "task_id": task_id,
            "success": False,
            "error": "Face swap failed",
        }
    
    # Save outputs
    if output_dir is not None:
        task_dir = output_dir / f"task_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        
        init_image.save(task_dir / "init.png")
        mask_image.save(task_dir / "mask.png")
        character_image.convert("RGB").save(task_dir / "character.png")
        result.save(task_dir / "result.png")
        
        import json
        with open(task_dir / "metrics.json", "w") as f:
            json.dump({"task_id": task_id, "method": "insightface"}, f, indent=2)
    
    return {
        "task_id": task_id,
        "success": True,
        "result_image": result,
    }


if __name__ == "__main__":
    # Test
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python faceswap.py <init.png> <character.png> <mask.png> [output.png]")
        sys.exit(1)
    
    init = Image.open(sys.argv[1])
    char = Image.open(sys.argv[2])
    mask = Image.open(sys.argv[3])
    
    result = swap_face(init, char, mask)
    
    if result:
        out_path = sys.argv[4] if len(sys.argv) > 4 else "faceswap_result.png"
        result.save(out_path)
        print(f"Saved to {out_path}")
    else:
        print("Face swap failed")
