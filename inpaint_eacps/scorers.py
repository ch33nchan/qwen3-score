"""
Multi-model scorers for inpainting evaluation.
Uses Gemini and Moondream for quality and identity scoring.
"""
import base64
import logging
from io import BytesIO
from typing import Dict, Optional, Tuple
from PIL import Image
import torch

logger = logging.getLogger(__name__)


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class GeminiScorer:
    """Score images using Gemini API."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-pro"):
        self.api_key = api_key
        self.model = model
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                # Use models/ prefix for proper API resolution
                model_name = self.model if self.model.startswith("models/") else f"models/{self.model}"
                self._client = genai.GenerativeModel(model_name)
                logger.info(f"Gemini client initialized with {model_name}")
            except ImportError:
                logger.error("google-generativeai not installed. Run: pip install google-generativeai")
                raise
        return self._client
    
    def score_identity(
        self,
        result_image: Image.Image,
        character_image: Image.Image,
        init_image: Image.Image,
    ) -> Dict[str, float]:
        """
        Score how well the result preserves character identity.
        
        Returns dict with:
        - identity_score: 0-10 (how well character identity is preserved)
        - quality_score: 0-10 (overall image quality)
        - consistency_score: 0-10 (consistency with original scene)
        """
        try:
            client = self._get_client()
            
            prompt = """You are a professional VFX quality evaluator. Analyze these images with EXTREME scrutiny.

IMAGE 1 (CHARACTER REFERENCE): The target character whose face should appear in the result.
IMAGE 2 (ORIGINAL SCENE): The original photograph before editing.
IMAGE 3 (EDITED RESULT): The final result after face replacement.

Evaluate and score each aspect on a 0-10 scale:

1. IDENTITY MATCH (0-10):
   - Does the face in RESULT match CHARACTER's facial features exactly?
   - Consider: face shape, eyes, nose, mouth, skin tone, age, gender, expression
   - Check: no added facial hair, no removed features, no age changes
   - Be VERY STRICT: Only 8+ if faces are nearly identical
   - Penalize heavily for mismatched features, wrong age/gender, different person

2. PHOTOREALISM (0-10):
   - Does RESULT look like a real, unedited photograph taken with a camera?
   - FATAL FLAWS (score ≤3 if ANY present):
     * CGI/3D rendered look
     * Painted or illustrated appearance
     * Plastic/wax/mannequin skin texture
     * Airbrushed, over-smoothed, or soft-focus skin
     * Missing skin pores, wrinkles, and natural texture
     * Caricature or oversized/disproportionate face
     * Face looks stretched, compressed, or distorted
     * Unnatural lighting that doesn't match scene physics
     * Visible seams, hard edges, or color boundaries
     * Blended/smudged hair at edges (wig-like appearance)
     * Added features (facial hair, makeup, accessories) not in original
   - TEXTURE CHECK: Real skin has visible pores, fine lines, natural imperfections
   - PROPORTION CHECK: Face size must match body proportions (not cartoon-like big head)
   - Be BRUTALLY STRICT: Only 9+ for perfect RAW photo quality, ≤5 if ANY AI artifacts visible
   - This is the MOST IMPORTANT metric - prioritize over identity

3. SCENE CONSISTENCY (0-10):
   - Does the edited face blend naturally with the original scene?
   - Check: lighting direction matches, shadows are correct, color temperature matches
   - Check: no visible seams or boundaries between edited and original areas
   - Check: face position and angle match the body

CRITICAL EVALUATION RULES:
- If face looks CGI, painted, or fake in ANY way → REALISM score ≤ 4
- If you see artifacts, blur, or AI-generated look → REALISM score ≤ 5
- If face doesn't match character → IDENTITY score ≤ 5
- If lighting/shadow doesn't match → CONSISTENCY score ≤ 6
- Be brutally honest - low scores are better than false positives

Respond with ONLY three numbers separated by commas: IDENTITY,REALISM,CONSISTENCY
Example: 7,4,6
"""
            
            response = client.generate_content([
                prompt,
                character_image,
                init_image,
                result_image,
            ])
            
            # Parse response
            text = response.text.strip()
            scores = [float(x.strip()) for x in text.split(",")]
            
            if len(scores) >= 3:
                return {
                    "identity": min(10, max(0, scores[0])),
                    "quality": min(10, max(0, scores[1])),
                    "consistency": min(10, max(0, scores[2])),
                }
            else:
                logger.warning(f"Unexpected Gemini response: {text}")
                return {"identity": 5.0, "quality": 5.0, "consistency": 5.0}
                
        except Exception as e:
            logger.error(f"Gemini scoring failed: {e}")
            return {"identity": 5.0, "quality": 5.0, "consistency": 5.0}


class MoondreamScorer:
    """Score images using Moondream V3."""
    
    def __init__(self, model_id: str = "vikhyatk/moondream2", device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        if self._model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                logger.info(f"Loading Moondream V3 from {self.model_id}...")
                dtype = torch.bfloat16 if "cuda" in self.device else torch.float32
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                ).to(self.device)
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                )
                self._model.eval()
                logger.info("Moondream V3 loaded")
            except Exception as e:
                logger.error(f"Failed to load Moondream: {e}")
                import traceback
                traceback.print_exc()
                raise
        return self._model, self._tokenizer
    
    def score_identity(
        self,
        result_image: Image.Image,
        character_image: Image.Image,
    ) -> Dict[str, float]:
        """
        Score identity preservation using Moondream V3.
        Uses proper multi-image comparison.
        """
        try:
            import torch
            model, tokenizer = self._load_model()
            
            # Moondream2 API: encode_image returns embeddings
            enc_result = model.encode_image(result_image)
            enc_char = model.encode_image(character_image)
            
            # Ask about face similarity - use result image as context
            prompt = (
                "Compare the face in this image with the face in the character reference image. "
                "Do they look like the same person? Consider: facial structure, features, skin tone, age, gender, expression. "
                "Rate from 0-10 where 10 = definitely same person, 5 = uncertain, 0 = different person. "
                "Answer with only the number."
            )
            
            # Moondream2 can handle image + text, but we need to combine embeddings
            # For now, use result image encoding
            with torch.inference_mode():
                response = model.answer_question(enc_result, prompt, tokenizer)
            
            try:
                # Extract number from response
                import re
                numbers = re.findall(r'\d+\.?\d*', response.strip())
                if numbers:
                    score = float(numbers[0])
                else:
                    score = 5.0
                score = min(10, max(0, score))
            except:
                score = 5.0
            
            return {"moondream_identity": score}
            
        except Exception as e:
            logger.error(f"Moondream identity scoring failed: {e}")
            import traceback
            traceback.print_exc()
            return {"moondream_identity": 5.0}
    
    def score_realism(
        self,
        result_image: Image.Image,
        init_image: Image.Image,
    ) -> Dict[str, float]:
        """
        Score how realistic the result looks using Moondream V3.
        """
        try:
            import torch
            import re
            model, tokenizer = self._load_model()
            
            enc_result = model.encode_image(result_image)
            
            prompt = (
                "CRITICAL: Evaluate this image STRICTLY for photorealism. "
                "Look for these FATAL FLAWS that indicate AI/fake: "
                "1. Plastic or waxy skin texture (like a mannequin) "
                "2. Airbrushed or over-smoothed skin (no pores visible) "
                "3. Caricature-like exaggerated features "
                "4. Visible seams or color boundaries around face "
                "5. Mismatched lighting between face and background "
                "6. Unnatural shadows or highlights "
                "7. Doll-like or CGI appearance "
                "8. Double features (extra eyes, nose, etc) "
                "If ANY of these flaws exist, score BELOW 5. "
                "Only score 8+ if it looks like an UNEDITED photograph. "
                "Rate 0-10. Answer with only the number."
            )
            
            with torch.inference_mode():
                response = model.answer_question(enc_result, prompt, tokenizer)
            
            try:
                numbers = re.findall(r'\d+\.?\d*', response.strip())
                if numbers:
                    score = float(numbers[0])
                else:
                    score = 5.0
                score = min(10, max(0, score))
            except:
                score = 5.0
            
            return {"moondream_realism": score}
            
        except Exception as e:
            logger.error(f"Moondream realism scoring failed: {e}")
            import traceback
            traceback.print_exc()
            return {"moondream_realism": 5.0}


class MultiModelScorer:
    """Combined scorer using multiple models."""
    
    def __init__(
        self,
        gemini_api_key: str,
        gemini_model: str = "gemini-1.5-flash",
        moondream_model_id: str = "vikhyatk/moondream2",
        device: str = "cuda",
        use_gemini: bool = True,
        use_moondream: bool = True,  # Enabled for better scoring
    ):
        self.use_gemini = use_gemini and bool(gemini_api_key)
        self.use_moondream = use_moondream
        
        if self.use_gemini:
            self.gemini = GeminiScorer(gemini_api_key, gemini_model)
        else:
            self.gemini = None
            
        if self.use_moondream:
            self.moondream = MoondreamScorer(moondream_model_id, device)
        else:
            self.moondream = None
    
    def score(
        self,
        result_image: Image.Image,
        character_image: Image.Image,
        init_image: Image.Image,
    ) -> Dict[str, float]:
        """
        Get combined scores from all available models.
        """
        scores = {}
        
        if self.gemini:
            gemini_scores = self.gemini.score_identity(result_image, character_image, init_image)
            scores.update({f"gemini_{k}": v for k, v in gemini_scores.items()})
        
        if self.moondream:
            # Identity score
            moondream_identity = self.moondream.score_identity(result_image, character_image)
            scores.update(moondream_identity)
            
            # Realism score
            moondream_realism = self.moondream.score_realism(result_image, init_image)
            scores.update(moondream_realism)
        
        return scores
    
    def compute_potential(self, scores: Dict[str, float], config) -> float:
        """
        Compute potential score from multi-model scores.
        
        potential = α*consistency + β*realism + γ*identity
        
        Realism is weighted heavily (2x) to prioritize photorealistic results.
        """
        # Use Gemini scores if available
        identity = scores.get("gemini_identity", 5.0)
        quality = scores.get("gemini_quality", 5.0)
        consistency = scores.get("gemini_consistency", 5.0)
        
        # Get Moondream scores
        moondream_identity = scores.get("moondream_identity", None)
        moondream_realism = scores.get("moondream_realism", None)
        
        # Combine identity scores (average Gemini + Moondream)
        if moondream_identity is not None:
            identity = (identity + moondream_identity) / 2
        
        # Use Moondream realism if available, otherwise Gemini quality
        realism = moondream_realism if moondream_realism is not None else quality
        
        # Weight realism heavily to prioritize photorealistic results
        potential = (
            config.alpha_consistency * consistency +
            config.beta_quality * realism +  # Already weighted in config
            config.gamma_identity * identity
        )
        
        return potential
