"""
Multi-model scorers for inpainting evaluation.
Uses Gemini and Moondream for quality and identity scoring.
"""
import base64
import logging
from io import BytesIO
from typing import Dict, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class GeminiScorer:
    """Score images using Gemini API."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
                logger.info("Gemini client initialized")
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
            
            prompt = """You are evaluating an image editing result for realism and quality.

Image 1 (CHARACTER): The reference character whose identity should appear in the result.
Image 2 (ORIGINAL): The original scene image.
Image 3 (RESULT): The edited result where the character should be placed.

Score the following on a scale of 0-10:
1. IDENTITY (0-10): How well does the face in RESULT match the CHARACTER? Consider facial features, skin tone, expression, age, gender. Be strict - only high scores for very close matches.
2. REALISM (0-10): Does RESULT look like a real photograph? Check for: artifacts, blurring, unnatural lighting, color mismatches, seams, ghosting, double faces, distorted features. Be very strict - only 8+ for truly photorealistic results.
3. CONSISTENCY (0-10): How well does the edit blend with the scene? Lighting direction, shadows, color temperature, style, texture all match the original scene.

Be CRITICAL. Low scores for any visible artifacts, blur, or unrealistic elements.

Respond ONLY with three numbers separated by commas, like: 7,8,6
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
                
                logger.info(f"Loading Moondream from {self.model_id}...")
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    torch_dtype="auto",
                ).to(self.device)
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                )
                logger.info("Moondream loaded")
            except Exception as e:
                logger.error(f"Failed to load Moondream: {e}")
                raise
        return self._model, self._tokenizer
    
    def score_identity(
        self,
        result_image: Image.Image,
        character_image: Image.Image,
    ) -> Dict[str, float]:
        """
        Score identity preservation using Moondream.
        Compares faces in both images.
        """
        try:
            model, tokenizer = self._load_model()
            
            # Encode both images
            enc_result = model.encode_image(result_image)
            enc_char = model.encode_image(character_image)
            
            # Ask about face similarity
            prompt = (
                "Look at the face in the first image (character reference) and the face in the second image (result). "
                "Do they look like the same person? Consider facial features, skin tone, expression, and overall appearance. "
                "Answer with a number from 0 to 10, where 10 means definitely the same person, 5 means uncertain, and 0 means definitely different. "
                "Just respond with the number."
            )
            
            # Use result image encoding for the question
            response = model.answer_question(enc_result, prompt, tokenizer)
            
            # Also ask about character image for comparison
            prompt2 = (
                "Now look at the character reference image. Does the face in the result image match this character's identity? "
                "Answer with a number from 0 to 10. Just the number."
            )
            response2 = model.answer_question(enc_char, prompt2, tokenizer)
            
            # Average the two responses
            try:
                score1 = float(response.strip().split()[0])
                score1 = min(10, max(0, score1))
            except:
                score1 = 5.0
            
            try:
                score2 = float(response2.strip().split()[0])
                score2 = min(10, max(0, score2))
            except:
                score2 = 5.0
            
            avg_score = (score1 + score2) / 2
            
            return {"moondream_identity": avg_score}
            
        except Exception as e:
            logger.error(f"Moondream scoring failed: {e}")
            return {"moondream_identity": 5.0}
    
    def score_realism(
        self,
        result_image: Image.Image,
        init_image: Image.Image,
    ) -> Dict[str, float]:
        """
        Score how realistic the result looks compared to the original.
        """
        try:
            model, tokenizer = self._load_model()
            
            enc_result = model.encode_image(result_image)
            
            prompt = (
                "Look at this image. Does it look photorealistic and natural? "
                "Are there any obvious artifacts, blurring, or unnatural elements? "
                "Does it look like a real photograph? "
                "Answer with a number from 0 to 10, where 10 means very realistic and natural, "
                "and 0 means clearly fake or artificial. Just the number."
            )
            
            response = model.answer_question(enc_result, prompt, tokenizer)
            
            try:
                score = float(response.strip().split()[0])
                score = min(10, max(0, score))
            except:
                score = 5.0
            
            return {"moondream_realism": score}
            
        except Exception as e:
            logger.error(f"Moondream realism scoring failed: {e}")
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
        
        # Weight realism heavily (2x) to prioritize photorealistic results
        potential = (
            config.alpha_consistency * consistency +
            config.beta_quality * realism * 2.0 +  # Double weight for realism
            config.gamma_identity * identity
        )
        
        return potential
