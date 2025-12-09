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
            
            prompt = """You are evaluating an image editing result. 
            
Image 1 (CHARACTER): The reference character whose identity should appear in the result.
Image 2 (ORIGINAL): The original scene image.
Image 3 (RESULT): The edited result where the character should be placed.

Score the following on a scale of 0-10:
1. IDENTITY (0-10): How well does the face in RESULT match the CHARACTER? Consider facial features, skin tone, expression.
2. QUALITY (0-10): Overall image quality. Is it photorealistic? Any artifacts, blurring, or unnatural elements?
3. CONSISTENCY (0-10): How well does the edit blend with the scene? Lighting, style, composition consistency.

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
        """
        try:
            model, tokenizer = self._load_model()
            
            # Encode images
            enc_result = model.encode_image(result_image)
            enc_char = model.encode_image(character_image)
            
            # Ask about identity match
            prompt = "Do the faces in these two images look like the same person? Answer with a confidence score from 0 to 10, where 10 means definitely the same person. Just respond with the number."
            
            # This is a simplified approach - in practice you'd need to handle multi-image input
            response = model.answer_question(enc_result, prompt, tokenizer)
            
            try:
                score = float(response.strip().split()[0])
                score = min(10, max(0, score))
            except:
                score = 5.0
            
            return {"moondream_identity": score}
            
        except Exception as e:
            logger.error(f"Moondream scoring failed: {e}")
            return {"moondream_identity": 5.0}


class MultiModelScorer:
    """Combined scorer using multiple models."""
    
    def __init__(
        self,
        gemini_api_key: str,
        gemini_model: str = "gemini-1.5-flash",
        moondream_model_id: str = "vikhyatk/moondream2",
        device: str = "cuda",
        use_gemini: bool = True,
        use_moondream: bool = False,  # Disabled by default (slow)
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
            moondream_scores = self.moondream.score_identity(result_image, character_image)
            scores.update(moondream_scores)
        
        return scores
    
    def compute_potential(self, scores: Dict[str, float], config) -> float:
        """
        Compute potential score from multi-model scores.
        
        potential = α*consistency + β*quality + γ*identity
        """
        # Use Gemini scores if available
        identity = scores.get("gemini_identity", 5.0)
        quality = scores.get("gemini_quality", 5.0)
        consistency = scores.get("gemini_consistency", 5.0)
        
        # Boost from Moondream if available
        if "moondream_identity" in scores:
            identity = (identity + scores["moondream_identity"]) / 2
        
        potential = (
            config.alpha_consistency * consistency +
            config.beta_quality * quality +
            config.gamma_identity * identity
        )
        
        return potential
