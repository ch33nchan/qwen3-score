import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from models import (
    BoundingBoxResponse,
    CharacterMatchRequest,
    CharacterMatchResponse,
    CharacterMatchResult,
    DetectFacesRequest,
    DetectFacesResponse,
    HealthResponse,
)
from services import detect_faces_in_image, download_image_from_url, process_character_matching
from storage import AzureFileSystem
from triton_client import get_triton_client

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global storage client
azure_storage = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global azure_storage

    # Startup
    logger.info("Starting Automask FastAPI service...")

    # Initialize Azure storage
    if settings.azure_storage_connection_string and settings.azure_storage_sas_token:
        try:
            azure_storage = AzureFileSystem()
            logger.info("Azure storage initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Azure storage: {e}")
            azure_storage = None
    else:
        logger.warning("Azure storage not configured - connection string or SAS token missing")

    # Initialize and test Triton client
    triton_client = get_triton_client()
    if triton_client and triton_client.is_healthy():
        logger.info("Triton client connected and healthy")
    else:
        logger.warning("Triton client not available or unhealthy")

    yield

    # Shutdown
    logger.info("Shutting down Automask FastAPI service...")


# Create FastAPI app
app = FastAPI(
    title="Automask Service",
    description="Character detection and mask generation service using Triton server",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_azure_storage():
    """Dependency to get Azure storage client"""
    if azure_storage is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Azure storage not configured or unavailable"
        )
    return azure_storage


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    triton_client = get_triton_client()

    return HealthResponse(
        status="healthy",
        triton_server_connected=triton_client is not None and triton_client.is_healthy(),
        gemini_client_ready=True,  # Assume it's ready if we got this far
        azure_storage_ready=azure_storage is not None,
        version="0.1.0",
    )


@app.post("/detect-faces", response_model=DetectFacesResponse)
async def detect_faces_endpoint(request: DetectFacesRequest):
    """
    Detect faces in an image from URL.

    Returns bounding boxes of detected faces without character matching.
    """
    try:
        # Download image
        image = await download_image_from_url(request.image_url)
        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to download image from provided URL"
            )

        # Detect faces
        bboxes = detect_faces_in_image(image)

        # Convert to response format
        faces = [BoundingBoxResponse(x0=bbox.x0, y0=bbox.y0, x1=bbox.x1, y1=bbox.y1) for bbox in bboxes]

        return DetectFacesResponse(success=True, faces=faces, message=f"Successfully detected {len(faces)} faces")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during face detection: {str(e)}",
        ) from e


@app.post("/match-characters", response_model=CharacterMatchResponse)
async def match_characters_endpoint(
    request: CharacterMatchRequest, storage: AzureFileSystem = Depends(get_azure_storage)
):
    """
    Detect faces and match them to character descriptions, then generate mask URLs.

    This is the main endpoint that performs the full workflow:
    1. Downloads image from URL
    2. Detects faces using Triton server
    3. Matches faces to character descriptions using Gemini
    4. Generates organic masks for matched faces
    5. Uploads masks to Azure storage
    6. Returns list of mask URLs
    """
    try:
        # Validate input
        if len(request.character_metadata) > settings.max_characters:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Too many characters. Maximum allowed: {settings.max_characters}",
            )

        # Download image
        image = await download_image_from_url(request.image_url)
        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to download image from provided URL"
            )

        # Detect faces first to get total count
        detected_bboxes = detect_faces_in_image(image)
        total_faces_detected = len(detected_bboxes)

        # Process character matching
        results_data = await process_character_matching(
            image=image, character_descriptions=request.character_metadata, azure_storage=storage
        )

        # Convert to response format
        results = []
        for char_desc, mask_url, matched, bbox, confidence in results_data:
            bbox_response = None
            if bbox is not None:
                bbox_response = BoundingBoxResponse(x0=bbox.x0, y0=bbox.y0, x1=bbox.x1, y1=bbox.y1)

            results.append(
                CharacterMatchResult(
                    character_description=char_desc,
                    mask_url=mask_url,
                    matched=matched,
                    bounding_box=bbox_response,
                    confidence_score=confidence,
                )
            )

        return CharacterMatchResponse(
            success=True,
            results=results,
            message=f"Successfully processed {len(request.character_metadata)} characters. "
            f"Total faces detected: {total_faces_detected}. "
            f"Successfully matched: {sum(1 for r in results if r.matched)}.",
            total_faces_detected=total_faces_detected,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in character matching: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during character matching: {str(e)}",
        ) from e


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Automask Service - Character Detection and Mask Generation",
        "version": "0.1.0",
        "endpoints": {"health": "/health", "detect_faces": "/detect-faces", "match_characters": "/match-characters"},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", workers=4)