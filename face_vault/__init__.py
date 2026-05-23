"""
FaceVault — Fast face recognition with anti-spoofing and vector database.
"""
from .vault import FaceVault
from .engine import FaceEngine
from .anti_spoof import AntiSpoof
from .database import FaceDatabase
from ._types import (
    BoundingBox, DetectedFace, DetectMode, FaceEmbedding,
    Identity, MatchResult, RegisterResult,
    IdentifyResult, IdentifyWithResult, IdentifyWithoutResult,
    VerifyResult,
    SpoofResult, SpoofVerdict,
)
from .utils import load_image, draw_results, cosine_similarity, stamp_identify_image

__version__ = "0.1.0"
__all__ = [
    "FaceVault", "FaceEngine", "AntiSpoof", "FaceDatabase",
    "BoundingBox", "DetectedFace", "DetectMode", "FaceEmbedding",
    "Identity", "MatchResult", "RegisterResult",
    "IdentifyResult", "IdentifyWithResult", "IdentifyWithoutResult",
    "VerifyResult",
    "SpoofResult", "SpoofVerdict",
    "load_image", "draw_results", "cosine_similarity", "stamp_identify_image",
]
