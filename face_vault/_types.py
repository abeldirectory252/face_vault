"""
face_vault._types
─────────────────
Dataclasses used throughout the FaceVault library.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class SpoofVerdict(Enum):
    """Verdict from the anti-spoofing module."""
    REAL = "real"
    FAKE = "fake"
    UNCERTAIN = "uncertain"


class DetectMode(Enum):
    """
    Identification mode.

    DETECT_WITH_IDENTITY
        Search the *entire* database.  Returns who the person is.
        Result: (matched, name, user_code)

    DETECT_WITHOUT_IDENTITY
        Given a specific ``user_code``, check ONLY that user's vectors.
        Much faster — skips the full FAISS scan.
        Result: (matched,)
    """
    DETECT_WITH_IDENTITY = "detect_with_identity"
    DETECT_WITHOUT_IDENTITY = "detect_without_identity"


@dataclass
class BoundingBox:
    """Axis-aligned bounding box for a detected face."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_int_tuple(self) -> tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))


@dataclass
class DetectedFace:
    """A face detected in an image."""
    bbox: BoundingBox
    landmarks: Optional[np.ndarray] = None  # (5, 2) key-points
    det_score: float = 0.0
    age: Optional[int] = None
    gender: Optional[int] = None  # 0=female, 1=male
    aligned_face: Optional[np.ndarray] = None  # cropped & aligned chip


@dataclass
class FaceEmbedding:
    """A face embedding vector with associated metadata."""
    vector: np.ndarray  # normalised 512-d vector
    face: DetectedFace
    quality: float = 0.0  # image-quality proxy score


@dataclass
class SpoofResult:
    """Result of anti-spoofing analysis on a face chip."""
    verdict: SpoofVerdict
    score: float  # 0.0 = definitely fake, 1.0 = definitely real
    method_scores: Dict[str, float] = field(default_factory=dict)
    details: str = ""


@dataclass
class Identity:
    """A registered identity in the database."""
    id: int
    user_code: str  # user-provided unique code, e.g. "OGH-00238"
    name: str
    num_vectors: int = 0
    created_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    reference_image_path: Optional[str] = None  # path to dataset folder or image


@dataclass
class MatchResult:
    """Result of searching the database for a face."""
    identity: Optional[Identity]
    distance: float  # L2 distance (lower = more similar)
    similarity: float  # cosine similarity (higher = more similar)
    is_match: bool = False


@dataclass
class RegisterResult:
    """Result of registering a new face."""
    success: bool
    identity: Optional[Identity] = None
    num_faces_found: int = 0
    images_registered: int = 0  # how many images were processed
    spoof_result: Optional[SpoofResult] = None
    message: str = ""


@dataclass
class IdentifyResult:
    """Result of identifying a face against the database."""
    faces_found: int = 0
    matches: List[MatchResult] = field(default_factory=list)
    spoof_results: List[SpoofResult] = field(default_factory=list)
    elapsed_ms: float = 0.0
    mode: Optional[str] = None  # which DetectMode was used
    image: Optional[np.ndarray] = None  # annotated output with stamp + info panel


@dataclass
class IdentifyWithResult:
    """
    Result of DETECT_WITH_IDENTITY mode.

    Searches the entire DB and returns who the person is.
    """
    matched: bool = False
    name: Optional[str] = None
    user_code: Optional[str] = None
    similarity: float = 0.0
    spoof_result: Optional[SpoofResult] = None
    elapsed_ms: float = 0.0
    image: Optional[np.ndarray] = None


@dataclass
class IdentifyWithoutResult:
    """
    Result of DETECT_WITHOUT_IDENTITY mode.

    Given a user_code, checks ONLY that user's vectors.
    Fast path — does not scan the full FAISS index.
    """
    matched: bool = False
    similarity: float = 0.0
    spoof_result: Optional[SpoofResult] = None
    elapsed_ms: float = 0.0
    image: Optional[np.ndarray] = None


@dataclass
class VerifyResult:
    """Result of 1:1 face verification."""
    is_same_person: bool = False
    similarity: float = 0.0
    distance: float = 0.0
    spoof_result: Optional[SpoofResult] = None
    elapsed_ms: float = 0.0
