"""
face_vault.engine
─────────────────
Wraps InsightFace for face detection + ArcFace embedding.
Provides a single, shared inference session for speed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

try:
    import insightface
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except ImportError:
    HAS_INSIGHTFACE = False

from ._types import BoundingBox, DetectedFace, FaceEmbedding

logger = logging.getLogger(__name__)


class FaceEngine:
    """
    Core detection + embedding engine.

    Uses InsightFace (RetinaFace detector + ArcFace recogniser) under the
    hood.  Both models run through ONNX-Runtime, so GPU acceleration is
    automatic when ``onnxruntime-gpu`` is installed.

    Parameters
    ----------
    ctx_id : int
        Execution-provider index.  ``-1`` → CPU, ``0`` → first GPU.
    det_size : tuple[int, int]
        Input resolution for the detector.  Larger = more accurate but
        slower.  ``(640, 640)`` is a good default.
    det_thresh : float
        Minimum detection confidence.
    model_pack : str
        InsightFace model-zoo pack name.  ``"buffalo_l"`` gives the best
        accuracy; ``"buffalo_sc"`` is the lightest.
    """

    def __init__(
        self,
        ctx_id: int = -1,
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        model_pack: str = "buffalo_l",
        model_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        if not HAS_INSIGHTFACE:
            raise ImportError(
                "insightface is required.  Install with:\n"
                "  pip install insightface onnxruntime"
            )

        self._det_thresh = det_thresh

        # Determine the root directory for models
        if model_dir is not None:
            model_root = Path(model_dir).resolve()
        else:
            proj_root = Path(__file__).resolve().parent.parent
            if "site-packages" in str(proj_root) or "dist-packages" in str(proj_root):
                model_root = Path.home() / ".insightface"
            else:
                model_root = proj_root / "model"

        try:
            self._app = FaceAnalysis(
                name=model_pack,
                root=str(model_root),
                allowed_modules=["detection", "recognition"],
                providers=self._get_providers(),
            )
            self._app.prepare(ctx_id=ctx_id, det_size=det_size)
        except (AssertionError, Exception) as e:
            logger.warning(
                "Failed to initialize FaceAnalysis with model pack '%s' at '%s' (Error: %s). "
                "Attempting to clear corrupted model download and retry...",
                model_pack, model_root, e
            )
            specific_model_dir = model_root / "models" / model_pack
            if specific_model_dir.exists():
                try:
                    logger.info("Removing corrupted model directory: %s", specific_model_dir)
                    import shutil
                    shutil.rmtree(specific_model_dir, ignore_errors=True)
                except Exception as clean_err:
                    logger.error("Failed to remove corrupted model directory %s: %s", specific_model_dir, clean_err)
            
            # Retry initializing after cleaning up
            self._app = FaceAnalysis(
                name=model_pack,
                root=str(model_root),
                allowed_modules=["detection", "recognition"],
                providers=self._get_providers(),
            )
            self._app.prepare(ctx_id=ctx_id, det_size=det_size)

        logger.info(
            "FaceEngine ready  (pack=%s, det_size=%s, ctx_id=%d, root=%s)",
            model_pack, det_size, ctx_id, model_root,
        )

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def detect(self, img: np.ndarray) -> List[DetectedFace]:
        """
        Detect faces in a BGR image.

        Returns
        -------
        list[DetectedFace]
            Detected faces sorted by area (largest first).
        """
        raw = self._app.get(img)
        faces: list[DetectedFace] = []
        for f in raw:
            if f.det_score < self._det_thresh:
                continue
            bbox = BoundingBox(*f.bbox.tolist())

            # Crop aligned face chip for anti-spoofing
            aligned = self._crop_aligned(img, f)

            faces.append(DetectedFace(
                bbox=bbox,
                landmarks=f.kps if hasattr(f, "kps") else None,
                det_score=float(f.det_score),
                age=int(f.age) if hasattr(f, "age") and f.age is not None else None,
                gender=int(f.gender) if hasattr(f, "gender") and f.gender is not None else None,
                aligned_face=aligned,
            ))

        # Sort by face area descending
        faces.sort(key=lambda f: f.bbox.area, reverse=True)
        return faces

    def embed(self, img: np.ndarray, face: DetectedFace) -> FaceEmbedding:
        """
        Compute the 512-d ArcFace embedding for a detected face.

        Parameters
        ----------
        img : np.ndarray
            Original BGR image.
        face : DetectedFace
            A face previously returned by ``detect()``.

        Returns
        -------
        FaceEmbedding
        """
        raw_faces = self._app.get(img)

        # Match by bbox overlap
        best, best_iou = None, 0.0
        for rf in raw_faces:
            iou = self._iou(face.bbox, BoundingBox(*rf.bbox.tolist()))
            if iou > best_iou:
                best, best_iou = rf, iou

        if best is None or best_iou < 0.3:
            raise ValueError("Could not re-match the face for embedding.")

        vec = best.normed_embedding.astype(np.float32)
        quality = float(best.det_score)
        return FaceEmbedding(vector=vec, face=face, quality=quality)

    def detect_and_embed(
        self, img: np.ndarray
    ) -> List[FaceEmbedding]:
        """
        One-shot: detect every face and compute its embedding.

        More efficient than calling ``detect`` then ``embed`` separately
        because the ONNX session runs only once.
        """
        raw = self._app.get(img)
        results: list[FaceEmbedding] = []

        for f in raw:
            if f.det_score < self._det_thresh:
                continue

            bbox = BoundingBox(*f.bbox.tolist())
            aligned = self._crop_aligned(img, f)
            det_face = DetectedFace(
                bbox=bbox,
                landmarks=f.kps if hasattr(f, "kps") else None,
                det_score=float(f.det_score),
                age=int(f.age) if hasattr(f, "age") and f.age is not None else None,
                gender=int(f.gender) if hasattr(f, "gender") and f.gender is not None else None,
                aligned_face=aligned,
            )

            vec = f.normed_embedding.astype(np.float32)
            results.append(FaceEmbedding(
                vector=vec,
                face=det_face,
                quality=float(f.det_score),
            ))

        results.sort(key=lambda e: e.face.bbox.area, reverse=True)
        return results

    # ──────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────

    @staticmethod
    def _get_providers() -> list[str]:
        """Return available ONNX-Runtime execution providers."""
        try:
            import onnxruntime as ort
            return ort.get_available_providers()
        except ImportError:
            return ["CPUExecutionProvider"]

    @staticmethod
    def _crop_aligned(
        img: np.ndarray, raw_face
    ) -> np.ndarray:
        """Crop and resize face region for downstream analysis."""
        x1, y1, x2, y2 = raw_face.bbox.astype(int)
        h, w = img.shape[:2]
        # Pad bbox by 20 %
        pad_w = int((x2 - x1) * 0.2)
        pad_h = int((y2 - y1) * 0.2)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((112, 112, 3), dtype=np.uint8)
        return cv2.resize(crop, (112, 112))

    @staticmethod
    def _iou(a: BoundingBox, b: BoundingBox) -> float:
        """Intersection-over-Union of two bounding boxes."""
        xi1 = max(a.x1, b.x1)
        yi1 = max(a.y1, b.y1)
        xi2 = min(a.x2, b.x2)
        yi2 = min(a.y2, b.y2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = a.area + b.area - inter
        return inter / union if union > 0 else 0.0
