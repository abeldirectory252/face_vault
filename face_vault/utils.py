"""
face_vault.utils
────────────────
Image loading, stamp overlay, info-panel drawing, and misc helpers.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from ._types import (
    BoundingBox, DetectedFace, MatchResult,
    SpoofResult, SpoofVerdict,
)

logger = logging.getLogger(__name__)

# ── Stamp image directory (ships with the package) ────────
_STAMP_DIR = Path(__file__).parent.parent / "img"

_STAMP_PATHS = {
    SpoofVerdict.REAL:      _STAMP_DIR / "verified.png",
    SpoofVerdict.FAKE:      _STAMP_DIR / "fake.png",
    SpoofVerdict.UNCERTAIN: _STAMP_DIR / "uncertain.png",
}

# Cache loaded stamp images
_stamp_cache: Dict[SpoofVerdict, Optional[np.ndarray]] = {}


# ═══════════════════════════════════════════════════════════
#  Image loading
# ═══════════════════════════════════════════════════════════

def load_image(source: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Load an image from file path or URL. Returns BGR ndarray."""
    if isinstance(source, np.ndarray):
        return source
    path = str(source)
    if path.startswith(("http://", "https://")):
        import urllib.request
        resp = urllib.request.urlopen(path)
        arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


# ═══════════════════════════════════════════════════════════
#  Stamp overlay  (top-left corner)
# ═══════════════════════════════════════════════════════════

def _load_stamp(verdict: SpoofVerdict) -> Optional[np.ndarray]:
    """Load and cache a stamp PNG (with alpha channel)."""
    if verdict in _stamp_cache:
        return _stamp_cache[verdict]

    path = _STAMP_PATHS.get(verdict)
    if path is None or not path.exists():
        _stamp_cache[verdict] = None
        return None

    stamp = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # BGRA
    if stamp is None:
        _stamp_cache[verdict] = None
        return None

    # If no alpha channel, create one from white background
    if stamp.shape[2] == 3:
        gray = cv2.cvtColor(stamp, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        stamp = np.dstack([stamp, alpha])

    _stamp_cache[verdict] = stamp
    return stamp


def _overlay_stamp(
    canvas: np.ndarray,
    verdict: SpoofVerdict,
    scale: float = 0.20,
) -> np.ndarray:
    """
    Place the verdict stamp (VERIFIED / FAKE / UNCERTAINTY) in the
    top-left corner of the image with transparency.
    """
    stamp = _load_stamp(verdict)
    if stamp is None:
        return canvas

    h_img, w_img = canvas.shape[:2]

    # Scale stamp to ~20% of image width
    stamp_w = int(w_img * scale)
    aspect = stamp.shape[0] / stamp.shape[1]
    stamp_h = int(stamp_w * aspect)
    stamp_resized = cv2.resize(stamp, (stamp_w, stamp_h),
                               interpolation=cv2.INTER_AREA)

    # Position: top-left with 15px margin
    margin = 15
    x1, y1 = margin, margin
    x2, y2 = x1 + stamp_w, y1 + stamp_h

    # Clamp to image bounds
    if x2 > w_img:
        x2 = w_img
    if y2 > h_img:
        y2 = h_img

    crop_w = x2 - x1
    crop_h = y2 - y1
    stamp_crop = stamp_resized[:crop_h, :crop_w]

    # Alpha blending
    bgr = stamp_crop[:, :, :3]
    alpha = stamp_crop[:, :, 3].astype(np.float32) / 255.0

    roi = canvas[y1:y2, x1:x2]
    for c in range(3):
        roi[:, :, c] = (alpha * bgr[:, :, c] +
                        (1 - alpha) * roi[:, :, c]).astype(np.uint8)
    canvas[y1:y2, x1:x2] = roi
    return canvas


# ═══════════════════════════════════════════════════════════
#  Info panel  (bottom-right corner)
# ═══════════════════════════════════════════════════════════

def _draw_info_panel(
    canvas: np.ndarray,
    detection_name: str,
    detection_type: str,
    confidence: float,
    spoof_verdict: str,
    spoof_score: float,
    user_code: Optional[str],
    elapsed_ms: float,
) -> np.ndarray:
    """
    Draw a semi-transparent info panel in the bottom-right corner.

    Layout:
    ┌─────────────────────────┐
    │  FACEVAULT              │
    │  Name: Alice Johnson    │
    │  Code: OGH-00238        │
    │  Mode: WITH_IDENTITY    │
    │  Conf: 0.8742           │
    │  Spoof: REAL (0.72)     │
    │  Time: 84.3 ms          │
    │  2026-05-23 11:27:40    │
    └─────────────────────────┘
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    h_img, w_img = canvas.shape[:2]

    # Scale font relative to image size
    base = max(w_img, h_img)
    font_scale = max(0.40, base / 1600)
    thickness = max(1, int(base / 800))
    line_gap = int(22 * font_scale / 0.4)

    # Build info lines
    lines = [
        "FACEVAULT",
        f"Name: {detection_name}",
    ]
    if user_code is not None:
        lines.append(f"Code: {user_code}")
    lines += [
        f"Mode: {detection_type}",
        f"Conf: {confidence:.4f}",
        f"Spoof: {spoof_verdict} ({spoof_score:.2f})",
        f"Time: {elapsed_ms:.1f} ms",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ]

    # Measure panel size
    max_tw = 0
    total_th = 0
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_tw = max(max_tw, tw)
        total_th += line_gap

    pad = 12
    panel_w = max_tw + pad * 2
    panel_h = total_th + pad * 2 + 4

    # Position: bottom-right with margin
    margin = 15
    px1 = w_img - panel_w - margin
    py1 = h_img - panel_h - margin
    px2 = w_img - margin
    py2 = h_img - margin

    # Clamp
    px1 = max(0, px1)
    py1 = max(0, py1)

    # Semi-transparent dark background
    overlay = canvas.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    # Border
    border_colour = _verdict_colour(spoof_verdict)
    cv2.rectangle(canvas, (px1, py1), (px2, py2), border_colour, 2)

    # Draw text
    y_cursor = py1 + pad + line_gap - 4
    for i, line in enumerate(lines):
        colour = (255, 255, 255)
        f_scale = font_scale
        f_thick = thickness

        if i == 0:
            # Title line — slightly bigger, accent colour
            f_scale = font_scale * 1.1
            f_thick = thickness + 1
            colour = border_colour

        cv2.putText(canvas, line, (px1 + pad, y_cursor),
                    font, f_scale, colour, f_thick, cv2.LINE_AA)
        y_cursor += line_gap

    return canvas


def _verdict_colour(verdict_str: str) -> tuple:
    """Return BGR colour for a verdict string."""
    v = verdict_str.upper()
    if v == "REAL" or v == "VERIFIED":
        return (0, 200, 0)     # green
    elif v == "FAKE":
        return (0, 0, 230)     # red
    else:
        return (0, 165, 255)   # orange


# ═══════════════════════════════════════════════════════════
#  Main stamping function — produces result.image
# ═══════════════════════════════════════════════════════════

def stamp_identify_image(
    img: np.ndarray,
    face: Optional[DetectedFace],
    spoof_result: Optional[SpoofResult],
    detection_name: str = "Unknown",
    detection_type: str = "WITH_IDENTITY",
    confidence: float = 0.0,
    user_code: Optional[str] = None,
    elapsed_ms: float = 0.0,
    reference_image_path: Optional[str] = None,
) -> np.ndarray:
    """
    Produce a fully annotated image with:
      • Bounding box around the face (colour-coded by verdict)
      • Stamp overlay in top-left corner (VERIFIED / FAKE / UNCERTAINTY)
      • Info panel in bottom-right corner
      • Side-by-side reference image if *reference_image_path* is given

    When ``reference_image_path`` is provided the output is a composite:

        ┌────────────────┬────────────────┐
        │  INPUT IMAGE   │  REFERENCE     │
        │  (unknown)     │  (registered)  │
        │  + stamp       │  + name label  │
        │  + bbox        │                │
        │  + info panel  │                │
        └────────────────┴────────────────┘

    This lets a judge compare the probe face against the stored identity.
    """
    canvas = img.copy()

    # Determine verdict
    if spoof_result:
        verdict = spoof_result.verdict
        spoof_str = verdict.value.upper()
        spoof_score = spoof_result.score
    else:
        verdict = SpoofVerdict.REAL
        spoof_str = "N/A"
        spoof_score = 0.0

    # ── 1. Bounding box ──────────────────────────
    if face is not None:
        x1, y1, x2, y2 = face.bbox.to_int_tuple()
        colour = _verdict_colour(spoof_str)

        # Draw box with thicker line
        box_thick = max(2, int(max(canvas.shape[:2]) / 300))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, box_thick)

        # Small label above box
        label = detection_name
        font = cv2.FONT_HERSHEY_SIMPLEX
        f_scale = max(0.45, canvas.shape[1] / 1600)
        f_thick = max(1, int(canvas.shape[1] / 800))
        (tw, th), _ = cv2.getTextSize(label, font, f_scale, f_thick)
        cv2.rectangle(canvas, (x1, y1 - th - 10), (x1 + tw + 8, y1), colour, -1)
        cv2.putText(canvas, label, (x1 + 4, y1 - 5),
                    font, f_scale, (255, 255, 255), f_thick, cv2.LINE_AA)

    # ── 2. Stamp overlay (top-left) ──────────────
    canvas = _overlay_stamp(canvas, verdict, scale=0.22)

    # ── 3. Info panel (bottom-right) ─────────────
    canvas = _draw_info_panel(
        canvas,
        detection_name=detection_name,
        detection_type=detection_type,
        confidence=confidence,
        spoof_verdict=spoof_str,
        spoof_score=spoof_score,
        user_code=user_code,
        elapsed_ms=elapsed_ms,
    )

    # ── 4. Side-by-side with reference image ─────
    if reference_image_path is not None:
        canvas = _compose_side_by_side(canvas, reference_image_path, detection_name)

    return canvas


def _compose_side_by_side(
    probe: np.ndarray,
    reference_path: str,
    identity_name: str,
) -> np.ndarray:
    """
    Create a side-by-side composite:
      LEFT  = annotated probe image (already stamped)
      RIGHT = reference (registration) image with name label

    A vertical divider bar separates the two.
    """
    import os

    # Load reference image
    if not os.path.isfile(reference_path):
        logger.warning("Reference image not found: %s", reference_path)
        return probe

    ref = cv2.imread(reference_path)
    if ref is None:
        logger.warning("Cannot read reference image: %s", reference_path)
        return probe

    h_probe, w_probe = probe.shape[:2]

    # Resize reference to match probe height
    h_ref, w_ref = ref.shape[:2]
    scale = h_probe / h_ref
    new_w = int(w_ref * scale)
    ref_resized = cv2.resize(ref, (new_w, h_probe), interpolation=cv2.INTER_AREA)

    # ── Draw label on reference image ─────────
    font = cv2.FONT_HERSHEY_SIMPLEX
    base = max(h_probe, new_w)
    f_scale = max(0.5, base / 1200)
    f_thick = max(1, int(base / 600))

    # "REFERENCE: <name>" label at the top
    ref_label = f"REFERENCE: {identity_name}"
    (tw, th), _ = cv2.getTextSize(ref_label, font, f_scale, f_thick)
    bar_h = th + 20
    # Dark bar at top of reference
    cv2.rectangle(ref_resized, (0, 0), (new_w, bar_h), (40, 40, 40), -1)
    cv2.putText(ref_resized, ref_label, (10, th + 10),
                font, f_scale, (100, 255, 100), f_thick, cv2.LINE_AA)

    # ── "INPUT" label on probe ────────────────
    probe_label = "INPUT (probe)"
    (tw2, th2), _ = cv2.getTextSize(probe_label, font, f_scale, f_thick)
    # Dark bar at bottom of probe
    bar_y = h_probe - th2 - 20
    cv2.rectangle(probe, (0, bar_y), (tw2 + 20, h_probe), (40, 40, 40), -1)
    cv2.putText(probe, probe_label, (10, h_probe - 10),
                font, f_scale, (255, 200, 100), f_thick, cv2.LINE_AA)

    # ── Divider bar ───────────────────────────
    divider_w = 4
    divider = np.full((h_probe, divider_w, 3), (200, 200, 200), dtype=np.uint8)

    # Concatenate: probe | divider | reference
    composite = np.hstack([probe, divider, ref_resized])

    return composite


# ═══════════════════════════════════════════════════════════
#  Legacy draw_results (kept for backward compat)
# ═══════════════════════════════════════════════════════════

def draw_results(img: np.ndarray, faces: List[DetectedFace],
                 matches: Optional[List[MatchResult]] = None,
                 spoof_results: Optional[List[SpoofResult]] = None) -> np.ndarray:
    """Draw bounding boxes, names, and spoof status on an image."""
    canvas = img.copy()
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox.to_int_tuple()

        # Colour by spoof status
        colour = (0, 255, 0)  # green = real / unknown
        label_parts = []

        if spoof_results and i < len(spoof_results):
            sr = spoof_results[i]
            if sr.verdict == SpoofVerdict.FAKE:
                colour = (0, 0, 255)  # red
                label_parts.append(f"FAKE({sr.score:.2f})")
            elif sr.verdict == SpoofVerdict.UNCERTAIN:
                colour = (0, 165, 255)  # orange
                label_parts.append(f"UNSURE({sr.score:.2f})")
            else:
                label_parts.append(f"REAL({sr.score:.2f})")

        if matches and i < len(matches):
            m = matches[i]
            if m.is_match and m.identity:
                label_parts.insert(0, f"{m.identity.name} ({m.similarity:.2f})")
            else:
                label_parts.insert(0, "Unknown")

        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        label = " | ".join(label_parts) if label_parts else f"det:{face.det_score:.2f}"
        # Background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return canvas


# ═══════════════════════════════════════════════════════════
#  Math helpers
# ═══════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
