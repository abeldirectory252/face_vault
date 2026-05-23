"""
face_vault.anti_spoof
─────────────────────
Multi-method anti-spoofing module.
Combines LBP texture, FFT frequency, YCrCb colour, Laplacian sharpness,
and moiré detection to distinguish live faces from spoofs.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional
import cv2
import numpy as np
from ._types import SpoofResult, SpoofVerdict

logger = logging.getLogger(__name__)

DEFAULT_METHODS = ["lbp", "frequency", "colour", "laplacian", "moire"]
DEFAULT_WEIGHTS = {"lbp": 0.30, "frequency": 0.25, "colour": 0.15, "laplacian": 0.15, "moire": 0.15}
DEFAULT_REAL_THRESHOLD = 0.55
DEFAULT_FAKE_THRESHOLD = 0.40


class AntiSpoof:
    _DISPATCH = {}

    def __init__(self, methods=None, weights=None, real_threshold=DEFAULT_REAL_THRESHOLD, fake_threshold=DEFAULT_FAKE_THRESHOLD):
        self.methods = methods or DEFAULT_METHODS
        self.weights = weights or DEFAULT_WEIGHTS
        self.real_threshold = real_threshold
        self.fake_threshold = fake_threshold

    def check(self, face_chip: np.ndarray) -> SpoofResult:
        if face_chip is None or face_chip.size == 0:
            return SpoofResult(verdict=SpoofVerdict.UNCERTAIN, score=0.5, details="Empty face chip")
        if face_chip.shape[:2] != (112, 112):
            face_chip = cv2.resize(face_chip, (112, 112))

        method_scores, total_weight, weighted_sum = {}, 0.0, 0.0
        for name in self.methods:
            fn = self._DISPATCH.get(name)
            if fn is None:
                continue
            try:
                score = float(np.clip(fn(face_chip), 0.0, 1.0))
            except Exception:
                score = 0.5
            method_scores[name] = score
            w = self.weights.get(name, 1.0)
            weighted_sum += score * w
            total_weight += w

        final = weighted_sum / total_weight if total_weight > 0 else 0.5
        if final >= self.real_threshold:
            verdict = SpoofVerdict.REAL
        elif final <= self.fake_threshold:
            verdict = SpoofVerdict.FAKE
        else:
            verdict = SpoofVerdict.UNCERTAIN

        return SpoofResult(verdict=verdict, score=round(final, 4),
                           method_scores={k: round(v, 4) for k, v in method_scores.items()},
                           details=f"Weighted ensemble: {final:.4f}")

    @staticmethod
    def _lbp_score(chip):
        gray = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)
        lbp = np.zeros_like(gray, dtype=np.uint8)
        for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
            lbp = (lbp << 1) | (shifted >= gray).astype(np.uint8)
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        hist = hist.astype(np.float64)
        hist /= hist.sum() + 1e-8
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return float(np.clip((entropy - 3.5) / 3.5, 0.0, 1.0))

    @staticmethod
    def _frequency_score(chip):
        gray = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY).astype(np.float64)
        fshift = np.fft.fftshift(np.fft.fft2(gray))
        magnitude = np.log1p(np.abs(fshift))
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        total = magnitude.sum()
        r = int(min(h, w) * 0.10)
        mask = np.ones_like(magnitude)
        cv2.circle(mask, (cx, cy), r, 0, -1)
        high_freq = (magnitude * mask).sum()
        ratio = high_freq / (total + 1e-8)
        if ratio > 0.88:
            return 0.3
        if ratio < 0.65:
            return 0.4
        return float(np.clip((ratio - 0.65) / 0.20, 0.0, 1.0))

    @staticmethod
    def _colour_score(chip):
        ycrcb = cv2.cvtColor(chip, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1].astype(np.float64)
        cb = ycrcb[:, :, 2].astype(np.float64)
        cr_mean, cr_std = cr.mean(), cr.std()
        cb_mean, cb_std = cb.mean(), cb.std()
        score = 0.5
        if 133 <= cr_mean <= 173 and 77 <= cb_mean <= 127:
            score += 0.3
        if 8 < cr_std < 25 and 5 < cb_std < 20:
            score += 0.2
        else:
            score -= 0.1
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _laplacian_score(chip):
        gray = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if var < 30:
            return 0.2
        if var > 1500:
            return 0.4
        return float(np.clip((var - 30) / 500, 0.3, 1.0))

    @staticmethod
    def _moire_score(chip):
        gray = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY).astype(np.float64)
        mag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
        h, w = mag.shape
        cv2.circle(mag, (w // 2, h // 2), 3, 0, -1)
        peak_ratio = np.sum(mag > 4 * np.median(mag)) / mag.size
        if peak_ratio > 0.015:
            return 0.2
        if peak_ratio > 0.008:
            return 0.5
        return 0.9


AntiSpoof._DISPATCH = {
    "lbp": AntiSpoof._lbp_score, "frequency": AntiSpoof._frequency_score,
    "colour": AntiSpoof._colour_score, "laplacian": AntiSpoof._laplacian_score,
    "moire": AntiSpoof._moire_score,
}
