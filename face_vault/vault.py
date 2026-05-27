"""
face_vault.vault
────────────────
Main orchestrator — the public-facing class that ties together the
detection engine, anti-spoofing, and vector database.
"""
from __future__ import annotations
import hashlib, logging, shutil, time
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from ._types import (DetectMode, IdentifyResult, IdentifyWithResult,
                     IdentifyWithoutResult, MatchResult, RegisterResult,
                     SpoofVerdict, VerifyResult)
from .engine import FaceEngine
from .anti_spoof import AntiSpoof
from .database import FaceDatabase
from .utils import load_image, cosine_similarity, stamp_identify_image

logger = logging.getLogger(__name__)

# Image extensions we recognise when scanning folders
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _image_hash(path: Path) -> str:
    """SHA-256 hash of a file's contents (for duplicate detection)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_images(source: Union[str, Path, List[str], List[Path]]) -> List[Path]:
    """
    Normalise a reference_image argument into a sorted list of image Paths.

    Accepts:
      • a single file path
      • a folder path  (all images inside it)
      • a list of file paths
    """
    if isinstance(source, (list, tuple)):
        paths = [Path(p) for p in source]
    else:
        p = Path(source)
        if p.is_dir():
            paths = sorted(
                f for f in p.iterdir()
                if f.is_file() and f.suffix.lower() in _IMG_EXTS
            )
        elif p.is_file():
            paths = [p]
        else:
            raise FileNotFoundError(f"Image source not found: {p}")
    # Validate
    for fp in paths:
        if not fp.is_file():
            raise FileNotFoundError(f"Image not found: {fp}")
    if not paths:
        raise FileNotFoundError(f"No images found in: {source}")
    return paths


class FaceVault:
    """
    High-level face-recognition vault.

    Typical workflow
    ----------------
    >>> vault = FaceVault("my_faces.db")
    >>> vault.register(
    ...     full_name="Afiya Kelifa Ahimed",
    ...     user_code="OGH-00238",
    ...     reference_image="dataset/OGH-00238/",
    ... )
    >>> result = vault.identify("unknown.jpg")
    >>> print(result.matched, result.name, result.user_code)

    Parameters
    ----------
    db_path : str | Path
        SQLite DB file (created if absent).
    dataset_dir : str | Path
        Root folder for per-user reference images.  Defaults to
        ``dataset/`` next to the DB file.
    ctx_id : int
        ONNX execution provider.  -1 = CPU, 0 = first GPU.
    det_size : tuple
        Detector input resolution.
    det_thresh : float
        Minimum detection confidence.
    match_threshold : float
        Cosine-similarity threshold for a positive match.
    anti_spoof : bool
        Enable/disable the anti-spoofing checks.
    spoof_block : bool
        If True, refuse to register or match faces flagged as fake.
    model_pack : str
        InsightFace model pack name.
    """

    def __init__(
        self,
        db_path: str = "faces.db",
        dataset_dir: Optional[str] = None,
        ctx_id: int = -1,
        det_size: tuple = (640, 640),
        det_thresh: float = 0.5,
        match_threshold: float = 0.4,
        anti_spoof: bool = True,
        spoof_block: bool = True,
        model_pack: str = "buffalo_l",
        model_dir: Optional[Union[str, Path]] = None,
    ):
        self.engine = FaceEngine(
            ctx_id=ctx_id,
            det_size=det_size,
            det_thresh=det_thresh,
            model_pack=model_pack,
            model_dir=model_dir,
        )
        self.db = FaceDatabase(db_path=db_path)
        self.spoofer = AntiSpoof() if anti_spoof else None
        self.match_threshold = match_threshold
        self.spoof_block = spoof_block

        # Dataset root (defaults to <db_dir>/dataset/)
        if dataset_dir:
            self.dataset_dir = Path(dataset_dir)
        else:
            self.dataset_dir = Path(db_path).parent / "dataset"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        # Startup Banner
        from . import __version__
        banner = r"""
  ___          __   __         _ _   
 | __|_ _ __ __\ \ / /_ _ _  _| | |_ 
 | _/ _` / _/ -_) V / _` | || | |  _|
 |_|\__,_\__\___|\_/\__,_|\_,_|_|\__|
                                     
 Created at Think Lab
 Created by Abel Yohannes
""" + (
            f" Version:            {__version__}\n"
            f" DB Path:            {self.db.db_path.resolve()}\n"
            f" Dataset Directory:  {self.dataset_dir.resolve()}\n"
            f" Model Directory:    {self.engine.model_root / 'models' / model_pack}\n"
            f" Pip Install Command: pip install git+https://github.com/abeldirectory252/face_vault.git\n"
        )
        print(banner)

    # ──────────────────────────────────────────────
    # Registration
    # ──────────────────────────────────────────────

    def register(
        self,
        full_name: str,
        user_code: str,
        reference_image: Union[str, Path, List[str], List[Path], None] = None,
    ) -> RegisterResult:
        """
        Register one or more face images for an identity.

        Parameters
        ----------
        full_name : str
            Person's full name (e.g. "Afiya Kelifa Ahimed").
        user_code : str
            Unique user code (e.g. "OGH-00238").  This is the primary
            key used throughout the system.
        reference_image : str | Path | list | None
            Can be:
              • A **folder** path — all images inside are registered
                (e.g. ``"dataset/OGH-00238/"``).
              • A **single image** file path.
              • A **list** of image file paths.
              • **None** — looks for ``<dataset_dir>/<user_code>/``
                automatically.

        Raises
        ------
        FileNotFoundError
            If no images can be found.
        ValueError
            If a duplicate image (same SHA-256 hash) is given.

        Returns
        -------
        RegisterResult
            success, identity, images_registered, message
        """
        # ── Resolve images ─────────────────────────
        print(f"[FaceVault] Resolving reference images for '{full_name}' ({user_code})...")
        if reference_image is None:
            # Auto-discover from dataset/<user_code>/
            user_dir = self.dataset_dir / user_code
            if user_dir.is_dir():
                image_paths = _collect_images(user_dir)
            else:
                raise FileNotFoundError(
                    f"No reference_image provided and no folder found at "
                    f"'{user_dir}'.  Pass reference_image or create the folder."
                )
        else:
            image_paths = _collect_images(reference_image)

        print(f"[FaceVault] Found {len(image_paths)} source image(s) for registration.")

        # ── Ensure dataset folder exists ───────────
        user_dir = self.dataset_dir / user_code
        user_dir.mkdir(parents=True, exist_ok=True)

        # Copy images into dataset/ if they aren't already there
        normalised_paths: List[Path] = []
        for ip in image_paths:
            resolved = ip.resolve()
            if not str(resolved).startswith(str(user_dir.resolve())):
                # Copy to dataset/<user_code>/
                dest = user_dir / ip.name
                # Avoid overwriting with a different file
                if dest.exists() and _image_hash(dest) != _image_hash(resolved):
                    dest = user_dir / f"{ip.stem}_{_image_hash(resolved)[:8]}{ip.suffix}"
                if not dest.exists():
                    shutil.copy2(str(resolved), str(dest))
                normalised_paths.append(dest)
            else:
                normalised_paths.append(resolved)

        # ── Register each image ────────────────────
        ref_path = str(user_dir.resolve())
        registered = 0
        last_identity = None
        last_spoof = None
        errors: List[str] = []

        for img_path in normalised_paths:
            print(f"[FaceVault] Processing: {img_path.name}...")
            img_hash = _image_hash(img_path)

            # Check for duplicates
            if self.db.has_image_hash(img_hash):
                msg = f"Duplicate image skipped: {img_path.name}"
                print(f"  [!] {msg}")
                errors.append(msg)
                continue

            img = load_image(str(img_path))
            print(f"  [-] Running face detection & embedding extraction...")
            embeddings = self.engine.detect_and_embed(img)

            if not embeddings:
                msg = f"No face detected: {img_path.name}"
                print(f"  [!] {msg}")
                errors.append(msg)
                continue

            emb = embeddings[0]  # largest face
            print(f"  [+] Face found! Detection score (quality): {emb.quality:.4f}")

            # Anti-spoofing
            spoof_result = None
            if self.spoofer and emb.face.aligned_face is not None:
                print(f"  [-] Running anti-spoofing check...")
                spoof_result = self.spoofer.check(emb.face.aligned_face)
                last_spoof = spoof_result
                print(f"  [+] Anti-spoof verdict: {spoof_result.verdict.name} (score={spoof_result.score:.4f})")
                if self.spoof_block and spoof_result.verdict == SpoofVerdict.FAKE:
                    msg = f"Spoof rejected: {img_path.name} (score={spoof_result.score:.3f})"
                    print(f"  [!] {msg}")
                    errors.append(msg)
                    continue

            try:
                print(f"  [-] Saving vector embedding to database...")
                last_identity = self.db.register(
                    user_code=user_code,
                    name=full_name,
                    vector=emb.vector,
                    quality=emb.quality,
                    image_path=str(img_path.resolve()),
                    image_hash=img_hash,
                    reference_image_path=ref_path,
                )
                registered += 1
                print(f"  [✓] Successfully registered embedding from {img_path.name}!")
            except ValueError as e:
                print(f"  [!] Database error: {e}")
                errors.append(str(e))

        if registered == 0:
            msg = "No images registered."
            if errors:
                msg += " Errors: " + "; ".join(errors)
            print(f"[FaceVault] Registration Failed: {msg}")
            return RegisterResult(
                success=False,
                num_faces_found=0,
                images_registered=0,
                spoof_result=last_spoof,
                message=msg,
            )

        num_vectors = last_identity.num_vectors if last_identity else 0
        remaining = max(0, 10 - num_vectors)
        msg = (
            f"Registered '{full_name}' [{user_code}] — "
            f"{registered} image(s), {num_vectors} total vectors. "
            f"({remaining} remaining to reach 10-vector baseline)"
        )
        if errors:
            msg += f"  Warnings: {'; '.join(errors)}"

        print(f"[FaceVault] Registration Completed! {msg}")
        return RegisterResult(
            success=True,
            identity=last_identity,
            num_faces_found=registered,
            images_registered=registered,
            spoof_result=last_spoof,
            message=msg,
        )

    # ──────────────────────────────────────────────
    # Identification — dual mode
    # ──────────────────────────────────────────────

    def identify(
        self,
        image: Union[str, np.ndarray],
        user_code: Optional[str] = None,
        mode: DetectMode = DetectMode.DETECT_WITH_IDENTITY,
        top_k: int = 1,
        image_overlay: bool = False,
        reference_image: bool = False,
    ) -> Union[IdentifyResult, IdentifyWithResult, IdentifyWithoutResult]:
        """
        Identify a face in an image.

        Two modes
        ---------
        **DETECT_WITH_IDENTITY** (default)
            Search the *entire* database.
            → ``IdentifyWithResult(matched, name, user_code, similarity)``

        **DETECT_WITHOUT_IDENTITY**
            Requires ``user_code``.  Checks ONLY that user's stored
            vectors — much faster.
            → ``IdentifyWithoutResult(matched, similarity)``

        Parameters
        ----------
        image : str | ndarray
            File path or BGR image.
        user_code : str, optional
            Required when *mode* is ``DETECT_WITHOUT_IDENTITY``.
        mode : DetectMode
            Which identification strategy to use.
        top_k : int
            Number of results (only for DETECT_WITH_IDENTITY).
        image_overlay : bool
            If True, annotated image → ``result.image``.  Default False.
        reference_image : bool
            If True (+ ``image_overlay=True``), side-by-side composite
            with the stored reference photo.

        Examples
        --------
        >>> r = vault.identify("photo.jpg")
        >>> print(r.matched, r.name, r.user_code)

        >>> r = vault.identify("photo.jpg", image_overlay=True, reference_image=True)
        >>> cv2.imwrite("side_by_side.jpg", r.image)

        >>> r = vault.identify("photo.jpg", user_code="OGH-00238",
        ...                    mode=DetectMode.DETECT_WITHOUT_IDENTITY)
        >>> print(r.matched)
        """
        if mode == DetectMode.DETECT_WITHOUT_IDENTITY:
            return self._identify_without(image, user_code, image_overlay)
        else:
            return self._identify_with(image, top_k, image_overlay, reference_image)

    # ── DETECT_WITH_IDENTITY ─────────────────────

    def _identify_with(
        self, image: Union[str, np.ndarray], top_k: int = 1,
        image_overlay: bool = False, reference_image: bool = False,
    ) -> IdentifyWithResult:
        print("[FaceVault] Starting identification (Mode: DETECT_WITH_IDENTITY)...")
        t0 = time.perf_counter()
        img = load_image(image)
        
        print("  [-] Extracting face embedding...")
        embeddings = self.engine.detect_and_embed(img)

        if not embeddings:
            print("  [!] No faces detected in input image.")
            return IdentifyWithResult(
                matched=False, elapsed_ms=self._elapsed(t0),
            )

        emb = embeddings[0]
        print(f"  [+] Face found with detection score: {emb.quality:.4f}")

        # Anti-spoofing
        sr = self._check_spoof(emb)
        if sr and self.spoof_block and sr.verdict == SpoofVerdict.FAKE:
            print(f"  [🚨] Spoofing attempt blocked! Verdict: FAKE (score={sr.score:.4f})")
            self.db.log_access(None, 0, True, sr.score)
            elapsed = self._elapsed(t0)
            stamped = None
            if image_overlay:
                stamped = stamp_identify_image(
                    img, emb.face, sr,
                    detection_name="BLOCKED (Fake)",
                    detection_type="WITH_IDENTITY",
                    confidence=0.0, user_code=None, elapsed_ms=elapsed,
                )
            return IdentifyWithResult(
                matched=False, spoof_result=sr,
                elapsed_ms=elapsed, image=stamped,
            )

        if sr:
            print(f"  [✓] Anti-spoof check passed: {sr.verdict.name} (score={sr.score:.4f})")

        # Full DB search
        print("  [-] Searching full database for matching vectors...")
        hits = self.db.search(
            emb.vector, top_k=top_k, threshold=self.match_threshold,
        )

        if hits and hits[0].is_match:
            best = hits[0]
            print(f"  [✓] MATCH FOUND: '{best.identity.name}' [{best.identity.user_code}] (similarity={best.similarity:.4f})")
            self.db.log_access(
                best.identity.id, best.similarity, False,
                sr.score if sr else 0,
            )
            elapsed = self._elapsed(t0)
            stamped = None
            if image_overlay:
                ref_path = None
                if reference_image and best.identity.reference_image_path:
                    ref_path = best.identity.reference_image_path
                stamped = stamp_identify_image(
                    img, emb.face, sr,
                    detection_name=best.identity.name,
                    detection_type="WITH_IDENTITY",
                    confidence=best.similarity,
                    user_code=best.identity.user_code,
                    elapsed_ms=elapsed,
                    reference_image_path=ref_path,
                )
            return IdentifyWithResult(
                matched=True,
                name=best.identity.name,
                user_code=best.identity.user_code,
                similarity=best.similarity,
                spoof_result=sr,
                elapsed_ms=elapsed,
                image=stamped,
            )

        # No match
        print("  [!] No matching identity found in database.")
        self.db.log_access(None, 0, False, sr.score if sr else 0)
        elapsed = self._elapsed(t0)
        stamped = None
        if image_overlay:
            stamped = stamp_identify_image(
                img, emb.face, sr,
                detection_name="Unknown",
                detection_type="WITH_IDENTITY",
                confidence=hits[0].similarity if hits else 0,
                user_code=None, elapsed_ms=elapsed,
            )
        return IdentifyWithResult(
            matched=False, spoof_result=sr,
            similarity=hits[0].similarity if hits else 0,
            elapsed_ms=elapsed, image=stamped,
        )

    # ── DETECT_WITHOUT_IDENTITY ──────────────────

    def _identify_without(
        self, image: Union[str, np.ndarray], user_code: Optional[str],
        image_overlay: bool = False,
    ) -> IdentifyWithoutResult:
        if user_code is None:
            raise ValueError(
                "user_code is required for DETECT_WITHOUT_IDENTITY mode."
            )

        print(f"[FaceVault] Starting targeted verification for employee: {user_code}...")
        t0 = time.perf_counter()
        img = load_image(image)
        
        print("  [-] Extracting face embedding...")
        embeddings = self.engine.detect_and_embed(img)

        if not embeddings:
            print("  [!] No faces detected in input image.")
            return IdentifyWithoutResult(
                matched=False, elapsed_ms=self._elapsed(t0),
            )

        emb = embeddings[0]
        print(f"  [+] Face found with detection score: {emb.quality:.4f}")

        # Anti-spoofing
        sr = self._check_spoof(emb)
        if sr and self.spoof_block and sr.verdict == SpoofVerdict.FAKE:
            print(f"  [🚨] Spoofing attempt blocked! Verdict: FAKE (score={sr.score:.4f})")
            self.db.log_access(None, 0, True, sr.score)
            elapsed = self._elapsed(t0)
            stamped = None
            if image_overlay:
                stamped = stamp_identify_image(
                    img, emb.face, sr,
                    detection_name="BLOCKED (Fake)",
                    detection_type="WITHOUT_IDENTITY",
                    confidence=0.0, user_code=user_code, elapsed_ms=elapsed,
                )
            return IdentifyWithoutResult(
                matched=False, spoof_result=sr,
                elapsed_ms=elapsed, image=stamped,
            )

        if sr:
            print(f"  [✓] Anti-spoof check passed: {sr.verdict.name} (score={sr.score:.4f})")

        # Targeted search — only this user's vectors
        print(f"  [-] Comparing face embedding against baseline for {user_code}...")
        result = self.db.search_by_identity(
            emb.vector,
            user_code=user_code,
            threshold=self.match_threshold,
        )

        self.db.log_access(
            result.identity.id if result.is_match and result.identity else None,
            result.similarity, False,
            sr.score if sr else 0,
        )

        elapsed = self._elapsed(t0)
        stamped = None
        if image_overlay:
            display_name = "No Match"
            if result.is_match and result.identity:
                display_name = result.identity.name
            elif result.identity:
                display_name = f"NOT {result.identity.name}"
            stamped = stamp_identify_image(
                img, emb.face, sr,
                detection_name=display_name,
                detection_type="WITHOUT_IDENTITY",
                confidence=result.similarity,
                user_code=user_code, elapsed_ms=elapsed,
            )

        if result.is_match:
            print(f"  [✓] VERIFIED: Match found! similarity = {result.similarity:.4f} (threshold = {self.match_threshold})")
        else:
            print(f"  [!] MISMATCH: Similarity ({result.similarity:.4f}) is below threshold ({self.match_threshold})")

        return IdentifyWithoutResult(
            matched=result.is_match,
            similarity=result.similarity,
            spoof_result=sr,
            elapsed_ms=elapsed,
            image=stamped,
        )

    # ── Legacy multi-face identify ───────────────

    def identify_all(
        self, image: Union[str, np.ndarray], top_k: int = 1,
    ) -> IdentifyResult:
        """Identify *all* faces in an image."""
        t0 = time.perf_counter()
        img = load_image(image)
        embeddings = self.engine.detect_and_embed(img)

        matches, spoof_results = [], []
        for emb in embeddings:
            sr = self._check_spoof(emb)
            spoof_results.append(sr)

            if sr and self.spoof_block and sr.verdict == SpoofVerdict.FAKE:
                matches.append(MatchResult(
                    identity=None, distance=999, similarity=0, is_match=False,
                ))
                self.db.log_access(None, 0, True, sr.score)
                continue

            hits = self.db.search(
                emb.vector, top_k=top_k, threshold=self.match_threshold,
            )
            if hits:
                best = hits[0]
                matches.append(best)
                self.db.log_access(
                    best.identity.id if best.is_match else None,
                    best.similarity, False, sr.score if sr else 0,
                )
            else:
                matches.append(MatchResult(
                    identity=None, distance=999, similarity=0, is_match=False,
                ))

        return IdentifyResult(
            faces_found=len(embeddings), matches=matches,
            spoof_results=spoof_results,
            elapsed_ms=self._elapsed(t0),
            mode=DetectMode.DETECT_WITH_IDENTITY.value,
        )

    # ── Helpers ───────────────────────────────────

    def _check_spoof(self, emb):
        if self.spoofer and emb.face.aligned_face is not None:
            return self.spoofer.check(emb.face.aligned_face)
        return None

    @staticmethod
    def _elapsed(t0: float) -> float:
        return round((time.perf_counter() - t0) * 1000, 2)

    # ──────────────────────────────────────────────
    # Verification (1 : 1)
    # ──────────────────────────────────────────────

    def verify(self, image_a: Union[str, np.ndarray],
               image_b: Union[str, np.ndarray]) -> VerifyResult:
        """Verify whether two images depict the same person."""
        t0 = time.perf_counter()
        emb_a = self.engine.detect_and_embed(load_image(image_a))
        emb_b = self.engine.detect_and_embed(load_image(image_b))

        if not emb_a or not emb_b:
            return VerifyResult(is_same_person=False, similarity=0,
                                distance=999, elapsed_ms=0)

        sim = cosine_similarity(emb_a[0].vector, emb_b[0].vector)
        sr = None
        if self.spoofer and emb_b[0].face.aligned_face is not None:
            sr = self.spoofer.check(emb_b[0].face.aligned_face)

        elapsed = (time.perf_counter() - t0) * 1000
        return VerifyResult(
            is_same_person=sim >= self.match_threshold,
            similarity=round(sim, 4), distance=round(1 - sim, 4),
            spoof_result=sr, elapsed_ms=round(elapsed, 2),
        )

    # ──────────────────────────────────────────────
    # Management
    # ──────────────────────────────────────────────

    def list_identities(self):
        return self.db.list_identities()

    def remove_identity(self, user_code: str) -> bool:
        return self.db.remove_identity(user_code)

    def unregister(self, user_code: str, clean_images: bool = True) -> bool:
        """
        Unregister (remove) an identity and all its face vectors by user_code.
        
        Parameters
        ----------
        user_code : str
            Unique user code of the identity to remove.
        clean_images : bool, default True
            If True, also deletes the user's associated image directory from dataset_dir.
        """
        removed = self.db.remove_identity(user_code)
        if clean_images:
            user_dir = self.dataset_dir / user_code
            if user_dir.exists() and user_dir.is_dir():
                try:
                    shutil.rmtree(user_dir)
                    print(f"[FaceVault] Cleaned up dataset folder: {user_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up dataset folder {user_dir}: {e}")
                    print(f"[FaceVault] Warning: Failed to clean up dataset folder {user_dir}: {e}")
        return removed

    def deregister(self, user_code: str, clean_images: bool = True) -> bool:
        """Alias for unregister."""
        return self.unregister(user_code, clean_images=clean_images)

    def clear(self, clean_images: bool = True) -> None:
        """
        Wipe the database completely (identities, vectors, access logs)
        and optionally clean all files/folders inside the dataset directory.
        
        Parameters
        ----------
        clean_images : bool, default True
            If True, also deletes all files and directories inside dataset_dir.
        """
        print("[FaceVault] Clearing database...")
        self.db.clear()
        
        if clean_images and self.dataset_dir.exists():
            print(f"[FaceVault] Cleaning dataset directory: {self.dataset_dir}")
            for item in self.dataset_dir.iterdir():
                if item.is_dir():
                    try:
                        shutil.rmtree(item)
                    except Exception as e:
                        logger.warning(f"Failed to delete directory {item}: {e}")
                elif item.is_file():
                    try:
                        item.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete file {item}: {e}")
        print("[FaceVault] Clear operation completed successfully.")

    def clean(self, clean_images: bool = True) -> None:
        """Alias for clear."""
        self.clear(clean_images=clean_images)

    def stats(self):
        return self.db.get_stats()

    def close(self):
        self.db.close()
