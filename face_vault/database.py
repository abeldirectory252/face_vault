"""
face_vault.database
───────────────────
FAISS vector index + SQLite metadata store for fast face search.
"""
from __future__ import annotations
import hashlib, json, logging, os, sqlite3, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from ._types import Identity, MatchResult

logger = logging.getLogger(__name__)

VECTOR_DIM = 512


class FaceDatabase:
    """
    Hybrid face-vector database.

    * **FAISS** flat-IP index for sub-millisecond cosine-similarity search.
    * **SQLite** for identity metadata and access logs.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite database file.
    dim : int
        Embedding dimensionality (512 for ArcFace).
    """

    def __init__(self, db_path: str = "faces.db", dim: int = VECTOR_DIM):
        self.db_path = Path(db_path)
        self.dim = dim
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

        if not HAS_FAISS:
            logger.warning("faiss not installed — falling back to numpy search")
        self._index: Optional[Any] = None
        self._id_map: List[int] = []
        self._rebuild_index()

    # ──────────────────────────────────────────────
    # Schema
    # ──────────────────────────────────────────────

    def _create_tables(self):
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS identities (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                user_code            TEXT UNIQUE NOT NULL,
                name                 TEXT NOT NULL,
                created_at           TEXT DEFAULT (datetime('now')),
                metadata             TEXT DEFAULT '{}',
                reference_image_path TEXT DEFAULT NULL
            );
            CREATE TABLE IF NOT EXISTS face_vectors (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                identity_id   INTEGER NOT NULL,
                vector        BLOB NOT NULL,
                quality_score REAL DEFAULT 0,
                image_hash    TEXT DEFAULT NULL,
                source_path   TEXT DEFAULT NULL,
                registered_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (identity_id) REFERENCES identities(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS access_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                identity_id  INTEGER,
                confidence   REAL,
                is_spoof     INTEGER DEFAULT 0,
                spoof_score  REAL,
                ts           TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (identity_id) REFERENCES identities(id)
            );
            CREATE INDEX IF NOT EXISTS idx_fv_identity ON face_vectors(identity_id);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_fv_hash ON face_vectors(image_hash);
        """)
        # Auto-migrate older schemas
        for col, sql in [
            ("user_code", "ALTER TABLE identities ADD COLUMN user_code TEXT"),
            ("reference_image_path", "ALTER TABLE identities ADD COLUMN reference_image_path TEXT"),
            ("image_hash", "ALTER TABLE face_vectors ADD COLUMN image_hash TEXT"),
            ("source_path", "ALTER TABLE face_vectors ADD COLUMN source_path TEXT"),
        ]:
            try:
                cur.execute(sql)
            except Exception:
                pass
        self._conn.commit()

    # ──────────────────────────────────────────────
    # FAISS index
    # ──────────────────────────────────────────────

    def _rebuild_index(self):
        rows = self._conn.execute(
            "SELECT id, vector FROM face_vectors ORDER BY id"
        ).fetchall()

        self._id_map = []
        if HAS_FAISS:
            self._index = faiss.IndexFlatIP(self.dim)
        else:
            self._index = None
            self._np_matrix = None

        if not rows:
            return

        vecs = []
        for row_id, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32).copy()
            if vec.shape[0] != self.dim:
                continue
            vecs.append(vec)
            self._id_map.append(row_id)

        if vecs:
            mat = np.vstack(vecs).astype(np.float32)
            if HAS_FAISS:
                faiss.normalize_L2(mat)
                self._index.add(mat)
            else:
                norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
                self._np_matrix = mat / norms

        logger.info("FAISS index rebuilt with %d vectors", len(vecs))

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    def _row_to_identity(self, row) -> Identity:
        """Build an Identity from a standard SELECT row."""
        # Expects: id, user_code, name, created_at, metadata,
        #          num_vectors, reference_image_path
        return Identity(
            id=row[0], user_code=row[1], name=row[2],
            created_at=row[3], metadata=json.loads(row[4] or "{}"),
            num_vectors=row[5], reference_image_path=row[6],
        )

    _IDENTITY_SELECT = """
        SELECT i.id, i.user_code, i.name, i.created_at, i.metadata,
               (SELECT COUNT(*) FROM face_vectors WHERE identity_id = i.id),
               i.reference_image_path
    """

    def has_image_hash(self, image_hash: str) -> bool:
        """Check if an image with this hash is already registered."""
        row = self._conn.execute(
            "SELECT 1 FROM face_vectors WHERE image_hash = ?", (image_hash,)
        ).fetchone()
        return row is not None

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def register(
        self,
        user_code: str,
        name: str,
        vector: np.ndarray,
        quality: float = 0.0,
        metadata: Optional[Dict] = None,
        image_path: Optional[str] = None,
        image_hash: Optional[str] = None,
        reference_image_path: Optional[str] = None,
    ) -> Identity:
        """Register a face vector for a user_code identity."""
        vec = vector.astype(np.float32)
        cur = self._conn.cursor()

        # Upsert identity
        cur.execute(
            "INSERT OR IGNORE INTO identities (user_code, name, metadata, reference_image_path) VALUES (?, ?, ?, ?)",
            (user_code, name, json.dumps(metadata or {}), reference_image_path),
        )
        # Update reference if provided on existing identity
        if reference_image_path is not None:
            cur.execute(
                "UPDATE identities SET reference_image_path = ? WHERE user_code = ? AND (reference_image_path IS NULL OR reference_image_path != ?)",
                (reference_image_path, user_code, reference_image_path),
            )
        cur.execute(
            self._IDENTITY_SELECT + " FROM identities i WHERE i.user_code = ?",
            (user_code,),
        )
        row = cur.fetchone()
        identity_id = row[0]

        # Check for duplicate image
        if image_hash and self.has_image_hash(image_hash):
            raise ValueError(
                f"Duplicate image: an image with this hash is already registered. "
                f"Hash: {image_hash}"
            )

        # Insert vector
        cur.execute(
            "INSERT INTO face_vectors (identity_id, vector, quality_score, image_hash, source_path) VALUES (?, ?, ?, ?, ?)",
            (identity_id, vec.tobytes(), quality, image_hash, image_path),
        )
        self._conn.commit()

        # Update FAISS incrementally
        normed = vec.copy().reshape(1, -1)
        if HAS_FAISS:
            faiss.normalize_L2(normed)
            self._index.add(normed)
        else:
            normed_np = normed / (np.linalg.norm(normed) + 1e-8)
            if self._np_matrix is not None:
                self._np_matrix = np.vstack([self._np_matrix, normed_np])
            else:
                self._np_matrix = normed_np
        self._id_map.append(cur.lastrowid)

        # Re-fetch to get updated counts
        cur.execute(
            self._IDENTITY_SELECT + " FROM identities i WHERE i.user_code = ?",
            (user_code,),
        )
        return self._row_to_identity(cur.fetchone())

    def search(self, vector: np.ndarray, top_k: int = 1,
               threshold: float = 0.4) -> List[MatchResult]:
        """Search the full FAISS index."""
        if not self._id_map:
            return []

        query = vector.astype(np.float32).reshape(1, -1)

        if HAS_FAISS:
            faiss.normalize_L2(query)
            scores, indices = self._index.search(query, min(top_k, len(self._id_map)))
            scores, indices = scores[0], indices[0]
        else:
            query_norm = query / (np.linalg.norm(query) + 1e-8)
            sims = (self._np_matrix @ query_norm.T).flatten()
            k = min(top_k, len(sims))
            indices = np.argsort(sims)[::-1][:k]
            scores = sims[indices]

        results = []
        for sim, idx in zip(scores, indices):
            if idx < 0:
                continue
            fv_id = self._id_map[idx]
            row = self._conn.execute(
                self._IDENTITY_SELECT + """
                FROM face_vectors fv
                JOIN identities i ON i.id = fv.identity_id
                WHERE fv.id = ?
                """, (fv_id,)
            ).fetchone()
            if row is None:
                continue

            identity = self._row_to_identity(row)
            similarity = float(sim)
            results.append(MatchResult(
                identity=identity, distance=1.0 - similarity,
                similarity=similarity, is_match=similarity >= threshold,
            ))
        return results

    def search_by_identity(
        self,
        vector: np.ndarray,
        user_code: Optional[str] = None,
        identity_id: Optional[int] = None,
        threshold: float = 0.4,
    ) -> MatchResult:
        """Search only a specific identity's vectors (fast path)."""
        if user_code is not None:
            row = self._conn.execute(
                self._IDENTITY_SELECT + " FROM identities i WHERE i.user_code = ?",
                (user_code,),
            ).fetchone()
        elif identity_id is not None:
            row = self._conn.execute(
                self._IDENTITY_SELECT + " FROM identities i WHERE i.id = ?",
                (identity_id,),
            ).fetchone()
        else:
            raise ValueError("Provide user_code or identity_id")

        if row is None:
            return MatchResult(identity=None, distance=999, similarity=0, is_match=False)

        identity = self._row_to_identity(row)

        vrows = self._conn.execute(
            "SELECT vector FROM face_vectors WHERE identity_id = ?", (identity.id,)
        ).fetchall()

        if not vrows:
            return MatchResult(identity=identity, distance=999, similarity=0, is_match=False)

        query = vector.astype(np.float32).flatten()
        query_norm = query / (np.linalg.norm(query) + 1e-8)

        best_sim = -1.0
        for (blob,) in vrows:
            stored = np.frombuffer(blob, dtype=np.float32)
            stored_norm = stored / (np.linalg.norm(stored) + 1e-8)
            sim = float(np.dot(query_norm, stored_norm))
            if sim > best_sim:
                best_sim = sim

        return MatchResult(
            identity=identity,
            distance=round(1.0 - best_sim, 4),
            similarity=round(best_sim, 4),
            is_match=best_sim >= threshold,
        )

    def get_identity_by_code(self, user_code: str) -> Optional[Identity]:
        """Look up an identity by user_code."""
        row = self._conn.execute(
            self._IDENTITY_SELECT + " FROM identities i WHERE i.user_code = ?",
            (user_code,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_identity(row)

    def remove_identity(self, user_code: str) -> bool:
        """Remove an identity and all its vectors by user_code."""
        cur = self._conn.cursor()
        cur.execute("SELECT id FROM identities WHERE user_code = ?", (user_code,))
        row = cur.fetchone()
        if row is None:
            return False
        cur.execute("DELETE FROM face_vectors WHERE identity_id = ?", (row[0],))
        cur.execute("DELETE FROM identities WHERE id = ?", (row[0],))
        self._conn.commit()
        self._rebuild_index()
        return True

    def list_identities(self) -> List[Identity]:
        """List all registered identities."""
        rows = self._conn.execute(
            self._IDENTITY_SELECT + " FROM identities i ORDER BY i.name"
        ).fetchall()
        return [self._row_to_identity(r) for r in rows]

    def log_access(self, identity_id: Optional[int], confidence: float,
                   is_spoof: bool, spoof_score: float):
        self._conn.execute(
            "INSERT INTO access_log (identity_id, confidence, is_spoof, spoof_score) VALUES (?,?,?,?)",
            (identity_id, confidence, int(is_spoof), spoof_score),
        )
        self._conn.commit()

    def get_stats(self) -> Dict[str, int]:
        ids = self._conn.execute("SELECT COUNT(*) FROM identities").fetchone()[0]
        vecs = self._conn.execute("SELECT COUNT(*) FROM face_vectors").fetchone()[0]
        logs = self._conn.execute("SELECT COUNT(*) FROM access_log").fetchone()[0]
        return {"identities": ids, "vectors": vecs, "access_logs": logs}

    def close(self):
        self._conn.close()

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            pass
