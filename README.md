# FaceVault 🔐

**Fast face recognition with anti-spoofing and vector database.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abeldirectory252/face_vault/blob/main/notebook/face_vault_colab.ipynb)

FaceVault is a library-structured Python package for face detection, recognition, and anti-spoofing. It wraps InsightFace (RetinaFace + ArcFace) with a FAISS vector database and a 5-method anti-spoofing ensemble into a clean, production-ready API.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Face Detection** | RetinaFace via InsightFace — fast & accurate |
| 🧬 **512-d Embeddings** | ArcFace model, L2-normalised for cosine similarity |
| ⚡ **FAISS Vector DB** | Sub-millisecond search across thousands of identities |
| 🛡️ **Anti-Spoofing** | 5-method ensemble: LBP, FFT, YCrCb, Laplacian, Moiré |
| 🎯 **Dual Detect Mode** | `WITH_IDENTITY` (who is this?) / `WITHOUT_IDENTITY` (is this user X?) |
| 🖼️ **Image Overlay** | Stamped output with VERIFIED/FAKE/UNCERTAIN + info panel |
| 👥 **Side-by-Side** | Reference image comparison for human judges |
| 📁 **Folder Registration** | Register a whole folder of images in one call |
| 🔑 **User Code** | User-provided unique codes (e.g. `OGH-00238`) as primary keys |
| 🚫 **Duplicate Detection** | SHA-256 hash prevents registering the same image twice |
| 📝 **Access Logging** | Every identification attempt logged to SQLite |

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/abeldirectory252/face_vault.git
cd face_vault

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### Requirements

- Python ≥ 3.9
- `insightface` ≥ 0.7
- `onnxruntime` ≥ 1.14
- `opencv-python` ≥ 4.8
- `numpy` ≥ 1.24
- `scipy` ≥ 1.10
- `faiss-cpu` ≥ 1.7

---

## 🚀 Quick Start

```python
from face_vault import FaceVault, DetectMode

# 1. Create vault
vault = FaceVault("my_faces.db", dataset_dir="dataset")

# 2. Register identities
vault.register(
    full_name="Afiya Kelifa Ahimed",
    user_code="OGH-00238",
    reference_image="dataset/OGH-00238/",   # folder of images
)

# 3. Identify — who is this person?
result = vault.identify("photos/unknown.jpg")
print(result.matched, result.name, result.user_code)
# True, "Afiya Kelifa Ahimed", "OGH-00238"

# 4. Verify — is this person OGH-00238?
result = vault.identify(
    "photos/unknown.jpg",
    user_code="OGH-00238",
    mode=DetectMode.DETECT_WITHOUT_IDENTITY,
)
print(result.matched)  # True
```

---

## 📁 Registration

### API

```python
vault.register(
    full_name="Afiya Kelifa Ahimed",
    user_code="OGH-00238",
    reference_image=...,   # see below
)
```

### `reference_image` accepts

| Input | Behaviour |
|-------|-----------|
| **Folder path** `"dataset/OGH-00238/"` | All images in the folder are registered |
| **Single file** `"photo.jpg"` | One image registered |
| **List of files** `["a.jpg", "b.jpg"]` | Multiple images registered |
| **None** | Auto-discovers `<dataset_dir>/<user_code>/` |

### What happens automatically

1. Images are **copied** to `dataset/<user_code>/` if not already there.
2. Each image gets a **SHA-256 hash** — duplicates are rejected.
3. Anti-spoofing runs on every image — fakes are blocked (when `spoof_block=True`).
4. The **reference image path** is stored in the DB for side-by-side comparison.

```python
# Auto-discover from dataset/<user_code>/
vault.register(full_name="Afiya Kelifa Ahimed", user_code="OGH-00238")

# Single image — gets copied to dataset/OGH-00238/
vault.register(
    full_name="Afiya Kelifa Ahimed",
    user_code="OGH-00238",
    reference_image="photos/afiya_face.jpg",
)

# Multiple images
vault.register(
    full_name="Afiya Kelifa Ahimed",
    user_code="OGH-00238",
    reference_image=["photo1.jpg", "photo2.jpg", "photo3.jpg"],
)
```

---

## 🎯 Detection Modes

### `DETECT_WITH_IDENTITY` — "Who is this?"

Searches the **entire** FAISS index. Returns the matched person's name and user code.

```python
r = vault.identify("photo.jpg")
print(r.matched, r.name, r.user_code, r.similarity)
# True, "Afiya Kelifa Ahimed", "OGH-00238", 0.8742
```

### `DETECT_WITHOUT_IDENTITY` — "Is this OGH-00238?"

Checks **only** the given user's vectors. Skips FAISS — much faster.

```python
r = vault.identify(
    "photo.jpg",
    user_code="OGH-00238",
    mode=DetectMode.DETECT_WITHOUT_IDENTITY,
)
print(r.matched, r.similarity)
# True, 0.8431
```

---

## 🖼️ Image Overlay

Generate annotated images with stamps, bounding boxes, and info panels.
**Off by default** — pass `image_overlay=True` to enable.

```python
# Stamped image only
r = vault.identify("photo.jpg", image_overlay=True)
cv2.imwrite("result.jpg", r.image)

# Side-by-side with reference photo (for human judges)
r = vault.identify("photo.jpg", image_overlay=True, reference_image=True)
cv2.imwrite("side_by_side.jpg", r.image)
```

**Output layout with `reference_image=True`:**

```
┌──────────────────────┬──────────────────────┐
│  INPUT (probe)       │  REFERENCE: Afiya    │
│  + VERIFIED stamp    │  (registration photo)│
│  + bbox + info panel │                      │
└──────────────────────┴──────────────────────┘
```

**Flag combinations:**

| `image_overlay` | `reference_image` | `result.image` |
|---|---|---|
| `False` (default) | — | `None` |
| `True` | `False` | Stamped image only |
| `True` | `True` | Side-by-side composite |

---

## 🛡️ Anti-Spoofing

5-method ensemble to detect fake faces:

| Method | Weight | Detects |
|--------|--------|---------|
| LBP Texture | 30% | Printed photos |
| FFT Frequency | 25% | Screen replay |
| YCrCb Colour | 15% | Non-skin colour |
| Laplacian Variance | 15% | Blur anomalies |
| Moiré Detection | 15% | LCD patterns |

```python
vault = FaceVault("faces.db", anti_spoof=True, spoof_block=True)

result = vault.register(
    full_name="Attacker",
    user_code="FAKE-001",
    reference_image="fake_photo.jpg",
)
print(result.success)  # False
print(result.message)  # "Spoof rejected: ..."
```

---

## 🗄️ Database Schema

```
identities              face_vectors            access_log
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ id (PK)          │    │ id (PK)          │    │ id (PK)          │
│ user_code (UQ)   │◄───│ identity_id (FK) │    │ identity_id (FK) │
│ name             │    │ vector (BLOB)    │    │ confidence       │
│ created_at       │    │ quality_score    │    │ is_spoof         │
│ metadata (JSON)  │    │ image_hash (UQ)  │    │ spoof_score      │
│ reference_image  │    │ source_path      │    │ ts               │
│   _path          │    │ registered_at    │    └──────────────────┘
└──────────────────┘    └──────────────────┘
```

---

## 📁 Project Structure

```
face_vault/
├── face_vault/
│   ├── __init__.py         # Public API exports
│   ├── _types.py           # Dataclasses & enums
│   ├── engine.py           # RetinaFace + ArcFace engine
│   ├── anti_spoof.py       # 5-method ensemble
│   ├── database.py         # FAISS + SQLite
│   ├── vault.py            # Main FaceVault orchestrator
│   └── utils.py            # Image loading, stamping, drawing
├── dataset/                # Per-user reference images
│   ├── OGH-00013/          # Sumaya Kedir Jemel (8 images)
│   ├── OGH-00044/          # Meklit Ayele Woldeyes (8 images)
│   ├── OGH-00238/          # Afiya Kelifa Ahimed (7 images)
│   └── Test/               # 10 unknown test images
│       ├── unk1.png
│       ├── unk2.png
│       └── ...
├── img/                    # Stamp overlay images
│   ├── verified.png
│   ├── fake.png
│   └── uncertain.png
├── notebook/
│   └── face_vault_colab.ipynb
├── demo.py
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

---

## ⚡ Performance

| Operation | CPU | GPU |
|-----------|-----|-----|
| Detection + Embedding | ~80 ms | ~15 ms |
| FAISS search (1K vectors) | < 1 ms | < 1 ms |
| Anti-spoof ensemble | ~5 ms | ~5 ms |
| Image overlay | ~3 ms | ~3 ms |
| **Total identify** | **~85 ms** | **~20 ms** |

---

## 🧪 Colab

https://github.com/abeldirectory252/face_vault.git

Try the interactive demo:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abeldirectory252/face_vault/blob/main/notebook/face_vault_colab.ipynb)

---

## 📄 License

MIT — see [LICENSE](LICENSE).
