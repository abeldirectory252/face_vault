"""
FaceVault Demo Script
─────────────────────
Run this script to see FaceVault in action.

Usage:
    python demo.py register --name "Alice Johnson" --code "OGH-00238" --images alice1.jpg alice2.jpg
    python demo.py register --name "Alice Johnson" --code "OGH-00238" --folder dataset/OGH-00238/
    python demo.py identify test_photo.jpg
    python demo.py identify test_photo.jpg --overlay --reference
    python demo.py check test_photo.jpg --code OGH-00238
    python demo.py verify alice.jpg bob.jpg
    python demo.py list
    python demo.py webcam
    python demo.py remove OGH-00238
"""
import argparse
import sys
import os

# Add parent dir to path so we can import face_vault
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from face_vault import FaceVault, DetectMode, draw_results


# ════════════════════════════════════════════════════════
# 1. INITIALISE
# ════════════════════════════════════════════════════════
def create_vault(db_path="demo_faces.db", dataset_dir="dataset", use_gpu=False):
    """Create a FaceVault instance."""
    vault = FaceVault(
        db_path=db_path,
        dataset_dir=dataset_dir,
        ctx_id=0 if use_gpu else -1,
        det_size=(640, 640),
        det_thresh=0.5,
        match_threshold=0.4,
        anti_spoof=True,
        spoof_block=True,
        model_pack="buffalo_l",
    )
    print(f"[✓] FaceVault ready  |  DB: {db_path}  |  Dataset: {dataset_dir}")
    return vault


# ════════════════════════════════════════════════════════
# 2. REGISTER FACES
# ════════════════════════════════════════════════════════
def demo_register(vault, full_name, user_code, reference_image):
    """Register images for an identity."""
    print(f"\n── Registering '{full_name}' [{user_code}] ──")
    result = vault.register(
        full_name=full_name,
        user_code=user_code,
        reference_image=reference_image,
    )
    status = "✓" if result.success else "✗"
    print(f"  [{status}] {result.message}")
    if result.spoof_result:
        sr = result.spoof_result
        print(f"      Spoof: {sr.verdict.value} (score={sr.score:.3f})")
    if result.identity:
        ident = result.identity
        print(f"      Code: {ident.user_code}")
        print(f"      Vectors: {ident.num_vectors}")
        print(f"      Ref path: {ident.reference_image_path}")
    print()


# ════════════════════════════════════════════════════════
# 3. IDENTIFY FACES — WITH_IDENTITY
# ════════════════════════════════════════════════════════
def demo_identify(vault, image_path, overlay=False, reference=False, show=True):
    """Identify faces in an image (searches entire DB)."""
    print(f"\n── Identifying: {image_path} ──")
    result = vault.identify(
        image_path,
        mode=DetectMode.DETECT_WITH_IDENTITY,
        image_overlay=overlay,
        reference_image=reference,
    )
    print(f"  Matched:    {result.matched}")
    print(f"  Name:       {result.name}")
    print(f"  User Code:  {result.user_code}")
    print(f"  Similarity: {result.similarity:.4f}")
    print(f"  Elapsed:    {result.elapsed_ms:.1f} ms")
    if result.spoof_result:
        sr = result.spoof_result
        print(f"  Spoof:      {sr.verdict.value} ({sr.score:.3f})")

    if result.image is not None:
        out_path = "result_identify.jpg"
        cv2.imwrite(out_path, result.image)
        print(f"  [✓] Saved: {out_path}")
        if show:
            cv2.imshow("FaceVault - Identify", result.image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return result


# ════════════════════════════════════════════════════════
# 4. QUICK CHECK — WITHOUT_IDENTITY
# ════════════════════════════════════════════════════════
def demo_check(vault, image_path, user_code, overlay=False, show=True):
    """Check if the face matches a specific user."""
    print(f"\n── Checking: is {image_path} == {user_code}? ──")
    result = vault.identify(
        image_path,
        user_code=user_code,
        mode=DetectMode.DETECT_WITHOUT_IDENTITY,
        image_overlay=overlay,
    )
    print(f"  Matched:    {result.matched}")
    print(f"  Similarity: {result.similarity:.4f}")
    print(f"  Elapsed:    {result.elapsed_ms:.1f} ms")
    if result.spoof_result:
        sr = result.spoof_result
        print(f"  Spoof:      {sr.verdict.value} ({sr.score:.3f})")

    if result.image is not None:
        out_path = "result_check.jpg"
        cv2.imwrite(out_path, result.image)
        print(f"  [✓] Saved: {out_path}")
        if show:
            cv2.imshow("FaceVault - Check", result.image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return result


# ════════════════════════════════════════════════════════
# 5. VERIFY TWO FACES
# ════════════════════════════════════════════════════════
def demo_verify(vault, img_a, img_b):
    """1:1 verification between two images."""
    print(f"\n── Verifying: {img_a} vs {img_b} ──")
    result = vault.verify(img_a, img_b)
    print(f"  Same person: {result.is_same_person}")
    print(f"  Similarity:  {result.similarity:.4f}")
    print(f"  Distance:    {result.distance:.4f}")
    print(f"  Elapsed:     {result.elapsed_ms:.1f} ms")
    if result.spoof_result:
        print(f"  Spoof:       {result.spoof_result.verdict.value}")
    return result


# ════════════════════════════════════════════════════════
# 6. LIST & STATS
# ════════════════════════════════════════════════════════
def demo_list(vault):
    """List all registered identities."""
    identities = vault.list_identities()
    stats = vault.stats()
    print(f"\n── Database Stats ──")
    print(f"  Identities:  {stats['identities']}")
    print(f"  Vectors:     {stats['vectors']}")
    print(f"  Access logs: {stats['access_logs']}")
    print(f"\n── Registered Identities ──")
    for ident in identities:
        print(f"  • [{ident.user_code}] {ident.name}"
              f"  ({ident.num_vectors} vectors, ref: {ident.reference_image_path})")
    if not identities:
        print("  (empty)")


# ════════════════════════════════════════════════════════
# 7. LIVE WEBCAM
# ════════════════════════════════════════════════════════
def demo_webcam(vault, camera_id=0):
    """Real-time face identification from webcam."""
    print("\n── Live webcam identification (press 'q' to quit) ──")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("  [✗] Cannot open camera")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 3 == 0:
            result = vault.identify_all(frame)
            embeddings = vault.engine.detect_and_embed(frame)
            faces = [e.face for e in embeddings]
            canvas = draw_results(frame, faces, result.matches, result.spoof_results)
        else:
            canvas = frame

        cv2.imshow("FaceVault - Live", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


# ════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="FaceVault Demo")
    parser.add_argument("--db", default="demo_faces.db", help="Database path")
    parser.add_argument("--dataset", default="dataset", help="Dataset directory")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")

    sub = parser.add_subparsers(dest="command")

    # register
    reg = sub.add_parser("register", help="Register face(s)")
    reg.add_argument("--name", required=True, help="Full name")
    reg.add_argument("--code", required=True, help="User code (e.g. OGH-00238)")
    reg.add_argument("--images", nargs="+", help="Image file path(s)")
    reg.add_argument("--folder", help="Folder of images")

    # identify (WITH_IDENTITY)
    idf = sub.add_parser("identify", help="Identify face (who is this?)")
    idf.add_argument("image", help="Image path")
    idf.add_argument("--overlay", action="store_true", help="Generate stamped image")
    idf.add_argument("--reference", action="store_true", help="Side-by-side reference")
    idf.add_argument("--no-show", action="store_true")

    # check (WITHOUT_IDENTITY)
    chk = sub.add_parser("check", help="Check face against a specific user")
    chk.add_argument("image", help="Image path")
    chk.add_argument("--code", required=True, help="User code to check against")
    chk.add_argument("--overlay", action="store_true")
    chk.add_argument("--no-show", action="store_true")

    # verify
    ver = sub.add_parser("verify", help="Verify two faces")
    ver.add_argument("image_a", help="First image")
    ver.add_argument("image_b", help="Second image")

    # list
    sub.add_parser("list", help="List identities & stats")

    # webcam
    sub.add_parser("webcam", help="Live webcam identification")

    # remove
    rem = sub.add_parser("remove", help="Remove identity")
    rem.add_argument("code", help="User code to remove")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    vault = create_vault(args.db, args.dataset, args.gpu)

    if args.command == "register":
        ref = args.folder if args.folder else (args.images if args.images else None)
        demo_register(vault, args.name, args.code, ref)

    elif args.command == "identify":
        demo_identify(vault, args.image, args.overlay, args.reference,
                      show=not args.no_show)

    elif args.command == "check":
        demo_check(vault, args.image, args.code, args.overlay,
                   show=not args.no_show)

    elif args.command == "verify":
        demo_verify(vault, args.image_a, args.image_b)

    elif args.command == "list":
        demo_list(vault)

    elif args.command == "webcam":
        demo_webcam(vault)

    elif args.command == "remove":
        ok = vault.remove_identity(args.code)
        print(f"{'✓' if ok else '✗'} Remove '{args.code}': {'done' if ok else 'not found'}")

    vault.close()


if __name__ == "__main__":
    main()
