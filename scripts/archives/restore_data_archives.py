"""Restore local runtime data from split zip archives in ``data_archives/``."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = PROJECT_ROOT / "data_archives"
MANIFEST_PATH = ARCHIVE_DIR / "manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Archive manifest not found: {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def verify_parts(manifest: dict) -> list[Path]:
    parts = []
    for part in manifest.get("parts", []):
        part_path = ARCHIVE_DIR / part["name"]
        if not part_path.exists():
            raise FileNotFoundError(f"Missing archive part: {part_path}")
        expected_size = int(part["size_bytes"])
        actual_size = part_path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"Archive part has wrong size: {part_path.name} "
                f"expected {expected_size}, got {actual_size}"
            )
        expected_sha = part.get("sha256")
        if expected_sha and sha256_file(part_path) != expected_sha:
            raise RuntimeError(f"Archive part failed sha256 check: {part_path.name}")
        parts.append(part_path)
    if not parts:
        raise RuntimeError("Manifest does not list any archive parts.")
    return parts


def build_zip_from_parts(parts: list[Path], manifest: dict, temp_dir: Path) -> Path:
    if len(parts) == 1 and parts[0].suffix == ".zip":
        zip_path = parts[0]
    else:
        zip_path = temp_dir / manifest.get("bundle_name", "career_rag_data_bundle.zip")
        with zip_path.open("wb") as output:
            for part_path in parts:
                with part_path.open("rb") as part_file:
                    shutil.copyfileobj(part_file, output, length=1024 * 1024)

    expected_sha = manifest.get("zip_sha256")
    if expected_sha and sha256_file(zip_path) != expected_sha:
        raise RuntimeError("Combined archive failed sha256 check.")
    return zip_path


def safe_target(member_name: str) -> Path:
    member_path = Path(member_name)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise RuntimeError(f"Unsafe archive path: {member_name}")
    return PROJECT_ROOT / member_path


def extract_zip(zip_path: Path, *, overwrite: bool) -> tuple[int, int]:
    extracted = 0
    skipped = 0
    with ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            target = safe_target(member.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not overwrite:
                skipped += 1
                continue
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            extracted += 1
    return extracted, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist locally.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest()
    parts = verify_parts(manifest)
    with tempfile.TemporaryDirectory(prefix="career_rag_restore_") as temp_name:
        zip_path = build_zip_from_parts(parts, manifest, Path(temp_name))
        extracted, skipped = extract_zip(zip_path, overwrite=args.overwrite)

    print(f"Archive parts verified: {len(parts)}")
    print(f"Files extracted: {extracted}")
    if skipped:
        print(f"Files skipped because they already existed: {skipped}")
        print("Run again with --overwrite if you want to replace existing local data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
