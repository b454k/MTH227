"""Create GitHub-friendly split zip archives for local runtime data.

The project keeps large generated data out of normal git tracking. This script
packages those folders into split zip parts under ``data_archives/`` so another
machine can restore the exact same local layout with ``restore_data_archives.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = PROJECT_ROOT / "data_archives"
BUNDLE_STEM = "career_rag_data_bundle"
DEFAULT_INCLUDE_PATHS = (
    "data",
    "chroma_ai_impact",
    "chroma_research",
    "onet_sql",
)
DEFAULT_PART_SIZE_MIB = 45


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remove_existing_bundle() -> None:
    patterns = (
        f"{BUNDLE_STEM}.zip",
        f"{BUNDLE_STEM}.zip.part*",
        "manifest.json",
        f"{BUNDLE_STEM}.zip.tmp",
    )
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for path in ARCHIVE_DIR.glob(pattern):
            if path.is_file():
                path.unlink()


def iter_bundle_files(include_paths: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for include_path in include_paths:
        root = PROJECT_ROOT / include_path
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(path for path in root.rglob("*") if path.is_file())
        else:
            print(f"Skipping missing path: {include_path}")
    return sorted(files, key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())


def write_zip(zip_path: Path, files: list[Path]) -> tuple[int, int]:
    total_input_bytes = 0
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for file_path in files:
            relative_path = file_path.relative_to(PROJECT_ROOT).as_posix()
            archive.write(file_path, relative_path)
            total_input_bytes += file_path.stat().st_size
    return len(files), total_input_bytes


def split_zip(zip_path: Path, part_size_bytes: int) -> list[dict[str, str | int]]:
    if zip_path.stat().st_size <= part_size_bytes:
        return [
            {
                "name": zip_path.name,
                "size_bytes": zip_path.stat().st_size,
                "sha256": sha256_file(zip_path),
            }
        ]

    parts: list[dict[str, str | int]] = []
    with zip_path.open("rb") as source:
        index = 1
        while True:
            chunk = source.read(part_size_bytes)
            if not chunk:
                break
            part_path = zip_path.with_name(f"{zip_path.name}.part{index:03d}")
            part_path.write_bytes(chunk)
            parts.append(
                {
                    "name": part_path.name,
                    "size_bytes": part_path.stat().st_size,
                    "sha256": sha256_file(part_path),
                }
            )
            index += 1
    zip_path.unlink()
    return parts


def write_manifest(
    *,
    include_paths: tuple[str, ...],
    file_count: int,
    input_size_bytes: int,
    zip_size_bytes: int,
    zip_sha256: str,
    part_size_bytes: int,
    parts: list[dict[str, str | int]],
) -> None:
    manifest = {
        "bundle_name": f"{BUNDLE_STEM}.zip",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": PROJECT_ROOT.name,
        "included_paths": list(include_paths),
        "file_count": file_count,
        "input_size_bytes": input_size_bytes,
        "zip_size_bytes": zip_size_bytes,
        "zip_sha256": zip_sha256,
        "part_size_bytes": part_size_bytes,
        "parts": parts,
    }
    manifest_path = ARCHIVE_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part-size-mib",
        type=int,
        default=DEFAULT_PART_SIZE_MIB,
        help="Maximum size of each archive part in MiB.",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=list(DEFAULT_INCLUDE_PATHS),
        help="Project-relative files or folders to include.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    include_paths = tuple(args.include)
    part_size_bytes = args.part_size_mib * 1024 * 1024
    if part_size_bytes <= 0:
        raise ValueError("--part-size-mib must be positive.")

    remove_existing_bundle()
    files = iter_bundle_files(include_paths)
    if not files:
        raise RuntimeError("No files found to package.")

    zip_path = ARCHIVE_DIR / f"{BUNDLE_STEM}.zip"
    file_count, input_size_bytes = write_zip(zip_path, files)
    zip_size_bytes = zip_path.stat().st_size
    zip_sha256 = sha256_file(zip_path)
    parts = split_zip(zip_path, part_size_bytes)

    write_manifest(
        include_paths=include_paths,
        file_count=file_count,
        input_size_bytes=input_size_bytes,
        zip_size_bytes=zip_size_bytes,
        zip_sha256=zip_sha256,
        part_size_bytes=part_size_bytes,
        parts=parts,
    )

    print(f"Packaged {file_count} files from: {', '.join(include_paths)}")
    print(f"Original size: {input_size_bytes / 1024 / 1024:.2f} MiB")
    print(f"Archive size:  {zip_size_bytes / 1024 / 1024:.2f} MiB")
    print(f"Archive parts: {len(parts)} in {ARCHIVE_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
