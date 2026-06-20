#!/usr/bin/env python3
"""Download source URLs and write a source-policy manifest for AI exposure work."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

try:
    from career_rag.ai_exposure_utils import (
        PROJECT_ROOT,
        created_at,
        detect_source_policy,
        one_line,
        resolve_project_path,
        sha256_file,
        short_hash,
        slugify,
        write_jsonl,
    )
except ImportError:  # Allows: py career_rag/download_research_sources.py
    from ai_exposure_utils import (  # type: ignore
        PROJECT_ROOT,
        created_at,
        detect_source_policy,
        one_line,
        resolve_project_path,
        sha256_file,
        short_hash,
        slugify,
        write_jsonl,
    )


DEFAULT_SOURCE_FILE = PROJECT_ROOT / "data" / "research" / "source_urls.txt"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "research_sources" / "raw"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "research_sources" / "source_manifest.jsonl"

HEADERS = {"User-Agent": "career-rag academic source collector"}
DOWNLOADABLE_EXTENSIONS = {".pdf", ".html", ".htm", ".txt", ".csv", ".json", ".jsonl", ".parquet"}


def read_urls(path: Path) -> list[str]:
    """Read non-empty, non-comment URL lines."""
    urls: list[str] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    return urls


def resolve_source_file(path_text: str | None) -> Path:
    """Resolve common source file spellings without surprising failures."""
    candidates: list[Path] = []
    if path_text:
        requested = resolve_project_path(path_text)
        candidates.append(requested)
        if requested.name == "source_url.txt":
            candidates.append(requested.with_name("source_urls.txt"))
    candidates.extend(
        [
            PROJECT_ROOT / "source_url.txt",
            PROJECT_ROOT / "source_urls.txt",
            DEFAULT_SOURCE_FILE,
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Source URL file not found. Checked source_url.txt, source_urls.txt, "
        f"and {DEFAULT_SOURCE_FILE}."
    )


def is_pdf_response(response: requests.Response) -> bool:
    """Return True when a response body looks like a PDF."""
    content_type = response.headers.get("content-type", "").lower()
    return "application/pdf" in content_type or response.content[:4] == b"%PDF"


def infer_source_type(url: str, response: requests.Response | None, local_path: str = "") -> str:
    """Infer source type for the manifest."""
    url_lower = url.lower()
    local_lower = local_path.lower()
    if "huggingface.co/datasets" in url_lower or "github.com" in url_lower:
        return "dataset"
    if local_lower.endswith(".pdf"):
        return "pdf"
    if response is not None and is_pdf_response(response):
        return "pdf"
    if local_lower.endswith((".html", ".htm", ".txt")):
        return "html"
    if local_lower.endswith((".csv", ".json", ".jsonl", ".parquet")):
        return "dataset"
    content_type = response.headers.get("content-type", "").lower() if response else ""
    if "html" in content_type:
        return "html"
    return "unknown"


def arxiv_pdf_url(url: str) -> str | None:
    """Convert arXiv abs links to direct PDF links."""
    match = re.search(r"arxiv\.org/abs/([^?#]+)", url)
    if match:
        return f"https://arxiv.org/pdf/{match.group(1)}.pdf"
    return None


def nber_pdf_url(url: str) -> str | None:
    """Prefer direct NBER PDF URLs for working paper pages."""
    match = re.search(r"nber\.org/papers/(w\d+)", url)
    if match:
        wp = match.group(1)
        return f"https://www.nber.org/system/files/working_papers/{wp}/{wp}.pdf"
    return None


def find_pdf_link(html: str, base_url: str) -> str | None:
    """Find a likely PDF link from an HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"])
        text = anchor.get_text(" ", strip=True).lower()
        if href.lower().endswith(".pdf") or "pdf" in text:
            return urljoin(base_url, href)
    return None


def html_title(html: str) -> str:
    """Extract a simple HTML title."""
    soup = BeautifulSoup(html, "html.parser")
    if soup.title:
        return one_line(soup.title.get_text(" ", strip=True))
    heading = soup.find(["h1", "h2"])
    if heading:
        return one_line(heading.get_text(" ", strip=True))
    return ""


def pdf_title(path: Path) -> str:
    """Extract a title from PDF metadata or first-page text when available."""
    try:
        import fitz
    except ImportError:
        return ""

    try:
        doc = fitz.open(path)
        try:
            metadata_title = one_line((doc.metadata or {}).get("title"))
            if metadata_title and metadata_title.lower() not in {"untitled", "anonymous"}:
                return metadata_title[:300]
            if doc.page_count:
                lines = [
                    one_line(line)
                    for line in doc[0].get_text("text").splitlines()
                    if one_line(line)
                ]
                return " ".join(lines[:3])[:300]
        finally:
            doc.close()
    except Exception:
        return ""
    return ""


def make_source_id(url: str, title: str, used_ids: set[str]) -> str:
    """Create a stable source_id, honoring special statistics sources."""
    policy = detect_source_policy(url, title)
    preferred = policy.get("source_id")
    if preferred:
        base_id = preferred
    else:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "").split(".")[0] or "source"
        path_slug = slugify(parsed.path.strip("/") or title or "homepage", 60)
        base_id = slugify(f"{domain}_{path_slug}_{short_hash(url, 8)}", 90)

    source_id = str(base_id)
    counter = 2
    while source_id in used_ids:
        source_id = f"{base_id}_{counter}"
        counter += 1
    used_ids.add(source_id)
    return source_id


def extension_for_response(url: str, response: requests.Response) -> str:
    """Pick a file extension for a response."""
    if is_pdf_response(response):
        return ".pdf"
    suffix = Path(urlparse(response.url or url).path).suffix.lower()
    if suffix in DOWNLOADABLE_EXTENSIONS:
        return suffix
    content_type = response.headers.get("content-type", "").lower()
    if "html" in content_type:
        return ".html"
    if "json" in content_type:
        return ".json"
    if "csv" in content_type:
        return ".csv"
    return ".bin"


def save_response(path: Path, response: requests.Response) -> None:
    """Save a response as bytes or text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".html", ".htm", ".txt", ".json", ".jsonl", ".csv"}:
        path.write_text(response.text, encoding="utf-8", errors="replace")
    else:
        path.write_bytes(response.content)


def base_manifest_row(url: str) -> dict[str, Any]:
    """Build a manifest row before download details are known."""
    return {
        "source_id": "",
        "source_name": "",
        "source_url": url,
        "final_url": "",
        "source_type": "unknown",
        "local_path": "",
        "title": "",
        "download_status": "not_started",
        "error_message": "",
        "created_at": created_at(),
        "statistics_allowed": False,
        "tables_allowed": False,
        "images_allowed": False,
        "chunks_allowed": True,
        "usage": "inference_methodology_caveat_only",
        "sha256": "",
        "content_type": "",
    }


def enrich_policy(row: dict[str, Any], used_ids: set[str]) -> dict[str, Any]:
    """Apply source-policy labels and fill source_id/source_name."""
    policy = detect_source_policy(row.get("source_url", ""), row.get("title", ""))
    source_id = make_source_id(row.get("source_url", ""), row.get("title", ""), used_ids)
    row.update(policy)
    row["source_id"] = source_id
    if not row.get("source_name"):
        row["source_name"] = row.get("title") or source_id
    return row


def request_first_success(urls: list[str], timeout: int) -> requests.Response | None:
    """Try candidate URLs and return the first successful response."""
    for url in urls:
        response = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if 200 <= response.status_code < 300:
            return response
    return None


def download_one(url: str, raw_dir: Path, used_ids: set[str], timeout: int) -> dict[str, Any]:
    """Download one URL and return its manifest row."""
    row = base_manifest_row(url)
    try:
        candidates = [candidate for candidate in [nber_pdf_url(url), arxiv_pdf_url(url), url] if candidate]
        response = request_first_success(candidates, timeout=timeout)
        if response is None:
            row["download_status"] = "failed"
            row["error_message"] = "No candidate URL returned HTTP 2xx."
            return enrich_policy(row, used_ids)

        row["final_url"] = response.url
        row["content_type"] = response.headers.get("content-type", "")

        if is_pdf_response(response):
            row["title"] = ""
            row = enrich_policy(row, used_ids)
            local_path = raw_dir / f"{row['source_id']}.pdf"
            save_response(local_path, response)
            row["title"] = pdf_title(local_path) or row["title"]
            policy = detect_source_policy(url, row["title"])
            row.update({key: value for key, value in policy.items() if key != "source_id"})
            row["source_name"] = row.get("source_name") or row.get("title") or row["source_id"]
            row["local_path"] = str(local_path.relative_to(PROJECT_ROOT))
            row["source_type"] = "pdf"
            row["download_status"] = "downloaded"
            row["sha256"] = sha256_file(local_path)
            return row

        suffix = extension_for_response(url, response)
        if suffix in {".html", ".htm"}:
            title = html_title(response.text)
            pdf_link = find_pdf_link(response.text, response.url)
            if pdf_link:
                pdf_response = requests.get(
                    pdf_link,
                    headers=HEADERS,
                    timeout=timeout,
                    allow_redirects=True,
                )
                if 200 <= pdf_response.status_code < 300 and is_pdf_response(pdf_response):
                    row["title"] = title
                    row = enrich_policy(row, used_ids)
                    local_path = raw_dir / f"{row['source_id']}.pdf"
                    save_response(local_path, pdf_response)
                    row["final_url"] = pdf_response.url
                    row["content_type"] = pdf_response.headers.get("content-type", "")
                    row["title"] = pdf_title(local_path) or title
                    policy = detect_source_policy(url, row["title"])
                    row.update({key: value for key, value in policy.items() if key != "source_id"})
                    row["source_name"] = row.get("source_name") or row.get("title") or row["source_id"]
                    row["local_path"] = str(local_path.relative_to(PROJECT_ROOT))
                    row["source_type"] = "pdf"
                    row["download_status"] = "downloaded_pdf_from_page"
                    row["sha256"] = sha256_file(local_path)
                    return row

            row["title"] = title
            row = enrich_policy(row, used_ids)
            local_path = raw_dir / f"{row['source_id']}.html"
            save_response(local_path, response)
            row["local_path"] = str(local_path.relative_to(PROJECT_ROOT))
            row["source_type"] = "html"
            row["download_status"] = "downloaded"
            row["sha256"] = sha256_file(local_path)
            return row

        row = enrich_policy(row, used_ids)
        local_path = raw_dir / f"{row['source_id']}{suffix}"
        save_response(local_path, response)
        row["local_path"] = str(local_path.relative_to(PROJECT_ROOT))
        row["source_type"] = infer_source_type(url, response, row["local_path"])
        row["download_status"] = "downloaded"
        row["sha256"] = sha256_file(local_path)
        return row

    except Exception as exc:
        row["download_status"] = "error"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        return enrich_policy(row, used_ids)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Download research sources and write a JSONL manifest.")
    parser.add_argument("--source-file", default=str(DEFAULT_SOURCE_FILE))
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--sleep", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    """Run the downloader."""
    args = parse_args()
    try:
        source_file = resolve_source_file(args.source_file)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    raw_dir = resolve_project_path(args.raw_dir)
    manifest_path = resolve_project_path(args.manifest)
    urls = read_urls(source_file)
    used_ids: set[str] = set()
    rows: list[dict[str, Any]] = []

    print(f"Source file: {source_file}")
    print(f"URLs found: {len(urls)}")
    print(f"Raw output: {raw_dir}")
    print(f"Manifest: {manifest_path}")

    for index, url in enumerate(urls, start=1):
        print(f"[{index}/{len(urls)}] {url}")
        row = download_one(url, raw_dir=raw_dir, used_ids=used_ids, timeout=args.timeout)
        rows.append(row)
        print(
            f"  {row.get('download_status')} | {row.get('source_id')} | "
            f"{row.get('source_type')} | stats={row.get('statistics_allowed')}"
        )
        if args.sleep > 0 and index < len(urls):
            time.sleep(args.sleep)

    write_jsonl(rows, manifest_path)
    downloaded = sum(1 for row in rows if str(row.get("download_status", "")).startswith("downloaded"))
    failed = len(rows) - downloaded
    print("\nDownload complete")
    print(f"Manifest rows: {len(rows)}")
    print(f"Downloaded: {downloaded}")
    print(f"Failed/error rows: {failed}")
    print(f"Statistics-allowed rows: {sum(1 for row in rows if row.get('statistics_allowed'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

