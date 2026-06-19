"""Collect research source files from URLs for the AI-impact evidence pipeline.

This script reads ``data/research/source_urls.txt``, downloads PDFs or webpage
snapshots, records stable source IDs and local paths, and writes the source
inventory CSV used by later metadata enrichment and chunking steps.
"""

from pathlib import Path
from urllib.parse import urljoin, urlparse
import hashlib
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_DIR = Path("data/research")
PDF_DIR = BASE_DIR / "pdfs"
WEB_DIR = BASE_DIR / "web_snapshots"

PDF_DIR.mkdir(parents=True, exist_ok=True)
WEB_DIR.mkdir(parents=True, exist_ok=True)

INPUT_TXT = BASE_DIR / "source_urls.txt"
OUTPUT_CSV = BASE_DIR / "research_sources.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 academic project source collection"
}


def short_hash(text: str, length: int = 8) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def slugify(text: str, max_length: int = 60) -> str:
    text = text.lower()
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"www\.", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    return text[:max_length].strip("_")


def make_source_id(url: str, used_ids: set[str]) -> str:
    """
    Creates a stable source_id from the URL.
    Examples:
    https://arxiv.org/abs/2303.10130
    -> arxiv_2303_10130

    https://openai.com/research/gpts-are-gpts
    -> openai_research_gpts_are_gpts_ab12cd34
    """

    parsed = urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "")
    path = parsed.path.strip("/")

    # Special case: arXiv
    arxiv_match = re.search(r"arxiv\.org/(abs|pdf)/([^?#/]+)", url)
    if arxiv_match:
        arxiv_id = arxiv_match.group(2).replace(".", "_")
        base_id = f"arxiv_{arxiv_id}"
    else:
        domain_part = domain.split(".")[0]
        path_part = slugify(path) if path else "homepage"
        base_id = f"{domain_part}_{path_part}_{short_hash(url)}"

    base_id = slugify(base_id, max_length=80)

    source_id = base_id
    counter = 2

    while source_id in used_ids:
        source_id = f"{base_id}_{counter}"
        counter += 1

    used_ids.add(source_id)
    return source_id


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)

    return h.hexdigest()


def is_pdf_response(response: requests.Response) -> bool:
    content_type = response.headers.get("content-type", "").lower()

    return (
        "application/pdf" in content_type
        or response.content[:4] == b"%PDF"
    )


def arxiv_pdf_url(url: str) -> str | None:
    """
    Converts:
    https://arxiv.org/abs/2303.10130

    Into:
    https://arxiv.org/pdf/2303.10130.pdf
    """

    match = re.search(r"arxiv\.org/abs/([^?#]+)", url)

    if match:
        arxiv_id = match.group(1)
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    return None


def find_pdf_link(html: str, base_url: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")

    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(" ", strip=True).lower()

        if href.lower().endswith(".pdf") or "pdf" in text:
            return urljoin(base_url, href)

    return None


def html_to_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text("\n", strip=True)

    return title, text


def infer_source_type(url: str, status: str) -> str:
    url_lower = url.lower()

    if "arxiv.org" in url_lower:
        return "paper"

    if status.startswith("saved_pdf"):
        return "paper_or_report"

    return "website"


def read_urls(path: Path) -> list[str]:
    urls = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            urls.append(line)

    return urls


def process_source(source_id: str, url: str) -> dict:
    print(f"Processing {source_id}: {url}")

    result = {
        "source_id": source_id,
        "title": "",
        "authors": "",
        "year": "",
        "source_type": "",
        "doi": "",
        "url": url,
        "final_url": "",
        "local_path": "",
        "access_date": pd.Timestamp.today().date().isoformat(),
        "content_type": "",
        "sha256": "",
        "status": "",
    }

    try:
        candidate_urls = [url]

        arxiv_pdf = arxiv_pdf_url(url)
        if arxiv_pdf:
            candidate_urls.insert(0, arxiv_pdf)

        response = None

        for candidate in candidate_urls:
            r = requests.get(
                candidate,
                headers=HEADERS,
                timeout=30,
                allow_redirects=True,
            )

            if r.status_code == 200:
                response = r
                break

        if response is None:
            result["status"] = "failed_download"
            result["source_type"] = infer_source_type(url, result["status"])
            return result

        result["final_url"] = response.url
        result["content_type"] = response.headers.get("content-type", "")

        # Case 1: direct PDF
        if is_pdf_response(response):
            pdf_path = PDF_DIR / f"{source_id}.pdf"
            pdf_path.write_bytes(response.content)

            result["local_path"] = str(pdf_path)
            result["sha256"] = sha256_file(pdf_path)
            result["status"] = "saved_pdf"
            result["source_type"] = infer_source_type(url, result["status"])
            return result

        # Case 2: HTML page that may contain a PDF link
        html = response.text
        pdf_link = find_pdf_link(html, response.url)

        if pdf_link:
            pdf_response = requests.get(
                pdf_link,
                headers=HEADERS,
                timeout=30,
                allow_redirects=True,
            )

            if pdf_response.status_code == 200 and is_pdf_response(pdf_response):
                pdf_path = PDF_DIR / f"{source_id}.pdf"
                pdf_path.write_bytes(pdf_response.content)

                result["final_url"] = pdf_response.url
                result["content_type"] = pdf_response.headers.get("content-type", "")
                result["local_path"] = str(pdf_path)
                result["sha256"] = sha256_file(pdf_path)
                result["status"] = "saved_pdf_from_page"
                result["source_type"] = infer_source_type(url, result["status"])
                return result

        # Case 3: no PDF found, save webpage snapshot
        html_path = WEB_DIR / f"{source_id}.html"
        txt_path = WEB_DIR / f"{source_id}.txt"

        title, text = html_to_text(html)

        html_path.write_text(html, encoding="utf-8")
        txt_path.write_text(text, encoding="utf-8")

        result["title"] = title
        result["local_path"] = str(txt_path)
        result["status"] = "saved_webpage_text"
        result["source_type"] = infer_source_type(url, result["status"])

        return result

    except Exception as e:
        result["status"] = f"error: {type(e).__name__}: {e}"
        result["source_type"] = infer_source_type(url, result["status"])
        return result


def main():
    urls = read_urls(INPUT_TXT)

    used_ids = set()
    records = []

    for url in urls:
        source_id = make_source_id(url, used_ids)
        record = process_source(source_id, url)
        records.append(record)

        time.sleep(2)

    sources_df = pd.DataFrame(records)
    sources_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved source registry to: {OUTPUT_CSV}")
    print(sources_df[["source_id", "source_type", "status", "local_path"]])


if __name__ == "__main__":
    main()
