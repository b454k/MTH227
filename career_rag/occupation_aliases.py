"""Occupation alias resolution against the local O*NET occupation index."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from career_rag.config import PROJECT_ROOT


DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "duckdb" / "onet.duckdb"

KNOWN_ALIAS_PREFERENCES = {
    "data analyst": [
        "Business Intelligence Analysts",
        "Data Scientists",
        "Financial Quantitative Analysts",
        "Database Architects",
    ],
    "business data analyst": ["Business Intelligence Analysts"],
    "business intelligence analyst": ["Business Intelligence Analysts"],
    "bi analyst": ["Business Intelligence Analysts"],
    "actuary": ["Actuaries"],
    "actuarial analyst": ["Actuaries"],
    "machine learning engineer": [
        "Computer and Information Research Scientists",
        "Data Scientists",
        "Computer Systems Engineers/Architects",
    ],
    "ml engineer": [
        "Computer and Information Research Scientists",
        "Data Scientists",
        "Computer Systems Engineers/Architects",
    ],
    "ai engineer": [
        "Computer and Information Research Scientists",
        "Data Scientists",
        "Computer Systems Engineers/Architects",
    ],
    "data scientist": ["Data Scientists"],
    "operations research analyst": ["Operations Research Analysts"],
    "statistician": ["Statisticians"],
    "financial quantitative analyst": ["Financial Quantitative Analysts"],
    "quantitative analyst": ["Financial Quantitative Analysts"],
    "information technology project manager": ["Information Technology Project Managers"],
    "it project manager": ["Information Technology Project Managers"],
    "search marketing strategist": ["Search Marketing Strategists"],
    "technical product manager": ["Information Technology Project Managers"],
    "product manager": ["Project Management Specialists"],
}


def build_occupation_index(db_path: str | Path = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Build a local occupation index from O*NET titles and title aliases."""
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("duckdb is required to build the occupation index.") from exc

    resolved_path = Path(db_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"O*NET DuckDB database not found: {resolved_path}")

    conn = duckdb.connect(str(resolved_path), read_only=True)
    try:
        rows = conn.execute(
            """
            SELECT onetsoc_code, title, description
            FROM occupation_data
            ORDER BY title
            """
        ).fetchall()
        index = [
            {
                "onet_soc_code": str(code),
                "title": str(title),
                "description": str(description or ""),
                "aliases": [],
                "reported_titles": [],
                "_alias_keys": set(),
                "_reported_title_keys": set(),
            }
            for code, title, description in rows
        ]
        by_code = {item["onet_soc_code"]: item for item in index}

        if _table_exists(conn, "job_titles"):
            title_rows = conn.execute(
                """
                SELECT onetsoc_code, job_title, short_title
                FROM job_titles
                ORDER BY onetsoc_code, job_title
                """
            ).fetchall()
            for code, job_title, short_title in title_rows:
                item = by_code.get(str(code))
                if not item:
                    continue
                _append_unique(item["aliases"], job_title, item["_alias_keys"])
                _append_unique(item["aliases"], short_title, item["_alias_keys"])

        if _table_exists(conn, "sample_of_reported_titles"):
            reported_rows = conn.execute(
                """
                SELECT onetsoc_code, reported_job_title
                FROM sample_of_reported_titles
                ORDER BY onetsoc_code, reported_job_title
                """
            ).fetchall()
            for code, reported_title in reported_rows:
                item = by_code.get(str(code))
                if item:
                    _append_unique(
                        item["reported_titles"],
                        reported_title,
                        item["_reported_title_keys"],
                    )
    finally:
        conn.close()

    for item in index:
        item.pop("_alias_keys", None)
        item.pop("_reported_title_keys", None)

    return index


def resolve_career_alias(title: str, occupation_index: Any) -> dict[str, Any]:
    """Resolve a display title to the closest local O*NET occupation."""
    requested_title = str(title or "").strip()
    occupations = _normalize_occupation_index(occupation_index)
    if not requested_title or not occupations:
        return _empty_resolution(requested_title)

    requested_key = _normalize_title(requested_title)
    requested_variants = _title_variants(requested_title)

    exact_official = _find_by_official_title(occupations, requested_variants)
    if exact_official:
        return _resolution(requested_title, exact_official, "official_title_exact", 0.99)

    preferred = _resolve_known_preference(
        requested_title,
        requested_key,
        requested_variants,
        occupations,
    )
    if preferred:
        return preferred

    exact_alias = _find_by_alias(occupations, requested_variants)
    if exact_alias:
        return _resolution(requested_title, exact_alias, "alias_exact", 0.94)

    contained_alias = _find_by_alias_containment(occupations, requested_key)
    if contained_alias:
        return _resolution(requested_title, contained_alias, "alias_partial", 0.82)

    semantic_match, semantic_score = _find_semantic_match(occupations, requested_key)
    if semantic_match and semantic_score >= 0.54:
        confidence = min(0.80, max(0.55, semantic_score))
        return _resolution(requested_title, semantic_match, "semantic_title_similarity", confidence)

    return _empty_resolution(requested_title)


def _table_exists(conn: Any, table_name: str) -> bool:
    tables = {str(row[0]) for row in conn.execute("SHOW TABLES").fetchall()}
    return table_name in tables


def _append_unique(values: list[str], value: Any, keys: set[str] | None = None) -> None:
    text = str(value or "").strip()
    if not text:
        return
    key = _normalize_title(text)
    if keys is not None:
        if key in keys:
            return
        keys.add(key)
        values.append(text)
        return
    if key not in {_normalize_title(item) for item in values}:
        values.append(text)


def _normalize_occupation_index(occupation_index: Any) -> list[dict[str, Any]]:
    if isinstance(occupation_index, dict):
        raw_items = occupation_index.values()
    elif isinstance(occupation_index, list):
        raw_items = occupation_index
    else:
        raw_items = []

    normalized = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        title = item.get("title") or item.get("occupation_title") or item.get("resolved_onet_title")
        code = item.get("onet_soc_code") or item.get("onetsoc_code") or item.get("soc_code")
        if not title or not code:
            continue
        aliases = _list_text(item.get("aliases")) + _list_text(item.get("job_titles"))
        reported = _list_text(item.get("reported_titles")) + _list_text(item.get("sample_titles"))
        normalized.append(
            {
                **item,
                "title": str(title).strip(),
                "onet_soc_code": str(code).strip(),
                "aliases": aliases,
                "reported_titles": reported,
            }
        )
    return normalized


def _list_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _normalize_title(value: Any) -> str:
    text = str(value or "").lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _title_variants(value: str) -> set[str]:
    base = _normalize_title(value)
    variants = {base}
    tokens = base.split()
    if tokens:
        last = tokens[-1]
        singular = _singularize(last)
        plural = _pluralize(last)
        variants.add(" ".join([*tokens[:-1], singular]).strip())
        variants.add(" ".join([*tokens[:-1], plural]).strip())
    return {variant for variant in variants if variant}


def _singularize(value: str) -> str:
    if value.endswith("ies") and len(value) > 3:
        return value[:-3] + "y"
    if value.endswith("ses"):
        return value[:-2]
    if value.endswith("s") and not value.endswith("ss"):
        return value[:-1]
    return value


def _pluralize(value: str) -> str:
    if value.endswith("y") and len(value) > 1:
        return value[:-1] + "ies"
    if value.endswith("s"):
        return value
    return value + "s"


def _occupation_text_keys(occupation: dict[str, Any]) -> set[str]:
    values = [occupation["title"], *occupation.get("aliases", []), *occupation.get("reported_titles", [])]
    keys: set[str] = set()
    for value in values:
        keys.update(_title_variants(str(value)))
    return keys


def _find_by_official_title(
    occupations: list[dict[str, Any]],
    requested_variants: set[str],
) -> dict[str, Any] | None:
    for occupation in occupations:
        if _title_variants(occupation["title"]) & requested_variants:
            return occupation
    return None


def _resolve_known_preference(
    requested_title: str,
    requested_key: str,
    requested_variants: set[str],
    occupations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    preference_titles = KNOWN_ALIAS_PREFERENCES.get(requested_key)
    if not preference_titles:
        return None

    by_title = {_normalize_title(item["title"]): item for item in occupations}
    for preferred_title in preference_titles:
        preferred = by_title.get(_normalize_title(preferred_title))
        if not preferred:
            continue
        keys = _occupation_text_keys(preferred)
        if keys & requested_variants or requested_key in keys:
            return _resolution(
                requested_title,
                preferred,
                "alias_exact_or_semantic",
                0.96,
            )

    for preferred_title in preference_titles:
        preferred = by_title.get(_normalize_title(preferred_title))
        if preferred:
            return _resolution(
                requested_title,
                preferred,
                "alias_preferred_title",
                0.88,
            )

    return None


def _find_by_alias(
    occupations: list[dict[str, Any]],
    requested_variants: set[str],
) -> dict[str, Any] | None:
    for occupation in occupations:
        alias_keys = set()
        for value in [*occupation.get("aliases", []), *occupation.get("reported_titles", [])]:
            alias_keys.update(_title_variants(str(value)))
        if alias_keys & requested_variants:
            return occupation
    return None


def _find_by_alias_containment(
    occupations: list[dict[str, Any]],
    requested_key: str,
) -> dict[str, Any] | None:
    for occupation in occupations:
        for value in [occupation["title"], *occupation.get("aliases", []), *occupation.get("reported_titles", [])]:
            key = _normalize_title(value)
            if requested_key and (requested_key in key or key in requested_key):
                return occupation
    return None


def _find_semantic_match(
    occupations: list[dict[str, Any]],
    requested_key: str,
) -> tuple[dict[str, Any] | None, float]:
    best: dict[str, Any] | None = None
    best_score = 0.0
    for occupation in occupations:
        candidates = [occupation["title"], *occupation.get("aliases", [])[:20], *occupation.get("reported_titles", [])[:10]]
        for candidate in candidates:
            candidate_key = _normalize_title(candidate)
            if not candidate_key:
                continue
            score = SequenceMatcher(None, requested_key, candidate_key).ratio()
            token_score = _token_overlap(requested_key, candidate_key)
            combined = max(score, token_score)
            if combined > best_score:
                best = occupation
                best_score = combined
    return best, best_score


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)


def _resolution(
    requested_title: str,
    occupation: dict[str, Any],
    method: str,
    confidence: float,
) -> dict[str, Any]:
    return {
        "requested_title": requested_title,
        "resolved_onet_title": occupation["title"],
        "onet_soc_code": occupation["onet_soc_code"],
        "resolution_method": method,
        "confidence": round(float(confidence), 2),
    }


def _empty_resolution(requested_title: str) -> dict[str, Any]:
    return {
        "requested_title": requested_title,
        "resolved_onet_title": None,
        "onet_soc_code": None,
        "resolution_method": "unresolved",
        "confidence": 0.0,
    }
