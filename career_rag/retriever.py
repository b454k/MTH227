#!/usr/bin/env python3
"""Reusable ChromaDB retriever for O*NET career guidance documents.

This module reads from the existing ChromaDB database created by the embedding
scripts. It does not create, delete, or re-embed documents.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any


MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHROMA_DB_PATH = "data/chroma_onet"

SECTION_COLLECTION_NAME = "onet_sections"
FULL_COLLECTION_NAME = "onet_full_occupations"
SUPPLEMENTAL_COLLECTION_NAME = "onet_supplemental"
COLLECTION_NAME = SECTION_COLLECTION_NAME

SECTION_COLLECTION_KEY = "sections"
FULL_COLLECTION_KEY = "full_occupations"
SUPPLEMENTAL_COLLECTION_KEY = "supplemental"
DEFAULT_K = 5

QUERY_TYPE_SECTION_FILTERS: dict[str, list[str] | None] = {
    "software": ["Software Skills"],
    "day_in_life": ["Tasks", "Work Activities", "Work Context"],
    "skills": ["Skills", "Knowledge", "Abilities"],
    "interests": ["Interests", "Work Styles", "Work Context"],
    "tasks": ["Tasks", "Work Activities"],
    "technology_career": None,
    "education": None,
    "general": None,
}

SECTION_QUERY_TYPES = {"software", "day_in_life", "skills", "interests", "tasks"}
FULL_QUERY_TYPES = {"education", "technology_career"}

OCCUPATION_TITLE_KEYS = ("occupation_title", "title", "occupation")
OCCUPATION_CODE_KEYS = ("onet_soc_code", "occupation_code", "soc_code", "code")
SECTION_KEYS = ("section", "section_name")
DOC_TYPE_KEYS = ("doc_type", "source_type")

TECHNOLOGY_CAREER_KEYWORDS = (
    "machine learning",
    "artificial intelligence",
    "data science",
    "deep learning",
    "predictive modeling",
    "predictive modelling",
)


def detect_query_type(query: str) -> str:
    """Detect a simple career query type using keyword rules."""
    query_lower = query.lower()

    if any(keyword in query_lower for keyword in TECHNOLOGY_CAREER_KEYWORDS):
        return "technology_career"

    keyword_rules = [
        (
            "day_in_life",
            (
                "day in the life",
                "daily",
                "routine",
                "what does",
                "typical day",
            ),
        ),
        (
            "education",
            ("education", "degree", "training", "qualification", "job zone"),
        ),
        (
            "interests",
            ("interests", "personality", "fit", "riasec"),
        ),
        (
            "tasks",
            ("tasks", "responsibilities", "duties"),
        ),
    ]

    for query_type, keywords in keyword_rules:
        if any(keyword in query_lower for keyword in keywords):
            return query_type

    if _looks_like_software_query(query_lower):
        return "software"

    if any(
        keyword in query_lower
        for keyword in ("skills", "learn", "abilities", "knowledge", "requirements")
    ):
        return "skills"

    return "general"


def _looks_like_software_query(query_lower: str) -> bool:
    """Return True when a query is asking about tools or software lists."""
    direct_software_keywords = (
        "software used",
        "software skills",
        "software tools",
        "tools",
        "technologies",
        "python",
        "excel",
        "sql",
    )
    if any(keyword in query_lower for keyword in direct_software_keywords):
        return True

    occupation_software_phrases = (
        "software developer",
        "software developers",
        "software engineer",
        "software engineers",
    )
    if "software" in query_lower and not any(
        phrase in query_lower for phrase in occupation_software_phrases
    ):
        return True

    return False


def format_distance_to_score(distance: float) -> float:
    """Convert Chroma's distance value into a simple 0-1 similarity score."""
    return 1.0 / (1.0 + distance)


def format_results(results: list[dict[str, Any]]) -> str:
    """Format retrieval results for readable console output."""
    if not results:
        return "No results found."

    lines: list[str] = []
    for rank, result in enumerate(results, 1):
        metadata = result.get("metadata") or {}
        text = result.get("text") or ""
        preview = text[:600].replace("\n", " ")
        if len(text) > 600:
            preview += "..."

        lines.extend(
            [
                f"Rank {rank}",
                f"  Score:            {result.get('score', 0.0):.4f}",
                f"  Distance:         {result.get('distance', 0.0):.4f}",
                f"  Collection:       {result.get('collection', 'N/A')}",
                f"  Document ID:      {result.get('id', 'N/A')}",
                f"  Occupation Title: {metadata.get('occupation_title', 'N/A')}",
                f"  Occupation Code:  {metadata.get('onet_soc_code', 'N/A')}",
                f"  Section:          {metadata.get('section', 'N/A')}",
                f"  Text Preview:     {preview}",
                "",
            ]
        )

    return "\n".join(lines).rstrip()


class OnetRetriever:
    """Retrieve relevant O*NET documents from section and full collections."""

    def __init__(
        self,
        chroma_path: str = CHROMA_DB_PATH,
        collection_name: str = SECTION_COLLECTION_NAME,
        embedding_model_name: str = MODEL_NAME,
        full_collection_name: str = FULL_COLLECTION_NAME,
        supplemental_collection_name: str = SUPPLEMENTAL_COLLECTION_NAME,
    ) -> None:
        """Load the embedding model and connect to both Chroma collections."""
        self.chroma_path = Path(chroma_path)
        self.collection_name = collection_name
        self.section_collection_name = collection_name
        self.full_collection_name = full_collection_name
        self.supplemental_collection_name = supplemental_collection_name
        self.embedding_model_name = embedding_model_name

        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "Could not import chromadb. Install project dependencies before "
                "using OnetRetriever."
            ) from exc

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Could not import sentence_transformers. Install project "
                "dependencies before using OnetRetriever."
            ) from exc

        if not self.chroma_path.exists():
            raise FileNotFoundError(
                f"ChromaDB path does not exist: {self.chroma_path}. "
                "Expected the existing database at data/chroma_onet."
            )

        try:
            self.model = SentenceTransformer(self.embedding_model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load embedding model '{self.embedding_model_name}'. "
                "Make sure the model is installed or can be downloaded."
            ) from exc

        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.collections = {
            SECTION_COLLECTION_KEY: self._load_collection(
                self.section_collection_name,
                "section-level O*NET documents",
            ),
            FULL_COLLECTION_KEY: self._load_collection(
                self.full_collection_name,
                "full O*NET occupation documents",
            ),
        }
        supplemental_collection = self._load_optional_collection(
            self.supplemental_collection_name,
            "supplemental O*NET documents",
        )
        if supplemental_collection is not None:
            self.collections[SUPPLEMENTAL_COLLECTION_KEY] = supplemental_collection

        # Backward-compatible aliases for code that used the original retriever.
        self.collection = self.collections[SECTION_COLLECTION_KEY]
        self.full_collection = self.collections[FULL_COLLECTION_KEY]
        self.supplemental_collection = supplemental_collection

        self.collection_counts = {
            key: collection.count() for key, collection in self.collections.items()
        }
        self.collection_count = self.collection_counts[SECTION_COLLECTION_KEY]

        for key, count in self.collection_counts.items():
            if count == 0:
                warnings.warn(
                    f"Chroma collection '{self.collections[key].name}' is empty.",
                    stacklevel=2,
                )

        self.collection_metadata = self._inspect_all_collection_metadata()

        # Backward-compatible metadata aliases for the section collection.
        section_metadata = self.collection_metadata[SECTION_COLLECTION_KEY]
        self.metadata_keys = section_metadata["metadata_keys"]
        self.section_key = section_metadata["section_key"]
        self.occupation_code_key = section_metadata["occupation_code_key"]
        self.occupation_title_key = section_metadata["occupation_title_key"]

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_K,
        section_filter: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve documents using intent-based collection routing.

        Passing ``section_filter`` manually always searches ``onet_sections``.
        Without a manual filter, the query type chooses the best collection:
        section-level documents, full occupation documents, or both.
        """
        self._validate_query(query)
        self._validate_k(k)

        if section_filter is not None:
            return self._retrieve_sections(query, k, section_filter)

        query_type = detect_query_type(query)
        return self._retrieve_by_query_type(query, k, query_type, allow_fallback=False)

    def retrieve_smart(self, query: str, k: int = DEFAULT_K) -> list[dict[str, Any]]:
        """Retrieve with intent routing and broad fallback if results are sparse."""
        self._validate_query(query)
        self._validate_k(k)

        query_type = detect_query_type(query)
        return self._retrieve_by_query_type(query, k, query_type, allow_fallback=True)

    def retrieve_for_occupation(
        self,
        query: str,
        onet_soc_code: str,
        k: int = DEFAULT_K,
    ) -> list[dict[str, Any]]:
        """Retrieve documents for one O*NET-SOC occupation code."""
        self._validate_query(query)
        self._validate_k(k)

        if not onet_soc_code or not onet_soc_code.strip():
            raise ValueError("onet_soc_code must not be empty.")

        query_type = detect_query_type(query)
        onet_soc_code = onet_soc_code.strip()

        if query_type in SECTION_QUERY_TYPES:
            section_filter = QUERY_TYPE_SECTION_FILTERS[query_type]
            where = self._combine_where_clauses(
                self._build_occupation_where(SECTION_COLLECTION_KEY, onet_soc_code),
                self._build_section_where(SECTION_COLLECTION_KEY, section_filter),
            )
            return self._retrieve_from_collection(
                SECTION_COLLECTION_KEY,
                query,
                k,
                where=where,
            )

        if query_type in FULL_QUERY_TYPES:
            where = self._build_occupation_where(FULL_COLLECTION_KEY, onet_soc_code)
            return self._retrieve_from_collection(
                FULL_COLLECTION_KEY,
                query,
                k,
                where=where,
            )

        return self._search_both_collections(
            query,
            k,
            where_by_collection={
                SECTION_COLLECTION_KEY: self._build_occupation_where(
                    SECTION_COLLECTION_KEY,
                    onet_soc_code,
                ),
                FULL_COLLECTION_KEY: self._build_occupation_where(
                    FULL_COLLECTION_KEY,
                    onet_soc_code,
                ),
            },
        )

    def retrieve_supplemental(
        self,
        query: str,
        k: int = DEFAULT_K,
        doc_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve from the optional supplemental collection.

        Normal ``retrieve`` and ``retrieve_smart`` calls do not use this
        collection. Call this method when title aliases, related occupations,
        task-DWA mappings, work-activity hierarchy, or content-model linkages
        are specifically useful.
        """
        self._validate_query(query)
        self._validate_k(k)

        if not self._has_supplemental_collection():
            return []

        where = self._build_doc_type_where(SUPPLEMENTAL_COLLECTION_KEY, doc_type)
        return self._retrieve_from_collection(
            SUPPLEMENTAL_COLLECTION_KEY,
            query,
            k,
            where=where,
        )

    def find_occupation_alias(
        self,
        query: str,
        k: int = DEFAULT_K,
    ) -> list[dict[str, Any]]:
        """Search supplemental occupation title and alias documents."""
        return self.retrieve_supplemental(
            query,
            k=k,
            doc_type="occupation_aliases",
        )

    def get_related_occupations(
        self,
        onet_soc_code: str,
        k: int = 10,
    ) -> list[dict[str, Any]]:
        """Retrieve the related-occupation document for one O*NET-SOC code."""
        self._validate_k(k)
        onet_soc_code = onet_soc_code.strip()
        if not onet_soc_code:
            raise ValueError("onet_soc_code must not be empty.")
        if not self._has_supplemental_collection():
            return []

        where = self._combine_where_clauses(
            self._build_doc_type_where(
                SUPPLEMENTAL_COLLECTION_KEY,
                "related_occupations",
            ),
            self._build_occupation_where(SUPPLEMENTAL_COLLECTION_KEY, onet_soc_code),
        )
        return self._retrieve_from_collection(
            SUPPLEMENTAL_COLLECTION_KEY,
            f"related occupations for {onet_soc_code}",
            k,
            where=where,
        )

    def get_task_dwa_mapping(
        self,
        onet_soc_code: str,
        k: int = 10,
    ) -> list[dict[str, Any]]:
        """Retrieve task-to-DWA mapping documents for one O*NET-SOC code."""
        self._validate_k(k)
        onet_soc_code = onet_soc_code.strip()
        if not onet_soc_code:
            raise ValueError("onet_soc_code must not be empty.")
        if not self._has_supplemental_collection():
            return []

        where = self._combine_where_clauses(
            self._build_doc_type_where(
                SUPPLEMENTAL_COLLECTION_KEY,
                "task_dwa_mapping",
            ),
            self._build_occupation_where(SUPPLEMENTAL_COLLECTION_KEY, onet_soc_code),
        )
        return self._retrieve_from_collection(
            SUPPLEMENTAL_COLLECTION_KEY,
            f"task detailed work activity mapping for {onet_soc_code}",
            k,
            where=where,
        )

    @staticmethod
    def detect_query_type(query: str) -> str:
        """Detect a simple career query type using the module-level rules."""
        return detect_query_type(query)

    def _retrieve_by_query_type(
        self,
        query: str,
        k: int,
        query_type: str,
        allow_fallback: bool,
    ) -> list[dict[str, Any]]:
        """Route a query to the correct collection based on its detected type."""
        if query_type in SECTION_QUERY_TYPES:
            section_filter = QUERY_TYPE_SECTION_FILTERS[query_type]
            results = self._retrieve_sections(query, k, section_filter)
            if allow_fallback and len(results) < k:
                fallback_results = self._search_both_collections(query, k)
                return self._merge_unique_results(results, fallback_results, k)
            return results

        if query_type in FULL_QUERY_TYPES:
            results = self._retrieve_from_collection(FULL_COLLECTION_KEY, query, k)
            if allow_fallback and len(results) < k:
                fallback_results = self._search_both_collections(query, k)
                return self._merge_unique_results(results, fallback_results, k)
            return results

        return self._search_both_collections(query, k)

    def _retrieve_sections(
        self,
        query: str,
        k: int,
        section_filter: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Retrieve from the section collection with an optional section filter."""
        where = self._build_section_where(SECTION_COLLECTION_KEY, section_filter)
        return self._retrieve_from_collection(
            SECTION_COLLECTION_KEY,
            query,
            k,
            where=where,
        )

    def _search_both_collections(
        self,
        query: str,
        k: int,
        where_by_collection: dict[str, dict[str, Any] | None] | None = None,
    ) -> list[dict[str, Any]]:
        """Search section and full collections, then return the best combined hits."""
        query_embedding = self._embed_query(query)
        all_results: list[dict[str, Any]] = []

        for collection_key in (SECTION_COLLECTION_KEY, FULL_COLLECTION_KEY):
            where = (where_by_collection or {}).get(collection_key)
            all_results.extend(
                self._retrieve_from_collection_embedding(
                    collection_key,
                    query_embedding,
                    k,
                    where=where,
                )
            )

        all_results.sort(key=lambda result: result["distance"])
        return self._merge_unique_results([], all_results, k)

    def _retrieve_from_collection(
        self,
        collection_key: str,
        query: str,
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Embed a query and retrieve from one Chroma collection."""
        query_embedding = self._embed_query(query)
        return self._retrieve_from_collection_embedding(
            collection_key,
            query_embedding,
            k,
            where=where,
        )

    def _retrieve_from_collection_embedding(
        self,
        collection_key: str,
        query_embedding: list[float],
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a Chroma query and fall back to Python filtering if needed."""
        try:
            raw_results = self._query_chroma_collection(
                collection_key,
                query_embedding,
                k,
                where=where,
            )
            return self._format_chroma_results(raw_results, collection_key)[:k]
        except Exception:
            if not where:
                raise

        fallback_count = self._fallback_candidate_count(collection_key, k)
        raw_results = self._query_chroma_collection(
            collection_key,
            query_embedding,
            fallback_count,
            where=None,
        )
        results = self._format_chroma_results(raw_results, collection_key)
        return self._filter_results_in_python(results, where)[:k]

    def _embed_query(self, query: str) -> list[float]:
        """Create an embedding for one query string."""
        embedding = self.model.encode([query.strip()])[0]
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return list(embedding)

    def _query_chroma_collection(
        self,
        collection_key: str,
        query_embedding: list[float],
        k: int,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Query one ChromaDB collection with an optional metadata filter."""
        count = self.collection_counts[collection_key]
        if count <= 0:
            return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, count),
            "include": ["documents", "distances", "metadatas"],
        }
        if where:
            kwargs["where"] = where

        return self.collections[collection_key].query(**kwargs)

    def _format_chroma_results(
        self,
        raw_results: dict[str, Any],
        collection_key: str,
    ) -> list[dict[str, Any]]:
        """Convert Chroma's nested query output into app-friendly dictionaries."""
        ids = self._first_result_list(raw_results, "ids")
        documents = self._first_result_list(raw_results, "documents")
        distances = self._first_result_list(raw_results, "distances")
        metadatas = self._first_result_list(raw_results, "metadatas")
        collection_name = self.collections[collection_key].name

        results: list[dict[str, Any]] = []
        for doc_id, document, distance, metadata in zip(
            ids,
            documents,
            distances,
            metadatas,
        ):
            distance_float = float(distance)
            normalized_metadata = self._normalize_metadata(metadata or {})
            results.append(
                {
                    "id": doc_id,
                    "text": document or "",
                    "score": format_distance_to_score(distance_float),
                    "distance": distance_float,
                    "collection": collection_name,
                    "metadata": normalized_metadata,
                }
            )

        return results

    def _normalize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Keep original metadata and add canonical keys used by the app."""
        normalized = dict(metadata)
        normalized["occupation_title"] = self._metadata_value(
            metadata,
            OCCUPATION_TITLE_KEYS,
        )
        normalized["onet_soc_code"] = self._metadata_value(
            metadata,
            OCCUPATION_CODE_KEYS,
        )
        normalized["section"] = self._metadata_value(metadata, SECTION_KEYS)
        normalized["doc_type"] = self._metadata_value(metadata, DOC_TYPE_KEYS)
        return normalized

    def _metadata_value(
        self,
        metadata: dict[str, Any],
        possible_keys: tuple[str, ...],
    ) -> Any:
        """Read the first non-empty metadata value from a list of possible keys."""
        for key in possible_keys:
            value = metadata.get(key)
            if value not in (None, ""):
                return value

        lower_to_key = {key.lower(): key for key in metadata}
        for key in possible_keys:
            real_key = lower_to_key.get(key.lower())
            if real_key:
                value = metadata.get(real_key)
                if value not in (None, ""):
                    return value

        return None

    def _build_section_where(
        self,
        collection_key: str,
        section_filter: list[str] | None,
    ) -> dict[str, Any] | None:
        """Build a Chroma metadata filter for one or more document sections."""
        if not section_filter:
            return None

        section_key = self.collection_metadata[collection_key]["section_key"]
        if len(section_filter) == 1:
            return {section_key: section_filter[0]}

        return {section_key: {"$in": section_filter}}

    def _build_doc_type_where(
        self,
        collection_key: str,
        doc_type: str | None,
    ) -> dict[str, Any] | None:
        """Build a Chroma metadata filter for supplemental document type."""
        if not doc_type:
            return None

        doc_type_key = self.collection_metadata[collection_key]["doc_type_key"]
        return {doc_type_key: doc_type}

    def _build_occupation_where(
        self,
        collection_key: str,
        onet_soc_code: str,
    ) -> dict[str, Any]:
        """Build a Chroma metadata filter for one occupation code."""
        occupation_code_key = self.collection_metadata[collection_key][
            "occupation_code_key"
        ]
        return {occupation_code_key: onet_soc_code}

    def _combine_where_clauses(
        self,
        *clauses: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Combine metadata filters using Chroma's $and syntax."""
        real_clauses = [clause for clause in clauses if clause]
        if not real_clauses:
            return None
        if len(real_clauses) == 1:
            return real_clauses[0]
        return {"$and": real_clauses}

    def _filter_results_in_python(
        self,
        results: list[dict[str, Any]],
        where: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Apply a small subset of Chroma-style filters in Python."""
        if not where:
            return results

        filtered: list[dict[str, Any]] = []
        for result in results:
            metadata = result.get("metadata") or {}
            if self._metadata_matches_where(metadata, where):
                filtered.append(result)
        return filtered

    def _metadata_matches_where(
        self,
        metadata: dict[str, Any],
        where: dict[str, Any],
    ) -> bool:
        """Return True if metadata matches exact, $in, or $and filter clauses."""
        if "$and" in where:
            return all(
                self._metadata_matches_where(metadata, clause)
                for clause in where["$and"]
            )

        for key, expected in where.items():
            actual = metadata.get(key)
            if isinstance(expected, dict) and "$in" in expected:
                if actual not in expected["$in"]:
                    return False
            elif actual != expected:
                return False
        return True

    def _has_supplemental_collection(self) -> bool:
        """Return True when the optional supplemental collection is available."""
        return (
            SUPPLEMENTAL_COLLECTION_KEY in self.collections
            and self.collection_counts.get(SUPPLEMENTAL_COLLECTION_KEY, 0) > 0
        )

    def _load_collection(self, collection_name: str, description: str) -> Any:
        """Load a required Chroma collection with a helpful error message."""
        try:
            return self.client.get_collection(name=collection_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not find Chroma collection '{collection_name}' for "
                f"{description} at {self.chroma_path}. Make sure the matching "
                "embedding script has been run."
            ) from exc

    def _load_optional_collection(
        self,
        collection_name: str,
        description: str,
    ) -> Any | None:
        """Load an optional Chroma collection, returning None when absent."""
        try:
            return self.client.get_collection(name=collection_name)
        except Exception:
            _ = description
            return None

    def _inspect_all_collection_metadata(self) -> dict[str, dict[str, Any]]:
        """Inspect metadata keys and useful aliases for each collection."""
        collection_metadata: dict[str, dict[str, Any]] = {}

        for collection_key, collection in self.collections.items():
            metadata_keys = self._inspect_metadata_keys(collection)
            collection_metadata[collection_key] = {
                "metadata_keys": metadata_keys,
                "section_key": self._choose_metadata_key(
                    metadata_keys,
                    SECTION_KEYS,
                    "section",
                ),
                "occupation_code_key": self._choose_metadata_key(
                    metadata_keys,
                    OCCUPATION_CODE_KEYS,
                    "onet_soc_code",
                ),
                "occupation_title_key": self._choose_metadata_key(
                    metadata_keys,
                    OCCUPATION_TITLE_KEYS,
                    "occupation_title",
                ),
                "doc_type_key": self._choose_metadata_key(
                    metadata_keys,
                    DOC_TYPE_KEYS,
                    "doc_type",
                ),
            }

        return collection_metadata

    def _inspect_metadata_keys(self, collection: Any) -> set[str]:
        """Inspect Chroma metadata keys without reading source JSONL files."""
        try:
            sample = collection.get(limit=5, include=["metadatas"])
        except Exception:
            return set()

        keys: set[str] = set()
        for metadata in sample.get("metadatas") or []:
            if metadata:
                keys.update(metadata.keys())
        return keys

    def _choose_metadata_key(
        self,
        metadata_keys: set[str],
        possible_keys: tuple[str, ...],
        default: str,
    ) -> str:
        """Pick the real metadata key present in Chroma, falling back to default."""
        for key in possible_keys:
            if key in metadata_keys:
                return key

        lower_to_key = {key.lower(): key for key in metadata_keys}
        for key in possible_keys:
            real_key = lower_to_key.get(key.lower())
            if real_key:
                return real_key

        return default

    def _fallback_candidate_count(self, collection_key: str, k: int) -> int:
        """Choose enough candidates for Python-side filtering."""
        count = self.collection_counts[collection_key]
        if count <= 0:
            return k
        return min(count, max(k * 10, 50))

    def _merge_unique_results(
        self,
        first: list[dict[str, Any]],
        second: list[dict[str, Any]],
        k: int,
    ) -> list[dict[str, Any]]:
        """Merge ranked lists without duplicate collection/document pairs."""
        merged: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()

        for result in first + second:
            result_key = (str(result.get("collection", "")), str(result.get("id", "")))
            if result_key in seen_keys:
                continue
            seen_keys.add(result_key)
            merged.append(result)
            if len(merged) >= k:
                break

        return merged

    @staticmethod
    def _first_result_list(raw_results: dict[str, Any], key: str) -> list[Any]:
        """Return Chroma's first query result list for a given key."""
        values = raw_results.get(key) or [[]]
        if not values:
            return []
        return values[0] or []

    @staticmethod
    def _validate_query(query: str) -> None:
        """Raise ValueError for empty queries."""
        if not query or not query.strip():
            raise ValueError("query must not be empty.")

    @staticmethod
    def _validate_k(k: int) -> None:
        """Raise ValueError for invalid result counts."""
        if k <= 0:
            raise ValueError("k must be greater than 0.")


def _run_cli_tests() -> None:
    """Run a small retrieval smoke test when this file is executed directly."""
    test_queries = [
        "careers involving mathematics",
        "careers involving machine learning",
        "jobs with analytical thinking",
        "day in the life of a data scientist",
        "software used by actuaries",
        "skills needed for software developers",
        "education needed for actuaries",
    ]

    try:
        retriever = OnetRetriever()
    except Exception as exc:
        print(f"Error starting O*NET retriever: {exc}")
        return

    print("=" * 90)
    print("O*NET RETRIEVER CLI TEST")
    print("=" * 90)
    print(f"Chroma path: {retriever.chroma_path}")
    print(f"Embedding model: {retriever.embedding_model_name}")
    print(
        f"Section collection: {retriever.section_collection_name} "
        f"({retriever.collection_counts[SECTION_COLLECTION_KEY]:,} documents)"
    )
    print(
        f"Full collection: {retriever.full_collection_name} "
        f"({retriever.collection_counts[FULL_COLLECTION_KEY]:,} documents)"
    )
    if retriever.supplemental_collection is not None:
        print(
            f"Supplemental collection: {retriever.supplemental_collection_name} "
            f"({retriever.collection_counts[SUPPLEMENTAL_COLLECTION_KEY]:,} documents)"
        )
    else:
        print(
            f"Supplemental collection: {retriever.supplemental_collection_name} "
            "(not loaded)"
        )

    for query in test_queries:
        query_type = detect_query_type(query)
        print("\n" + "-" * 90)
        print(f"Query: {query}")
        print(f"Detected query type: {query_type}")
        print("-" * 90)

        try:
            results = retriever.retrieve_smart(query, k=DEFAULT_K)
        except Exception as exc:
            print(f"Error retrieving results: {exc}")
            continue

        print(format_results(results))


if __name__ == "__main__":
    _run_cli_tests()
