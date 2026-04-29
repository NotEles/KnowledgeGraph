import os
import re
import unicodedata
import uuid

import numpy as np


class EntityResolver:
    """Resolve entity mentions to stable Neo4j nodes.

    The resolver prefers exact normalized matches, then graph candidates
    scored by a simple context heuristic.
    """

    DOMAIN_MAP = {
        "Person": "people",
        "PER": "people",
        "人物": "people",
        "Location": "location",
        "LOC": "location",
        "地点": "location",
        "Country": "location",
        "Organization": "organization",
        "ORG": "organization",
        "机构": "organization",
        "Work": "work",
        "WORK": "work",
        "作品": "work",
        "Field": "concept",
        "Concept": "concept",
        "Sport": "concept",
        "Award": "concept",
        "Event": "concept",
        "Entity": "entity",
    }

    def __init__(self, kg, candidate_limit=20, embedding_threshold=0.78):
        self.kg = kg
        self.candidate_limit = candidate_limit
        self.embedding_threshold = embedding_threshold
        self._embed_model = None
        self._embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self._local_model_path = getattr(kg, "_local_model_path", None)

    @staticmethod
    def normalize_text(text):
        if text is None:
            return ""
        text = unicodedata.normalize("NFKC", str(text)).strip().lower()
        text = re.sub(r"[\s\u3000]+", "", text)
        text = re.sub(r"[\"'“”‘’·•,，。;；:：!?！？()\[\]{}<>《》【】]", "", text)
        return text

    def infer_domain(self, entity_label):
        return self.DOMAIN_MAP.get(entity_label, "entity")

    def get_disambiguation_candidates(self, entity_label, norm_name, limit=200):
        query = f"""
        CALL () {{
            MATCH (e:{entity_label})
             WHERE e.norm_name = $norm_name AND e.entity_key IS NOT NULL
            RETURN e.entity_key AS entity_key, e.name AS name, e.norm_name AS norm_name,
                   e.domain AS domain, e.context_terms AS context_terms, 'norm' AS matched_by
        }}
        RETURN DISTINCT entity_key, name, norm_name, domain, context_terms, matched_by
        LIMIT $limit
        """
        with self.kg.driver.session() as session:
            result = session.run(
                query,
                norm_name=norm_name,
                entity_label=entity_label,
                limit=limit,
            )
            return [dict(row) for row in result]

    @staticmethod
    def _unique_terms(values):
        terms = []
        seen = set()
        for value in values or []:
            if not value:
                continue
            term = str(value).strip()
            if not term:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            terms.append(term)
        return terms

    def _score_candidate(self, candidate, context_terms=None, source_text=""):
        score = 0.0
        candidate_terms = set(candidate.get("context_terms") or [])
        mention_terms = set(self._unique_terms(context_terms))

        if candidate.get("matched_by") == "norm":
            score += 0.2

        if candidate_terms and mention_terms:
            overlap = len(candidate_terms & mention_terms)
            score += overlap / max(len(candidate_terms), len(mention_terms))

        candidate_name = self.normalize_text(candidate.get("name") or "")
        source_norm = self.normalize_text(source_text)
        if candidate_name and candidate_name in source_norm:
            score += 0.15

        domain = candidate.get("domain") or ""
        if domain and domain != "entity":
            score += 0.05

        return score

    def _new_entity_key(self, entity_label):
        domain = self.infer_domain(entity_label)
        return f"{domain}_{uuid.uuid4().hex[:12]}"

    def _get_embed_model(self):
        if self._embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as e:
                raise ImportError(
                    "sentence-transformers is required for embedding features. "
                    "Install with: pip install sentence-transformers"
                ) from e
            if self._local_model_path:
                if os.path.isdir(self._local_model_path):
                    self._embed_model = SentenceTransformer(self._local_model_path)
                else:
                    parent = os.path.dirname(self._local_model_path)
                    if parent and not os.path.exists(parent):
                        os.makedirs(parent, exist_ok=True)
                    model = SentenceTransformer(self._embed_model_name)
                    model.save(self._local_model_path)
                    self._embed_model = model
            else:
                self._embed_model = SentenceTransformer(self._embed_model_name)
        return self._embed_model

    def encode_texts(self, texts):
        model = self._get_embed_model()
        emb = model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return emb

    def ensure_entity_embeddings(self, entity_label, batch_size=64):
        fetch_query = f"""
        MATCH (e:{entity_label})
        WHERE e.entity_key IS NOT NULL AND (e.embedding IS NULL OR size(e.embedding) = 0)
        RETURN e.entity_key AS entity_key, coalesce(e.norm_name, e.name) AS text
        """
        with self.kg.driver.session() as session:
            rows = list(session.run(fetch_query))
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                texts = [r["text"] or "" for r in batch]
                if not texts:
                    continue
                embeddings = self.encode_texts(texts)
                for r, emb in zip(batch, embeddings):
                    session.run(
                        "MATCH (e) WHERE e.entity_key = $entity_key SET e.embedding = $embedding",
                        entity_key=r["entity_key"],
                        embedding=emb.tolist(),
                    )

    def disambiguate_by_embedding(self, entity_label, mention_text, threshold=0.8, top_k=5, candidate_limit=500):
        self.ensure_entity_embeddings(entity_label)

        mention_emb = self.encode_texts([mention_text])[0]

        fetch_query = f"""
        MATCH (e:{entity_label})
        WHERE e.entity_key IS NOT NULL AND e.embedding IS NOT NULL
        RETURN e.entity_key AS entity_key, e.name AS name, e.embedding AS embedding
        LIMIT $limit
        """
        with self.kg.driver.session() as session:
            result = session.run(fetch_query, limit=candidate_limit)
            candidates = [dict(r) for r in result]

        scored = []
        for c in candidates:
            emb = np.array(c.get("embedding") or [], dtype=float)
            if emb.size == 0:
                continue
            score = float(np.dot(mention_emb, emb) / (np.linalg.norm(mention_emb) * np.linalg.norm(emb)))
            if score >= threshold:
                scored.append({"entity_key": c["entity_key"], "name": c["name"], "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def resolve(self, entity_name, entity_label, source_text="", context_terms=None, domain=None):
        norm_name = self.normalize_text(entity_name)
        domain = domain or self.infer_domain(entity_label)
        context_terms = self._unique_terms(context_terms)

        candidates = self.get_disambiguation_candidates(
            entity_label,
            norm_name,
            limit=self.candidate_limit,
        )

        if candidates:
            exact_matches = [candidate for candidate in candidates if candidate.get("matched_by") == "norm"]
            pool = exact_matches or candidates
            best_candidate = max(
                pool,
                key=lambda candidate: self._score_candidate(candidate, context_terms, source_text),
            )

            score = self._score_candidate(best_candidate, context_terms, source_text)
            if len(pool) == 1 or score >= self.embedding_threshold:
                return {
                    "entity_key": best_candidate["entity_key"],
                    "canonical_name": best_candidate["name"] or entity_name,
                    "norm_name": best_candidate.get("norm_name") or norm_name,
                    "domain": best_candidate.get("domain") or domain,
                    "context_terms": self._unique_terms(
                        (best_candidate.get("context_terms") or []) + context_terms
                    ),
                    "decision": "resolved",
                }

        return {
            "entity_key": self._new_entity_key(entity_label),
            "canonical_name": entity_name,
            "norm_name": norm_name,
            "domain": domain,
            "context_terms": context_terms,
            "decision": "new",
        }