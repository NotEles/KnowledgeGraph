import re
import unicodedata
import uuid


class EntityResolver:
    """Resolve entity mentions to stable Entity nodes in Neo4j.

    The resolver prefers exact normalized matches, then alias matches,
    then graph candidates scored by a simple context heuristic.
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
        if candidate.get("matched_by") == "alias":
            score += 0.1

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

    def resolve(self, entity_name, entity_label, source_text="", context_terms=None, domain=None):
        norm_name = self.normalize_text(entity_name)
        domain = domain or self.infer_domain(entity_label)
        context_terms = self._unique_terms(context_terms)

        candidates = self.kg.get_disambiguation_candidates(
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