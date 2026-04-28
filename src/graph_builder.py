from neo4j import GraphDatabase
import uuid
import os
import numpy as np
# optional: sentence-transformers is used for embeddings (lazy-loaded)
# 默认的密码与账号
user = "neo4j"
password = "your_new_password"
uri = "bolt://localhost:7688"

class KGBuilder:
    def __init__(self, uri=uri, user=user, password=password, embed_model_name=None, local_model_path=None):
        # 初始化数据库连接
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # Embedding model is lazy-loaded; provide a HF model name like
        # 'sentence-transformers/all-MiniLM-L6-v2' or leave None to use default.
        self._embed_model_name = embed_model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self._embed_model = None
        # Optional local model path (directory where a SentenceTransformer model is saved)
        self._local_model_path = local_model_path

    def close(self):
        self.driver.close()

    def ensure_disambiguation_schema(self):
        """Create indexes/constraints for entity disambiguation."""
        statements = [
            "CREATE CONSTRAINT entity_key_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_key IS UNIQUE",
            "CREATE INDEX entity_norm_name_index IF NOT EXISTS FOR (e:Entity) ON (e.norm_name)",
            "CREATE INDEX entity_domain_index IF NOT EXISTS FOR (e:Entity) ON (e.domain)",
            "CREATE INDEX alias_norm_name_index IF NOT EXISTS FOR (a:Alias) ON (a.norm_name)",
            "CREATE INDEX alias_entity_label_index IF NOT EXISTS FOR (a:Alias) ON (a.entity_label)",
        ]
        with self.driver.session() as session:
            for statement in statements:
                session.run(statement)

    def backfill_entity_keys(self):
        """Assign stable keys to historical Entity nodes that predate disambiguation upgrade."""
        with self.driver.session() as session:
            rows = list(
                session.run(
                    "MATCH (e:Entity) WHERE e.entity_key IS NULL RETURN elementId(e) AS eid"
                )
            )
            for row in rows:
                generated_key = f"legacy_{uuid.uuid4().hex[:12]}"
                session.run(
                    "MATCH (e) WHERE elementId(e) = $eid SET e.entity_key = $entity_key",
                    eid=row["eid"],
                    entity_key=generated_key,
                )

    def upsert_entity_by_key(self, entity_key, entity_name, entity_label, norm_name=None, domain=None):
        query = f"""
        MERGE (e:Entity {{entity_key: $entity_key}})
        SET e:{entity_label}
        SET e.name = coalesce(e.name, $entity_name)
        SET e.norm_name = coalesce(e.norm_name, $norm_name)
        SET e.domain = coalesce(e.domain, $domain)
        RETURN e.entity_key AS entity_key, e.name AS name
        """
        with self.driver.session() as session:
            record = session.run(
                query,
                entity_key=entity_key,
                entity_name=entity_name,
                norm_name=norm_name,
                domain=domain,
            ).single()
            return dict(record) if record else None

    def add_document_mention(self, doc_name, entity_key, entity_label):
        query = f"""
        MERGE (d:Document {{name: $doc_name}})
        MATCH (e:Entity:{entity_label} {{entity_key: $entity_key}})
        MERGE (d)-[:MENTIONS]->(e)
        RETURN d.name AS document, e.name AS entity
        """
        with self.driver.session() as session:
            record = session.run(query, doc_name=doc_name, entity_key=entity_key).single()
            return dict(record) if record else None

    def get_disambiguation_candidates(self, entity_label, norm_name, limit=200):
        query = f"""
        CALL () {{
            MATCH (e:Entity:{entity_label})
             WHERE e.norm_name = $norm_name AND e.entity_key IS NOT NULL
            RETURN e.entity_key AS entity_key, e.name AS name, e.norm_name AS norm_name,
                   e.domain AS domain, e.context_terms AS context_terms, 'norm' AS matched_by
            UNION
            MATCH (a:Alias {{norm_name: $norm_name, entity_label: $entity_label}})-[:ALIAS_OF]->(e:Entity:{entity_label})
             WHERE e.entity_key IS NOT NULL
            RETURN e.entity_key AS entity_key, e.name AS name, e.norm_name AS norm_name,
                   e.domain AS domain, e.context_terms AS context_terms, 'alias' AS matched_by
        }}
        RETURN DISTINCT entity_key, name, norm_name, domain, context_terms, matched_by
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(
                query,
                norm_name=norm_name,
                entity_label=entity_label,
                limit=limit,
            )
            return [dict(row) for row in result]

    def _get_embed_model(self):
        if self._embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as e:
                raise ImportError(
                    "sentence-transformers is required for embedding features. "
                    "Install with: pip install sentence-transformers"
                ) from e
            # If a local model path is provided and exists, load from disk.
            if self._local_model_path:
                # if it's a directory saved by .save(), load directly
                if os.path.isdir(self._local_model_path):
                    self._embed_model = SentenceTransformer(self._local_model_path)
                else:
                    # attempt to create parent dir and save downloaded model there
                    parent = os.path.dirname(self._local_model_path)
                    if parent and not os.path.exists(parent):
                        os.makedirs(parent, exist_ok=True)
                    # download from HF and save locally for future runs
                    model = SentenceTransformer(self._embed_model_name)
                    model.save(self._local_model_path)
                    self._embed_model = model
            else:
                # no local path requested, load directly (may download from HF)
                self._embed_model = SentenceTransformer(self._embed_model_name)
        return self._embed_model

    def encode_texts(self, texts):
        """Encode a list of texts into normalized embeddings (numpy arrays).

        Returns a 2D numpy array with shape (len(texts), dim).
        """
        model = self._get_embed_model()
        emb = model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return emb

    def ensure_entity_embeddings(self, entity_label, batch_size=64):
        """Ensure entities of given label have an `embedding` property stored in Neo4j.

        Embeddings are computed from `e.norm_name` if present otherwise from `e.name`.
        """
        fetch_query = f"""
        MATCH (e:Entity:{entity_label})
        WHERE e.entity_key IS NOT NULL AND (e.embedding IS NULL OR size(e.embedding) = 0)
        RETURN e.entity_key AS entity_key, coalesce(e.norm_name, e.name) AS text
        """
        with self.driver.session() as session:
            rows = list(session.run(fetch_query))
            # Process in batches
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                texts = [r["text"] or "" for r in batch]
                if not texts:
                    continue
                embeddings = self.encode_texts(texts)
                for r, emb in zip(batch, embeddings):
                    # Neo4j accepts arrays of floats
                    session.run(
                        "MATCH (e:Entity) WHERE e.entity_key = $entity_key SET e.embedding = $embedding",
                        entity_key=r["entity_key"],
                        embedding=emb.tolist(),
                    )

    def disambiguate_by_embedding(self, entity_label, mention_text, threshold=0.8, top_k=5, candidate_limit=500):
        """Disambiguate a mention by cosine similarity of embeddings.

        Returns a list of candidates with scores sorted desc. Each item is a dict:
        `{entity_key, name, score}`. If no candidate meets `threshold`, an empty list
        is returned.
        """
        # Ensure we have embeddings for entities
        self.ensure_entity_embeddings(entity_label)

        # Encode mention
        mention_emb = self.encode_texts([mention_text])[0]

        # Fetch candidate embeddings
        fetch_query = f"""
        MATCH (e:Entity:{entity_label})
        WHERE e.entity_key IS NOT NULL AND e.embedding IS NOT NULL
        RETURN e.entity_key AS entity_key, e.name AS name, e.embedding AS embedding
        LIMIT $limit
        """
        with self.driver.session() as session:
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

    def update_entity_profile(self, entity_key, context_terms=None, domain=None):
        context_terms = context_terms or []
        with self.driver.session() as session:
            existing = session.run(
                "MATCH (e:Entity {entity_key: $entity_key}) RETURN e.context_terms AS context_terms, e.domain AS domain",
                entity_key=entity_key,
            ).single()

            if not existing:
                return

            old_terms = existing["context_terms"] or []
            merged_terms = list(dict.fromkeys(old_terms + context_terms))[:80]
            target_domain = existing["domain"] or domain

            session.run(
                """
                MATCH (e:Entity {entity_key: $entity_key})
                SET e.context_terms = $context_terms
                SET e.domain = coalesce(e.domain, $domain)
                """,
                entity_key=entity_key,
                context_terms=merged_terms,
                domain=target_domain,
            )

    def upsert_entity(self, entity_name, entity_label, norm_name=None):
        query = f"""
        MERGE (e:{entity_label} {{name: $entity_name}})
        SET e:Entity
        SET e.norm_name = coalesce(e.norm_name, $norm_name)
        RETURN e.name AS name
        """
        with self.driver.session() as session:
            record = session.run(query, entity_name=entity_name, norm_name=norm_name).single()
            return record["name"] if record else None

    def add_alias(self, entity_name, entity_label, alias_name, norm_alias_name):
        query = f"""
        MATCH (e:{entity_label} {{name: $entity_name}})
        MERGE (a:Alias {{norm_name: $norm_alias_name}})
        ON CREATE SET a.name = $alias_name
        MERGE (a)-[:ALIAS_OF]->(e)
        """
        with self.driver.session() as session:
            session.run(
                query,
                entity_name=entity_name,
                alias_name=alias_name,
                norm_alias_name=norm_alias_name,
            )

    def add_alias_by_key(self, entity_key, entity_label, alias_name, norm_alias_name):
        query = f"""
        MATCH (e:Entity:{entity_label} {{entity_key: $entity_key}})
        MERGE (a:Alias {{norm_name: $norm_alias_name, entity_label: $entity_label}})
        ON CREATE SET a.name = $alias_name
        MERGE (a)-[:ALIAS_OF]->(e)
        """
        with self.driver.session() as session:
            session.run(
                query,
                entity_key=entity_key,
                entity_label=entity_label,
                alias_name=alias_name,
                norm_alias_name=norm_alias_name,
            )

    def find_entity_by_norm_name(self, entity_label, norm_name):
        query = f"""
        MATCH (e:{entity_label})
        WHERE e.norm_name = $norm_name
        RETURN e.name AS name
        LIMIT 1
        """
        with self.driver.session() as session:
            record = session.run(query, norm_name=norm_name).single()
            return record["name"] if record else None

    def find_entity_by_alias(self, entity_label, norm_alias_name):
        query = f"""
        MATCH (a:Alias {{norm_name: $norm_alias_name}})-[:ALIAS_OF]->(e:{entity_label})
        RETURN e.name AS name
        LIMIT 1
        """
        with self.driver.session() as session:
            record = session.run(query, norm_alias_name=norm_alias_name).single()
            return record["name"] if record else None

    def get_entities_by_label(self, entity_label, limit=200):
        query = f"""
        MATCH (e:{entity_label})
        RETURN e.name AS name, e.norm_name AS norm_name
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, limit=limit)
            return [(row["name"], row["norm_name"]) for row in result]

    def add_relation(self, head_entity, head_label, relation, tail_entity, tail_label):
        """
        核心入库方法。
        使用 MERGE 而不是 CREATE：如果实体已存在则匹配，不存在则创建。
        这本身就是一种基础的“实体对齐/消歧”机制。
        """
        query = f"""
        MERGE (h:{head_label} {{name: $head_name}})
        MERGE (t:{tail_label} {{name: $tail_name}})
        MERGE (h)-[r:{relation}]->(t)
        RETURN h.name, type(r), t.name
        """
        with self.driver.session() as session:
            result = session.run(query, head_name=head_entity, tail_name=tail_entity)
            return result.single()

    def add_relation_by_key(self, head_entity_key, head_label, relation, tail_entity_key, tail_label):
        """Create a relation between two resolved Entity nodes using stable keys."""
        query = f"""
        MATCH (h:Entity:{head_label} {{entity_key: $head_entity_key}})
        MATCH (t:Entity:{tail_label} {{entity_key: $tail_entity_key}})
        MERGE (h)-[r:{relation}]->(t)
        RETURN h.entity_key AS head_entity_key, type(r) AS relation, t.entity_key AS tail_entity_key
        """
        with self.driver.session() as session:
            result = session.run(
                query,
                head_entity_key=head_entity_key,
                tail_entity_key=tail_entity_key,
            )
            return result.single()
