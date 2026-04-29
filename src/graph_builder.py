from neo4j import GraphDatabase
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
        """Legacy compatibility hook for import scripts.

        The current import flow writes directly to the concrete entity labels,
        so there is no shared `Entity` label to constrain here.
        """
        return None

    def backfill_entity_keys(self):
        """Legacy compatibility hook for historical databases."""
        return None

    def upsert_entity_by_key(self, entity_key, entity_name, entity_label, norm_name=None, domain=None):
        query = f"""
        MERGE (e:{entity_label} {{entity_key: $entity_key}})
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
        MATCH (e:{entity_label} {{entity_key: $entity_key}})
        MERGE (d)-[:MENTIONS]->(e)
        RETURN d.name AS document, e.name AS entity
        """
        with self.driver.session() as session:
            record = session.run(query, doc_name=doc_name, entity_key=entity_key).single()
            return dict(record) if record else None

    def update_entity_profile(self, entity_key, entity_label, context_terms=None, domain=None):
        context_terms = context_terms or []
        with self.driver.session() as session:
            existing = session.run(
                f"MATCH (e:{entity_label} {{entity_key: $entity_key}}) RETURN e.context_terms AS context_terms, e.domain AS domain",
                entity_key=entity_key,
            ).single()

            if not existing:
                return

            old_terms = existing["context_terms"] or []
            merged_terms = list(dict.fromkeys(old_terms + context_terms))[:80]
            target_domain = existing["domain"] or domain

            session.run(
                """
                MATCH (e:%s {entity_key: $entity_key})
                SET e.context_terms = $context_terms
                SET e.domain = coalesce(e.domain, $domain)
                """ % entity_label,
                entity_key=entity_key,
                context_terms=merged_terms,
                domain=target_domain,
            )

    def upsert_entity(self, entity_name, entity_label, norm_name=None):
        query = f"""
        MERGE (e:{entity_label} {{name: $entity_name}})
        SET e.norm_name = coalesce(e.norm_name, $norm_name)
        RETURN e.name AS name
        """
        with self.driver.session() as session:
            record = session.run(query, entity_name=entity_name, norm_name=norm_name).single()
            return record["name"] if record else None

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
        MATCH (h:{head_label} {{entity_key: $head_entity_key}})
        MATCH (t:{tail_label} {{entity_key: $tail_entity_key}})
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
