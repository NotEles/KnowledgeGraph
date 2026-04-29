import json
from pathlib import Path

from crf import CRFNERExtractor
from entity_resolver import EntityResolver
from graph_builder import KGBuilder


def load_duie_records(data_path):
    """Load DuIE data from either JSONL or a JSON array file."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    with data_path.open("r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    if content.startswith("["):
        return json.loads(content)

    records = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _safe_label(label, default="Entity"):
    if not label:
        return default
    label = str(label).strip()
    if not label:
        return default
    safe = []
    for char in label:
        if char.isalnum() or char == "_":
            safe.append(char)
        else:
            safe.append("_")
    result = "".join(safe).strip("_")
    return result or default


def import_duie_entities_to_neo4j(
    data_path,
    model_path="ner_model.pkl",
    uri="bolt://localhost:7688",
    user="neo4j",
    password="your_new_password",
    limit=None,
):
    """
    Read DuIE records, extract entities with CRF, and import them into Neo4j.

    Graph shape:
        (Document)-[:MENTIONS]->(PER/LOC/ORG/WORK)
    """
    extractor = CRFNERExtractor(model_path=model_path)
    if not extractor.model:
        extractor.load()

    records = load_duie_records(data_path)
    if limit is not None:
        records = records[:limit]

    kg = KGBuilder(uri, user, password)
    resolver = EntityResolver(kg)
    try:
        kg.ensure_disambiguation_schema()
        total_mentions = 0
        decision_stats = {"resolved": 0, "new": 0}
        for index, record in enumerate(records, start=1):
            text = record.get("text", "")
            if not text:
                continue

            doc_name = f"duie_{index}"
            entities = extractor.extract_entities(text)
            if not entities:
                continue

            for entity_name, entity_type in entities:
                entity_label = _safe_label(entity_type, default="Entity")
                resolved = resolver.resolve(entity_name, entity_label, source_text=text)
                canonical_name = resolved["canonical_name"]
                entity_key = resolved["entity_key"]

                kg.upsert_entity_by_key(
                    entity_key,
                    canonical_name,
                    entity_label,
                    norm_name=resolved["norm_name"],
                    domain=resolved["domain"],
                )
                kg.add_document_mention(
                    doc_name,
                    entity_key,
                    entity_label,
                )
                kg.update_entity_profile(
                    entity_key,
                    entity_label,
                    context_terms=resolved["context_terms"],
                    domain=resolved["domain"],
                )
                total_mentions += 1
                decision_stats[resolved["decision"]] += 1

            print(f"已处理第 {index} 条，识别到 {len(entities)} 个实体")

        print(f"导入完成，共写入 {total_mentions} 条 MENTIONS 关系")
        print(
            "消歧决策统计: "
            f"resolved={decision_stats['resolved']}, "
            f"new={decision_stats['new']}"
        )
    finally:
        kg.close()


if __name__ == "__main__":
    default_path = Path(__file__).resolve().parent.parent / "sample" / "DuIE2.0" / "duie_sample.json" / "duie_sample.json"
    import_duie_entities_to_neo4j(
        data_path=default_path,
        model_path="ner_model.pkl",
        uri="bolt://localhost:7688",
        user="neo4j",
        password="your_new_password",
        limit=100,
    )
