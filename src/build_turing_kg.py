import argparse
import json
import os
import re
from pathlib import Path

from bilstm_crf import CRFNERExtractor
from entity_resolver import EntityResolver
from graph_builder import KGBuilder
from relation_lstm_crf import CRFRelationExtractor


PROPERTY_RELATION_MAP = {
    "出生": {"date": "BORN_ON", "place": "BORN_IN", "label": "Location"},
    "逝世": {"date": "DIED_ON", "place": "DIED_IN", "label": "Location"},
    "死因": {"relation": "CAUSE_OF_DEATH", "label": "Concept"},
    "墓地": {"relation": "BURIED_AT", "label": "Location"},
    "居住地": {"relation": "RESIDED_IN", "label": "Location"},
    "国籍": {"relation": "NATIONALITY", "label": "Country"},
    "母校": {"relation": "ALMA_MATER", "label": "Organization"},
    "知名于": {"relation": "KNOWN_FOR", "label": "Concept"},
    "研究领域": {"relation": "FIELD_OF", "label": "Concept"},
    "机构": {"relation": "AFFILIATED_WITH", "label": "Organization"},
    "论文": {"relation": "AUTHORED", "label": "Work"},
    "博士导师": {"relation": "ADVISED_BY", "label": "Person"},
    "博士生": {"relation": "HAS_DOCTORAL_STUDENT", "label": "Person"},
    "运动": {"relation": "SPORT", "label": "Concept"},
    "项目": {"relation": "HAS_EVENT", "label": "Concept"},
    "全国性决赛": {"relation": "PARTICIPATED_IN", "label": "Event"},
}


def load_json_object(file_path):
    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl_records(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def safe_relation_name(name):
    relation = []
    for char in str(name):
        if char.isalnum() or char == "_":
            relation.append(char)
        else:
            relation.append("_")
    cleaned = "".join(relation).strip("_")
    return cleaned or "RELATED_TO"


def split_values(value):
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    text = re.sub(r"[（）()\[\]]", " ", text)
    parts = re.split(r"[、，,；;\/\s]+", text)
    return [part.strip() for part in parts if part.strip()]


def extract_date(value):
    match = re.search(r"(?:\d{4}年\d{1,2}月\d{1,2}日|\d{4}-\d{2}-\d{2})", str(value))
    return match.group(0) if match else ""


def remove_date_fragments(value):
    text = re.sub(r"(?:\d{4}年\d{1,2}月\d{1,2}日|\d{4}-\d{2}-\d{2})", " ", str(value))
    text = re.sub(r"[（）()\[\]]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def infer_label(text):
    text = str(text).strip()
    if not text:
        return "Entity"

    if re.search(r"(?:\d{4}年\d{1,2}月\d{1,2}日|\d{4}-\d{2}-\d{2})", text):
        return "Date"
    if any(keyword in text for keyword in ["大学", "学院", "研究所", "实验室", "公司", "委员会", "协会", "学会", "银行", "学校", "实验中心"]):
        return "Organization"
    if any(keyword in text for keyword in ["英国", "美国", "法国", "德国", "中国", "印度", "伦敦", "剑桥", "曼彻斯特", "吉尔福德", "威姆斯洛", "多塞特郡", "新墨西哥州", "柴郡", "英格兰"]):
        return "Location"
    if any(keyword in text for keyword in ["奖", "勋章", "测试", "理论", "问题", "程序", "论文", "定理", "演算", "逻辑", "计算", "模型", "算法", "战争", "生物学", "棋"]):
        if any(mark in text for mark in ["论文", "程序", "机", "系统", "模型"]):
            return "Work"
        return "Concept"
    if any(keyword in text for keyword in ["图灵", "邱奇", "布朗", "乔布斯", "理查兹", "诺依曼", "丘吉尔", "维特根斯坦", "爱因斯坦", "甘迪", "斯托尼"]):
        return "Person"
    return "Entity"


def property_triples(base_entity, key, value):
    mapping = PROPERTY_RELATION_MAP.get(key)
    if not mapping:
        relation = f"HAS_{safe_relation_name(key)}"
        for item in split_values(value) or [str(value).strip()]:
            if item:
                yield {
                    "head": base_entity,
                    "head_label": "Person",
                    "relation": relation,
                    "tail": item,
                    "tail_label": infer_label(item),
                }
        return

    if "date" in mapping:
        date_value = extract_date(value)
        if date_value:
            yield {
                "head": base_entity,
                "head_label": "Person",
                "relation": mapping["date"],
                "tail": date_value,
                "tail_label": "Date",
            }
        place_value = remove_date_fragments(value)
        if place_value:
            yield {
                "head": base_entity,
                "head_label": "Person",
                "relation": mapping["place"],
                "tail": place_value,
                "tail_label": mapping["label"],
            }
        return

    for item in split_values(value) or [str(value).strip()]:
        if not item:
            continue
        yield {
            "head": base_entity,
            "head_label": "Person",
            "relation": mapping["relation"],
            "tail": item,
            "tail_label": mapping["label"],
        }


def normalize_context_terms(context):
    tokens = re.split(r"[、，,；;。！？!?:：\s]+", str(context))
    return [token.strip() for token in tokens if token.strip()]
def import_turing_kg(
    profile_path,
    corpus_path,
    uri="bolt://localhost:7688",
    user="neo4j",
    password="your_new_password",
    model_path=None,
    relation_model_path=None,
    duie_train_path=None,
    train_limit=2000,
    train_epochs=3,
    retrain=False,
    limit_corpus=None,
    limit_content=None,
    dry_run=False,
):
    profile = load_json_object(profile_path)
    records = load_jsonl_records(corpus_path)
    if limit_corpus is not None:
        records = records[:limit_corpus]

    base_entity = profile.get("entity", "艾伦·图灵")
    properties = profile.get("properties", {})
    one_hop_links = profile.get("one_hop_links", [])
    content_items = profile.get("content", [])
    if limit_content is not None:
        content_items = content_items[:limit_content]

    if dry_run:
        property_count = sum(1 for key, value in properties.items() for _ in property_triples(base_entity, key, value))
        print(f"基准实体: {base_entity}")
        print(f"半结构化属性数: {property_count}")
        print(f"一跳节点数: {len(one_hop_links)}")
        print(f"关系语料数: {len(records)}")
        print(f"内容段落数: {len(content_items)}")
        return {
            "base_entity": base_entity,
            "property_count": property_count,
            "one_hop_count": len(one_hop_links),
            "corpus_count": len(records),
            "content_count": len(content_items),
        }

    model_path = model_path or str(Path(__file__).resolve().parent.parent / "models" / "duie_ner_bilstm_crf.pt")
    relation_model_path = relation_model_path or str(Path(__file__).resolve().parent.parent / "models" / "duie_rel_bilstm_crf.pt")
    duie_train_path = duie_train_path or str(
        Path(__file__).resolve().parent.parent / "sample" / "DuIE2.0" / "duie_train.json" / "duie_train.json"
    )
    extractor = CRFNERExtractor(model_path=model_path)
    relation_extractor = CRFRelationExtractor(model_path=relation_model_path)

    if retrain or not os.path.exists(model_path):
        extractor.train(duie_train_path, limit=train_limit, epochs=train_epochs)
    else:
        extractor.load()

    if retrain or not os.path.exists(relation_model_path):
        relation_extractor.train(duie_train_path, limit=train_limit, epochs=train_epochs)
    else:
        relation_extractor.load()

    entity_stats = {"resolved": 0, "new": 0}

    kg = KGBuilder(uri, user, password)
    resolver = EntityResolver(kg)

    try:
        kg.ensure_disambiguation_schema()

        base_label = infer_label(base_entity)
        base_resolved = resolver.resolve(base_entity, base_label, source_text=base_entity)
        kg.upsert_entity_by_key(
            base_resolved["entity_key"],
            base_resolved["canonical_name"],
            base_label,
            norm_name=base_resolved["norm_name"],
            domain=base_resolved["domain"],
        )

        for link in one_hop_links:
            link_label = infer_label(link)
            resolved = resolver.resolve(link, link_label, source_text=base_entity)
            kg.upsert_entity_by_key(
                resolved["entity_key"],
                resolved["canonical_name"],
                link_label,
                norm_name=resolved["norm_name"],
                domain=resolved["domain"],
            )
            kg.add_relation_by_key(
                base_resolved["entity_key"],
                base_label,
                "HAS_ONE_HOP_LINK",
                resolved["entity_key"],
                link_label,
            )

        for key, value in properties.items():
            for triple in property_triples(base_entity, key, value):
                tail_label = triple["tail_label"]
                tail_value = triple["tail"]
                if tail_label == "Date":
                    tail_resolved = {
                        "entity_key": f"date_{re.sub(r'[^0-9A-Za-z]+', '_', tail_value)}",
                        "canonical_name": tail_value,
                        "norm_name": tail_value,
                        "domain": "entity",
                    }
                else:
                    tail_resolved = resolver.resolve(tail_value, tail_label, source_text=str(value))

                kg.upsert_entity_by_key(
                    tail_resolved["entity_key"],
                    tail_resolved["canonical_name"],
                    tail_label,
                    norm_name=tail_resolved["norm_name"],
                    domain=tail_resolved["domain"],
                )
                kg.add_relation_by_key(
                    base_resolved["entity_key"],
                    base_label,
                    triple["relation"],
                    tail_resolved["entity_key"],
                    tail_label,
                )

        for index, record in enumerate(records, start=1):
            source = record.get("source", "")
            target = record.get("target", "")
            relation = safe_relation_name(record.get("relation", "LINKED_TO"))
            context = record.get("context", "")
            if not source or not target:
                continue

            source_label = infer_label(source)
            target_label = infer_label(target)
            source_resolved = resolver.resolve(source, source_label, source_text=context)
            target_resolved = resolver.resolve(target, target_label, source_text=context)

            kg.upsert_entity_by_key(
                source_resolved["entity_key"],
                source_resolved["canonical_name"],
                source_label,
                norm_name=source_resolved["norm_name"],
                domain=source_resolved["domain"],
            )
            kg.upsert_entity_by_key(
                target_resolved["entity_key"],
                target_resolved["canonical_name"],
                target_label,
                norm_name=target_resolved["norm_name"],
                domain=target_resolved["domain"],
            )
            kg.add_relation_by_key(
                source_resolved["entity_key"],
                source_label,
                relation,
                target_resolved["entity_key"],
                target_label,
            )

            paragraph_name = f"turing_context_{index}"
            mentions = extractor.extract_entities(context)
            for mention_text, mention_type in mentions:
                mention_label = infer_label(mention_text) if mention_type == "ENT" else mention_type
                mention_resolved = resolver.resolve(mention_text, mention_label, source_text=context, context_terms=normalize_context_terms(context))
                kg.upsert_entity_by_key(
                    mention_resolved["entity_key"],
                    mention_resolved["canonical_name"],
                    mention_label,
                    norm_name=mention_resolved["norm_name"],
                    domain=mention_resolved["domain"],
                )
                kg.add_document_mention(paragraph_name, mention_resolved["entity_key"], mention_label)
                kg.update_entity_profile(
                    mention_resolved["entity_key"],
                    mention_label,
                    context_terms=normalize_context_terms(context),
                    domain=mention_resolved["domain"],
                )
                entity_stats[mention_resolved["decision"]] += 1

            print(f"已处理语料 {index}/{len(records)}，关系: {source} -> {relation} -> {target}，抽取到 {len(mentions)} 个上下文实体")

        for index, paragraph in enumerate(content_items, start=1):
            paragraph_text = str(paragraph).strip()
            if not paragraph_text:
                continue
            paragraph_name = f"turing_content_{index}"
            resolved_mentions = []
            mentions = extractor.extract_entities(paragraph_text)
            for mention_text, mention_type in mentions:
                mention_label = infer_label(mention_text) if mention_type == "ENT" else mention_type
                mention_resolved = resolver.resolve(mention_text, mention_label, source_text=paragraph_text, context_terms=normalize_context_terms(paragraph_text))
                kg.upsert_entity_by_key(
                    mention_resolved["entity_key"],
                    mention_resolved["canonical_name"],
                    mention_label,
                    norm_name=mention_resolved["norm_name"],
                    domain=mention_resolved["domain"],
                )
                kg.add_document_mention(paragraph_name, mention_resolved["entity_key"], mention_label)
                kg.update_entity_profile(
                    mention_resolved["entity_key"],
                    mention_label,
                    context_terms=normalize_context_terms(paragraph_text),
                    domain=mention_resolved["domain"],
                )
                entity_stats[mention_resolved["decision"]] += 1
                resolved_mentions.append(
                    {
                        "text": mention_text,
                        "label": mention_label,
                        "entity_key": mention_resolved["entity_key"],
                        "canonical_name": mention_resolved["canonical_name"],
                        "domain": mention_resolved["domain"],
                    }
                )

            relation_counter = 0
            seen_relations = set()
            mention_map = {}
            for mention in resolved_mentions:
                mention_map[mention["text"]] = mention

            predicted_relations = relation_extractor.extract_relations(
                paragraph_text,
                allowed_entities=list(mention_map.keys()),
            )
            for predicted in predicted_relations:
                head_entity = mention_map.get(predicted["head"])
                tail_entity = mention_map.get(predicted["tail"])
                if not head_entity or not tail_entity:
                    continue
                relation = safe_relation_name(predicted["relation"])
                relation_key = (head_entity["entity_key"], relation, tail_entity["entity_key"])
                if relation_key in seen_relations:
                    continue
                seen_relations.add(relation_key)

                kg.add_relation_by_key(
                    head_entity["entity_key"],
                    head_entity["label"],
                    relation,
                    tail_entity["entity_key"],
                    tail_entity["label"],
                )
                relation_counter += 1

            print(f"已处理内容段落 {index}/{len(content_items)}，抽取到 {len(mentions)} 个实体，预测到 {relation_counter} 条关系")

        print("图灵知识图谱构建完成")
        print(f"消歧统计: resolved={entity_stats['resolved']}, new={entity_stats['new']}")
    finally:
        kg.close()


def main():
    default_root = Path(__file__).resolve().parent.parent / "sample"
    default_profile = default_root / "艾伦·图灵_with_links.json"
    default_corpus = default_root / "艾伦·图灵_relational_corpus.jsonl"

    parser = argparse.ArgumentParser(description="Build the Alan Turing knowledge graph from sample JSON files.")
    parser.add_argument("--profile", default=str(default_profile), help="Path to 艾伦·图灵_with_links.json")
    parser.add_argument("--corpus", default=str(default_corpus), help="Path to 艾伦·图灵_relational_corpus.jsonl")
    parser.add_argument("--uri", default="bolt://localhost:7688", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j user")
    parser.add_argument("--password", default="your_new_password", help="Neo4j password")
    parser.add_argument("--model-path", default=None, help="Optional BiLSTM+CRF model path")
    parser.add_argument("--relation-model-path", default=None, help="Optional BiLSTM+CRF relation model path")
    parser.add_argument("--duie-train-path", default=None, help="Path to external DuIE training set used by NER and relation models")
    parser.add_argument("--train-limit", type=int, default=2000, help="Max DuIE samples for NER/RE training (smaller is faster)")
    parser.add_argument("--train-epochs", type=int, default=3, help="Training epochs for NER/RE models")
    parser.add_argument("--retrain", action="store_true", help="Retrain both BiLSTM+CRF NER and relation models before import")
    parser.add_argument("--limit-corpus", type=int, default=None, help="Limit the relation corpus rows for quick runs")
    parser.add_argument("--limit-content", type=int, default=None, help="Limit content paragraphs for quick runs")
    parser.add_argument("--dry-run", action="store_true", help="Only parse and summarize without writing to Neo4j")
    args = parser.parse_args()

    import_turing_kg(
        profile_path=args.profile,
        corpus_path=args.corpus,
        uri=args.uri,
        user=args.user,
        password=args.password,
        model_path=args.model_path,
        relation_model_path=args.relation_model_path,
        duie_train_path=args.duie_train_path,
        train_limit=args.train_limit,
        train_epochs=args.train_epochs,
        retrain=args.retrain,
        limit_corpus=args.limit_corpus,
        limit_content=args.limit_content,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
