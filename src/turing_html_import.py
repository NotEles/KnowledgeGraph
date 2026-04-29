import argparse
import re
from pathlib import Path

from bs4 import BeautifulSoup

from entity_resolver import EntityResolver
from graph_builder import KGBuilder


class TuringHTMLExtractor:
    """Extract entities and relations from the Alan Turing Wikipedia HTML dump."""

    INFOBOX_RELATIONS = {
        "出生": ("BORN_IN", "Location"),
        "逝世": ("DIED_IN", "Location"),
        "死因": ("CAUSE_OF_DEATH", "Concept"),
        "墓地": ("BURIED_AT", "Location"),
        "居住地": ("RESIDED_IN", "Location"),
        "国籍": ("NATIONALITY", "Country"),
        "母校": ("ALMA_MATER", "Organization"),
        "知名于": ("KNOWN_FOR", "Concept"),
        "研究领域": ("FIELD_OF", "Concept"),
        "机构": ("AFFILIATED_WITH", "Organization"),
        "论文": ("AUTHORED", "Work"),
        "博士导师": ("ADVISED_BY", "Person"),
        "博士生": ("HAS_DOCTORAL_STUDENT", "Person"),
        "运动": ("SPORT", "Concept"),
        "项目": ("HAS_EVENT", "Concept"),
    }

    BODY_RELATION_PATTERNS = [
        (re.compile(r"与([^，。！？]+?)一起"), "COLLABORATED_WITH", "Person"),
        (re.compile(r"被授予([^，。！？]+?)"), "AWARDED", "Concept"),
        (re.compile(r"选为([^，。！？]+?)"), "ELECTED_TO", "Organization"),
        (re.compile(r"成为([^，。！？]+?)的(副主任|成员|研究员|教授)"), "WORKED_AT", "Organization"),
        (re.compile(r"提出了(?:一个叫做)?([^，。！？]+?)"), "PROPOSED", "Concept"),
        (re.compile(r"负责([^，。！？]+?)"), "WORKED_ON", "Organization"),
    ]

    ENTITY_HINTS = {
        "大学": "Organization",
        "学院": "Organization",
        "研究所": "Organization",
        "实验室": "Organization",
        "庄园": "Location",
        "公园": "Location",
        "城市": "Location",
        "国家": "Country",
        "英国": "Country",
        "法国": "Country",
        "德国": "Country",
        "奖": "Award",
        "机": "Work",
        "测试": "Concept",
        "理论": "Concept",
        "论文": "Work",
        "程序": "Work",
        "运动": "Concept",
        "图灵": "Person",
    }

    def __init__(self, html_path, model_path=None):
        self.html_path = Path(html_path)
        self.model_path = Path(model_path) if model_path else None
        self.soup = None

    @staticmethod
    def _clean_text(value):
        if value is None:
            return ""
        value = re.sub(r"\s+", " ", str(value))
        return value.strip()

    @staticmethod
    def _split_terms(text):
        if not text:
            return []
        parts = re.split(r"[、，,；;\s]+", text)
        return [part.strip() for part in parts if part.strip()]

    def _load(self):
        if self.soup is None:
            self.soup = BeautifulSoup(self.html_path.read_text(encoding="utf-8"), "html.parser")
        return self.soup

    def page_title(self):
        soup = self._load()
        header = soup.select_one("table th[colspan='2'] span")
        if header:
            return self._clean_text(header.get_text())

        title_tag = soup.find("title")
        if title_tag:
            title_text = self._clean_text(title_tag.get_text())
            return title_text.split(" - ", 1)[0]

        return "Alan Turing"

    def extract_infobox_rows(self):
        soup = self._load()
        table = soup.find("table")
        if not table:
            return []

        rows = []
        for row in table.find_all("tr"):
            header = row.find("th")
            cell = row.find("td")
            if not header or not cell:
                continue
            label = self._clean_text(header.get_text(" ", strip=True))
            if label not in self.INFOBOX_RELATIONS:
                continue

            relation, tail_label = self.INFOBOX_RELATIONS[label]
            values = [self._clean_text(anchor.get_text(" ", strip=True)) for anchor in cell.find_all("a")]
            values = [value for value in values if value]

            if not values:
                raw_text = self._clean_text(cell.get_text(" ", strip=True))
                values = self._split_terms(raw_text)

            for value in values:
                rows.append(
                    {
                        "head": self.page_title(),
                        "head_label": "Person",
                        "relation": relation,
                        "tail": value,
                        "tail_label": tail_label,
                        "source": f"infobox:{label}",
                    }
                )
        return rows

    def extract_mentions(self):
        soup = self._load()
        mentions = []
        seen = set()

        lead_paragraph = soup.select_one("#mw-content-text p")
        if lead_paragraph:
            lead_text = self._clean_text(lead_paragraph.get_text(" ", strip=True))
            alias_patterns = [
                r"又译([^，。]+)",
                r"也常翻译成([^，。]+)",
                r"英语：([^，。]+)",
            ]
            for pattern in alias_patterns:
                match = re.search(pattern, lead_text)
                if match:
                    for alias in self._split_terms(match.group(1)):
                        key = (alias, "Person")
                        if key in seen:
                            continue
                        seen.add(key)
                        mentions.append(
                            {
                                "text": alias,
                                "label": "Person",
                                "context": lead_text,
                                "source": "lead_alias",
                            }
                        )

        for node in soup.select("#mw-content-text p, #mw-content-text li, #mw-content-text blockquote"):
            context = self._clean_text(node.get_text(" ", strip=True))
            if not context:
                continue
            for anchor in node.find_all("a"):
                text = self._clean_text(anchor.get_text(" ", strip=True))
                if not text or len(text) < 2:
                    continue
                label = self._infer_label(text, context)
                key = (text, label)
                if key in seen:
                    continue
                seen.add(key)
                mentions.append(
                    {
                        "text": text,
                        "label": label,
                        "context": context,
                        "source": "body_anchor",
                    }
                )

        return mentions

    def extract_body_relations(self):
        soup = self._load()
        relations = []
        head = self.page_title()

        for node in soup.select("#mw-content-text p, #mw-content-text li, #mw-content-text blockquote"):
            sentence_text = self._clean_text(node.get_text(" ", strip=True))
            if not sentence_text:
                continue

            anchors = [self._clean_text(anchor.get_text(" ", strip=True)) for anchor in node.find_all("a")]
            anchors = [anchor for anchor in anchors if anchor and anchor != head]

            for pattern, relation, tail_label in self.BODY_RELATION_PATTERNS:
                match = pattern.search(sentence_text)
                if not match:
                    continue
                tail_text = self._clean_text(match.group(1))
                if tail_text:
                    relations.append(
                        {
                            "head": head,
                            "head_label": "Person",
                            "relation": relation,
                            "tail": tail_text,
                            "tail_label": tail_label,
                            "source": sentence_text,
                        }
                    )

            if "与" in sentence_text and "一起" in sentence_text and anchors:
                for anchor in anchors:
                    relations.append(
                        {
                            "head": head,
                            "head_label": "Person",
                            "relation": "COLLABORATED_WITH",
                            "tail": anchor,
                            "tail_label": self._infer_label(anchor, sentence_text),
                            "source": sentence_text,
                        }
                    )
                    break

            if "被授予" in sentence_text or "授予" in sentence_text:
                for anchor in anchors:
                    if "奖" in anchor or "勋章" in anchor or "学会" in anchor or "银行" in anchor:
                        relations.append(
                            {
                                "head": head,
                                "head_label": "Person",
                                "relation": "AWARDED",
                                "tail": anchor,
                                "tail_label": self._infer_label(anchor, sentence_text),
                                "source": sentence_text,
                            }
                        )
                        break

        return relations

    def _infer_label(self, text, context=""):
        for hint, label in self.ENTITY_HINTS.items():
            if hint in text or hint in context:
                return label
        if "大学" in text or "学院" in text or "实验室" in text or "研究所" in text:
            return "Organization"
        if re.search(r"\d{4}年|\d+小时|\d+分钟", text):
            return "Concept"
        return "Person"


def import_turing_html(
    html_path,
    uri="bolt://localhost:7688",
    user="neo4j",
    password="your_new_password",
    model_path=None,
):
    extractor = TuringHTMLExtractor(html_path, model_path=model_path)
    kg = KGBuilder(uri, user, password)
    resolver = EntityResolver(kg)

    try:
        kg.ensure_disambiguation_schema()

        document_name = extractor.html_path.stem
        mention_counter = 0
        relation_counter = 0

        mentions = extractor.extract_mentions()
        infobox_rows = extractor.extract_infobox_rows()
        body_relations = extractor.extract_body_relations()

        for mention in mentions:
            resolved = resolver.resolve(
                mention["text"],
                mention["label"],
                source_text=mention.get("context", ""),
                context_terms=mention.get("context", "").split(),
            )
            kg.upsert_entity_by_key(
                resolved["entity_key"],
                resolved["canonical_name"],
                mention["label"],
                norm_name=resolved["norm_name"],
                domain=resolved["domain"],
            )
            kg.add_document_mention(document_name, resolved["entity_key"], mention["label"])
            kg.update_entity_profile(
                resolved["entity_key"],
                mention["label"],
                context_terms=mention.get("context", "").split(),
                domain=resolved["domain"],
            )
            mention_counter += 1

        for triple in infobox_rows + body_relations:
            head_resolved = resolver.resolve(
                triple["head"],
                triple["head_label"],
                source_text=triple.get("source", ""),
            )
            tail_resolved = resolver.resolve(
                triple["tail"],
                triple["tail_label"],
                source_text=triple.get("source", ""),
            )

            kg.upsert_entity_by_key(
                head_resolved["entity_key"],
                head_resolved["canonical_name"],
                triple["head_label"],
                norm_name=head_resolved["norm_name"],
                domain=head_resolved["domain"],
            )
            kg.upsert_entity_by_key(
                tail_resolved["entity_key"],
                tail_resolved["canonical_name"],
                triple["tail_label"],
                norm_name=tail_resolved["norm_name"],
                domain=tail_resolved["domain"],
            )
            kg.add_relation_by_key(
                head_resolved["entity_key"],
                triple["head_label"],
                triple["relation"],
                tail_resolved["entity_key"],
                triple["tail_label"],
            )
            relation_counter += 1

        print(f"文档: {document_name}")
        print(f"实体写入/对齐: {mention_counter}")
        print(f"关系写入: {relation_counter}")
    finally:
        kg.close()


def main():
    default_html = Path(__file__).resolve().parent.parent / "sample" / "艾伦·图灵 - 维基百科，自由的百科全书.html"
    parser = argparse.ArgumentParser(description="Build a Turing knowledge graph from the Wikipedia HTML dump.")
    parser.add_argument("--html", default=str(default_html), help="Path to the Alan Turing HTML file")
    parser.add_argument("--uri", default="bolt://localhost:7688", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j user")
    parser.add_argument("--password", default="your_new_password", help="Neo4j password")
    parser.add_argument("--model-path", default=None, help="Optional CRF model path for extra body mentions")
    args = parser.parse_args()

    import_turing_html(
        html_path=args.html,
        uri=args.uri,
        user=args.user,
        password=args.password,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()