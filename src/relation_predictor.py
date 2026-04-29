import json
import os
import random
from dataclasses import dataclass

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


NO_RELATION = "NO_RELATION"


@dataclass
class RelationExample:
    text: str
    label: str


class RelationPredictor:
    def __init__(self, model_path="relation_model.joblib", threshold=0.45, random_seed=42):
        self.model_path = model_path
        self.threshold = threshold
        self.random_seed = random_seed
        self.pipeline = None

    @staticmethod
    def _normalize_value(value):
        if isinstance(value, dict):
            if "@value" in value:
                return value["@value"]
            for item in value.values():
                if isinstance(item, str):
                    return item
            return ""
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _load_records(data_file_path, limit=None):
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"找不到关系语料文件: {data_file_path}")

        with open(data_file_path, "r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]

        if limit is not None:
            records = records[:limit]
        return records

    @staticmethod
    def _replace_span(text, span_text, token):
        if not text or not span_text:
            return text, False
        index = text.find(span_text)
        if index == -1:
            return text, False
        end = index + len(span_text)
        marked = text[:index] + token + span_text + token.replace("[", "[/") + text[end:]
        return marked, True

    def build_text(self, context, head, tail):
        context = self._normalize_value(context)
        head = self._normalize_value(head)
        tail = self._normalize_value(tail)

        if not context:
            return f"HEAD={head} TAIL={tail}"

        if head == tail:
            return f"HEAD={head} CONTEXT={context} TAIL={tail}"

        if head and tail:
            head_index = context.find(head)
            tail_index = context.find(tail)
            if head_index != -1 and tail_index != -1:
                spans = sorted(
                    [(head_index, head, "HEAD"), (tail_index, tail, "TAIL")],
                    key=lambda item: item[0],
                    reverse=True,
                )
                marked = context
                for _, entity, token in spans:
                    placeholder = f"[{token}]"
                    suffix = f"[/{token}]"
                    position = marked.find(entity)
                    if position == -1:
                        continue
                    marked = marked[:position] + placeholder + entity + suffix + marked[position + len(entity):]
                return marked

        return f"HEAD={head} CONTEXT={context} TAIL={tail}"

    def _generate_examples(self, records, negative_ratio=1):
        positive_examples = []
        entities = []
        for record in records:
            source = self._normalize_value(record.get("source", ""))
            target = self._normalize_value(record.get("target", ""))
            context = self._normalize_value(record.get("context", ""))
            relation = self._normalize_value(record.get("relation", ""))
            if not (source and target and context and relation):
                continue
            positive_examples.append(RelationExample(self.build_text(context, source, target), relation))
            entities.extend([source, target])

        negatives = []
        if positive_examples:
            rng = random.Random(self.random_seed)
            pool = list(dict.fromkeys(entities))
            for record in records:
                source = self._normalize_value(record.get("source", ""))
                target = self._normalize_value(record.get("target", ""))
                context = self._normalize_value(record.get("context", ""))
                if not (source and target and context):
                    continue

                for _ in range(max(1, negative_ratio)):
                    candidates = [entity for entity in pool if entity not in {source, target}]
                    if not candidates:
                        continue
                    random_target = rng.choice(candidates)
                    negatives.append(RelationExample(self.build_text(context, source, random_target), NO_RELATION))
                    negatives.append(RelationExample(self.build_text(context, target, source), NO_RELATION))

        return positive_examples + negatives

    def _generate_examples_from_triples(self, triples, negative_ratio=1):
        positive_examples = []
        entities = []
        for triple in triples:
            head = self._normalize_value(triple.get("head", ""))
            tail = self._normalize_value(triple.get("tail", ""))
            relation = self._normalize_value(triple.get("relation", ""))
            context = self._normalize_value(triple.get("source", "")) or self._normalize_value(triple.get("context", ""))
            if not (head and tail and relation and context):
                continue
            positive_examples.append(RelationExample(self.build_text(context, head, tail), relation))
            entities.extend([head, tail])

        negatives = []
        if positive_examples:
            rng = random.Random(self.random_seed)
            pool = list(dict.fromkeys(entities))
            for triple in triples:
                head = self._normalize_value(triple.get("head", ""))
                tail = self._normalize_value(triple.get("tail", ""))
                context = self._normalize_value(triple.get("source", "")) or self._normalize_value(triple.get("context", ""))
                if not (head and tail and context):
                    continue

                for _ in range(max(1, negative_ratio)):
                    candidates = [entity for entity in pool if entity not in {head, tail}]
                    if not candidates:
                        continue
                    random_tail = rng.choice(candidates)
                    negatives.append(RelationExample(self.build_text(context, head, random_tail), NO_RELATION))
                    negatives.append(RelationExample(self.build_text(context, tail, head), NO_RELATION))

        return positive_examples + negatives

    def train(self, data_file_path, limit=None, negative_ratio=1):
        records = self._load_records(data_file_path, limit=limit)
        examples = self._generate_examples(records, negative_ratio=negative_ratio)
        if not examples:
            raise ValueError("没有可用于训练关系分类器的样本。")

        texts = [example.text for example in examples]
        labels = [example.label for example in examples]

        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(2, 4),
                        min_df=1,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        self.pipeline.fit(texts, labels)
        self.save()

    def train_from_triples(self, triples, negative_ratio=1):
        examples = self._generate_examples_from_triples(triples, negative_ratio=negative_ratio)
        if not examples:
            raise ValueError("没有可用于训练关系分类器的三元组。")

        texts = [example.text for example in examples]
        labels = [example.label for example in examples]

        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(2, 4),
                        min_df=1,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        self.pipeline.fit(texts, labels)
        self.save()

    def save(self):
        if self.pipeline is None:
            return
        model_dir = os.path.dirname(self.model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "threshold": self.threshold,
            },
            self.model_path,
        )

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("请先调用 train() 训练关系分类器。")
        payload = joblib.load(self.model_path)
        self.pipeline = payload["pipeline"]
        self.threshold = payload.get("threshold", self.threshold)

    def predict(self, context, head, tail):
        if self.pipeline is None:
            self.load()

        text = self.build_text(context, head, tail)
        if not text.strip():
            return None

        probabilities = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        best_index = int(probabilities.argmax())
        best_label = classes[best_index]
        best_score = float(probabilities[best_index])

        if best_label == NO_RELATION or best_score < self.threshold:
            return None

        return {
            "relation": best_label,
            "score": best_score,
        }
