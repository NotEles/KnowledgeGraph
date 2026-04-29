import json
import os
import random

import torch

from bilstm_crf import BiLSTMCRF, START_TAG, STOP_TAG, PAD_TOKEN, UNK_TOKEN


class CRFRelationExtractor:
    def __init__(
        self,
        model_path="relation_lstm_crf.pt",
        embedding_dim=128,
        hidden_dim=128,
        device=None,
        use_amp=True,
        compile_model=False,
    ):
        self.model_path = model_path
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_amp = use_amp
        self.compile_model = compile_model
        self.model = None
        self.vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.tag_to_ix = {}
        self.ix_to_tag = {}

    @staticmethod
    def _normalize_value(value):
        if isinstance(value, dict):
            if "@value" in value:
                return str(value["@value"])
            for item in value.values():
                if isinstance(item, str):
                    return item
            return ""
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _safe_relation(relation):
        relation = str(relation).strip()
        if not relation:
            return "REL"
        return relation.replace("\n", " ").replace("\t", " ")

    @staticmethod
    def _find_spans(text, entity):
        if not text or not entity:
            return []
        spans = []
        start = 0
        while True:
            index = text.find(entity, start)
            if index == -1:
                break
            spans.append((index, index + len(entity)))
            start = index + len(entity)
        return spans

    @staticmethod
    def _load_records(data_file_path, limit=None):
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"找不到关系训练文件: {data_file_path}")

        with open(data_file_path, "r", encoding="utf-8") as handle:
            content = handle.read().strip()

        if not content:
            return []

        if content.startswith("["):
            records = json.loads(content)
        else:
            records = [json.loads(line) for line in content.splitlines() if line.strip()]

        if limit is not None:
            records = records[:limit]
        return records

    def _label_span(self, labels, text, entity, role, relation):
        tag_prefix = f"{role}::{relation}"
        for start, end in self._find_spans(text, entity):
            if labels[start] != "O":
                continue
            labels[start] = f"B-{tag_prefix}"
            for index in range(start + 1, end):
                if labels[index] == "O":
                    labels[index] = f"I-{tag_prefix}"
            return

    def _build_labels(self, text, spo_list):
        labels = ["O"] * len(text)
        for spo in spo_list:
            relation = self._safe_relation(self._normalize_value(spo.get("predicate", "REL")))
            subject = self._normalize_value(spo.get("subject", ""))
            obj = self._normalize_value(spo.get("object", ""))
            if not subject or not obj:
                continue
            self._label_span(labels, text, subject, "SUBJ", relation)
            self._label_span(labels, text, obj, "OBJ", relation)
        return labels

    def _records_to_examples(self, records):
        examples = []
        for record in records:
            text = self._normalize_value(record.get("text", ""))
            spo_list = record.get("spo_list", [])
            if not text or not spo_list:
                continue
            labels = self._build_labels(text, spo_list)
            if all(label == "O" for label in labels):
                continue
            examples.append((text, labels))
        return examples

    def _build_vocab_and_tags(self, examples):
        chars = set()
        tags = {"O", START_TAG, STOP_TAG}
        for text, labels in examples:
            chars.update(text)
            tags.update(labels)

        vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for char in sorted(chars):
            if char not in vocab:
                vocab[char] = len(vocab)

        tag_to_ix = {tag: index for index, tag in enumerate(sorted(tags))}
        if START_TAG not in tag_to_ix:
            tag_to_ix[START_TAG] = len(tag_to_ix)
        if STOP_TAG not in tag_to_ix:
            tag_to_ix[STOP_TAG] = len(tag_to_ix)

        self.vocab = vocab
        self.tag_to_ix = tag_to_ix
        self.ix_to_tag = {index: tag for tag, index in tag_to_ix.items()}

    def _build_model(self):
        self.model = BiLSTMCRF(
            vocab_size=len(self.vocab),
            tag_to_ix=self.tag_to_ix,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        if self.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

    def _encode_text(self, text):
        return [self.vocab.get(char, self.vocab[UNK_TOKEN]) for char in text]

    def _encode_labels(self, labels):
        return [self.tag_to_ix[label] for label in labels]

    def train(self, data_file_path, limit=10000, epochs=8, learning_rate=0.001, seed=42):
        records = self._load_records(data_file_path, limit=limit)
        examples = self._records_to_examples(records)
        if not examples:
            raise ValueError("没有可用于训练关系模型的样本。")

        random.seed(seed)
        torch.manual_seed(seed)

        use_cuda = self.device.type == "cuda"
        if use_cuda:
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

        self._build_vocab_and_tags(examples)
        self._build_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        use_amp = use_cuda and self.use_amp
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        self.model.train()
        for epoch in range(epochs):
            random.shuffle(examples)
            total_loss = 0.0
            for text, labels in examples:
                sentence_ids = torch.tensor(self._encode_text(text), dtype=torch.long, device=self.device)
                label_ids = torch.tensor(self._encode_labels(labels), dtype=torch.long, device=self.device)
                self.model.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = self.model.neg_log_likelihood(sentence_ids, label_ids)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach().item())
            avg_loss = total_loss / max(len(examples), 1)
            print(f"[REL] Epoch {epoch + 1}/{epochs} - avg loss: {avg_loss:.4f}")

        self.save()

    def save(self):
        if self.model is None:
            return
        model_dir = os.path.dirname(self.model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "vocab": self.vocab,
            "tag_to_ix": self.tag_to_ix,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
        }
        torch.save(payload, self.model_path)

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("请先调用 train() 训练关系模型。")
        payload = torch.load(self.model_path, map_location=self.device)
        self.vocab = payload["vocab"]
        self.tag_to_ix = payload["tag_to_ix"]
        self.ix_to_tag = {index: tag for tag, index in self.tag_to_ix.items()}
        self.embedding_dim = payload.get("embedding_dim", self.embedding_dim)
        self.hidden_dim = payload.get("hidden_dim", self.hidden_dim)

        self._build_model()
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _decode_tag(tag):
        if tag == "O" or "-" not in tag:
            return None
        prefix, rest = tag.split("-", 1)
        if "::" not in rest:
            return None
        role, relation = rest.split("::", 1)
        return prefix, role, relation

    def extract_relations(self, text, allowed_entities=None):
        if self.model is None:
            self.load()
        if not text:
            return []

        sentence_ids = torch.tensor(self._encode_text(text), dtype=torch.long, device=self.device)
        with torch.no_grad():
            _, best_path = self.model(sentence_ids)

        tags = [self.ix_to_tag[index] for index in best_path]
        spans_by_relation = {}

        current_chars = []
        current_role = None
        current_relation = None

        def flush_current():
            nonlocal current_chars, current_role, current_relation
            if current_chars and current_role and current_relation:
                entity = "".join(current_chars)
                rel_slot = spans_by_relation.setdefault(current_relation, {"SUBJ": [], "OBJ": []})
                rel_slot[current_role].append(entity)
            current_chars = []
            current_role = None
            current_relation = None

        for char, tag in zip(text, tags):
            decoded = self._decode_tag(tag)
            if decoded is None:
                flush_current()
                continue

            prefix, role, relation = decoded
            if prefix == "B":
                flush_current()
                current_chars = [char]
                current_role = role
                current_relation = relation
            elif prefix == "I" and current_chars and role == current_role and relation == current_relation:
                current_chars.append(char)
            else:
                flush_current()

        flush_current()

        allow_set = set(allowed_entities or [])
        triples = []
        seen = set()
        for relation, slots in spans_by_relation.items():
            subjects = slots.get("SUBJ", [])
            objects = slots.get("OBJ", [])
            for subject in subjects:
                for obj in objects:
                    if subject == obj:
                        continue
                    if allow_set and (subject not in allow_set or obj not in allow_set):
                        continue
                    triple = (subject, relation, obj)
                    if triple in seen:
                        continue
                    seen.add(triple)
                    triples.append(
                        {
                            "head": subject,
                            "relation": relation,
                            "tail": obj,
                        }
                    )
        return triples
