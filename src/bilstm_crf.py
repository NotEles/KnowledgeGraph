import json
import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader


START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass
class TrainingExample:
    text: str
    labels: list


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim=128, hidden_dim=128):
        super().__init__()
        if hidden_dim % 2 != 0:
            hidden_dim += 1

        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000.0
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000.0

    @staticmethod
    def _log_sum_exp(tensor):
        max_score = tensor.max()
        return max_score + torch.log(torch.sum(torch.exp(tensor - max_score)))

    def _get_lstm_features(self, sentence_ids, lengths=None):
        single_input = sentence_ids.dim() == 1
        if single_input:
            sentence_ids = sentence_ids.unsqueeze(0)

        embeds = self.embedding(sentence_ids)
        if lengths is not None:
            packed = pack_padded_sequence(
                embeds,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=sentence_ids.size(1),
            )
        else:
            lstm_out, _ = self.lstm(embeds)

        feats = self.hidden2tag(lstm_out)
        if single_input and lengths is None:
            return feats.squeeze(0)
        return feats

    def _forward_alg(self, feats):
        init_alphas = torch.full((self.tagset_size,), -10000.0, device=feats.device)
        init_alphas[self.tag_to_ix[START_TAG]] = 0.0

        forward_var = init_alphas
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1)
                trans_score = self.transitions[next_tag]
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(self._log_sum_exp(next_tag_var))
            forward_var = torch.stack(alphas_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = self._log_sum_exp(terminal_var)
        return alpha

    def _forward_alg_batch(self, feats, lengths):
        batch_size, max_len, _ = feats.shape
        init_alphas = torch.full((batch_size, self.tagset_size), -10000.0, device=feats.device)
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.0

        forward_var = init_alphas
        for step in range(max_len):
            active_mask = (step < lengths).unsqueeze(1)
            if not torch.any(active_mask):
                break

            feat = feats[:, step]
            score = forward_var.unsqueeze(1) + self.transitions.unsqueeze(0)
            score = score + feat.unsqueeze(2)
            next_forward_var = torch.logsumexp(score, dim=2)
            forward_var = torch.where(active_mask, next_forward_var, forward_var)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]].unsqueeze(0)
        return torch.logsumexp(terminal_var, dim=1)

    def _score_sentence(self, feats, tags):
        score = torch.tensor(0.0, device=feats.device)
        tags = torch.cat(
            [
                torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long, device=feats.device),
                tags,
            ]
        )
        for index, feat in enumerate(feats):
            score = score + self.transitions[tags[index + 1], tags[index]] + feat[tags[index + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _score_sentence_batch(self, feats, tags, lengths):
        batch_size, max_len, _ = feats.shape
        start_tags = torch.full((batch_size, 1), self.tag_to_ix[START_TAG], dtype=torch.long, device=feats.device)
        prev_tags = torch.cat([start_tags, tags[:, :-1]], dim=1)
        current_tags = tags

        emission_scores = feats.gather(2, current_tags.unsqueeze(-1)).squeeze(-1)
        transition_scores = self.transitions[current_tags, prev_tags]

        mask = torch.arange(max_len, device=feats.device).unsqueeze(0) < lengths.unsqueeze(1)
        score = ((emission_scores + transition_scores) * mask).sum(dim=1)

        last_tag_indices = lengths - 1
        last_tags = tags.gather(1, last_tag_indices.unsqueeze(1)).squeeze(1)
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], last_tags]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        init_vvars = torch.full((self.tagset_size,), -10000.0, device=feats.device)
        init_vvars[self.tag_to_ix[START_TAG]] = 0.0
        forward_var = init_vvars

        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = torch.argmax(next_tag_var).item()
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[best_tag_id].view(1) + feat[next_tag].view(1))
            forward_var = torch.cat(viterbivars_t)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = torch.argmax(terminal_var).item()
        path_score = terminal_var[best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        best_path.pop()
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence_ids, tags):
        feats = self._get_lstm_features(sentence_ids)
        return self._neg_log_likelihood_from_feats(feats, tags)

    def _neg_log_likelihood_from_feats(self, feats, tags):
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def neg_log_likelihood_batch(self, sentence_ids, tags, lengths):
        feats = self._get_lstm_features(sentence_ids, lengths=lengths)
        forward_score = self._forward_alg_batch(feats, lengths)
        gold_score = self._score_sentence_batch(feats, tags, lengths)
        return (forward_score - gold_score).mean()

    def forward(self, sentence_ids):
        feats = self._get_lstm_features(sentence_ids)
        score, tag_seq = self._viterbi_decode(feats)
        return score, tag_seq


class CRFNERExtractor:
    def __init__(
        self,
        model_path="ner_model.pt",
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
        self.type_map = {"人物": "PER", "地点": "LOC", "机构": "ORG", "作品": "WORK"}

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

    def _label_entity(self, labels, text, entity, tag):
        for start, end in self._find_spans(text, entity):
            if labels[start] != "O":
                continue
            labels[start] = f"B-{tag}"
            for index in range(start + 1, end):
                if labels[index] == "O":
                    labels[index] = f"I-{tag}"

    def _get_bio_labels(self, text, spo_list):
        labels = ["O"] * len(text)
        for spo in spo_list:
            subject = self._normalize_value(spo.get("subject", ""))
            subject_type = self._normalize_value(spo.get("subject_type", "实体"))
            self._label_entity(labels, text, subject, self.type_map.get(subject_type, "ENT"))

            obj_value = self._normalize_value(spo.get("object", {}))
            obj_type = self._normalize_value(spo.get("object_type", "实体"))
            self._label_entity(labels, text, obj_value, self.type_map.get(obj_type, "ENT"))
        return labels

    def _records_to_examples(self, records):
        examples = []
        for record in records:
            text = record.get("text") or record.get("context") or ""
            if not text:
                continue

            labels = ["O"] * len(text)
            if record.get("spo_list"):
                labels = self._get_bio_labels(text, record.get("spo_list", []))
            elif record.get("source") or record.get("target"):
                for entity in (record.get("source", ""), record.get("target", "")):
                    entity = self._normalize_value(entity)
                    self._label_entity(labels, text, entity, "ENT")
            else:
                continue

            examples.append(TrainingExample(text=text, labels=labels))
        return examples

    @staticmethod
    def _load_records(data_file_path, limit=None):
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"找不到训练文件: {data_file_path}")

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

    def _build_vocab_and_tags(self, examples):
        chars = set()
        tags = {"O", START_TAG, STOP_TAG}
        for example in examples:
            chars.update(example.text)
            tags.update(example.labels)

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

    def _collate_examples(self, batch):
        sentence_ids = [torch.tensor(self._encode_text(example.text), dtype=torch.long) for example in batch]
        label_ids = [torch.tensor(self._encode_labels(example.labels), dtype=torch.long) for example in batch]
        lengths = torch.tensor([item.size(0) for item in sentence_ids], dtype=torch.long)
        padded_sentences = pad_sequence(sentence_ids, batch_first=True, padding_value=self.vocab[PAD_TOKEN])
        padded_labels = pad_sequence(label_ids, batch_first=True, padding_value=self.tag_to_ix[STOP_TAG])
        return padded_sentences, padded_labels, lengths

    def train(self, data_file_path, limit=5000, epochs=8, learning_rate=0.001, seed=42, batch_size=32, num_workers=0):
        records = self._load_records(data_file_path, limit=limit)
        examples = self._records_to_examples(records)
        if not examples:
            raise ValueError("没有可用于训练的样本。")

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
        train_loader = DataLoader(
            examples,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_cuda,
            collate_fn=self._collate_examples,
        )

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for sentence_ids, label_ids, lengths in train_loader:
                sentence_ids = sentence_ids.to(self.device, non_blocking=use_cuda)
                label_ids = label_ids.to(self.device, non_blocking=use_cuda)
                lengths = lengths.to(self.device, non_blocking=use_cuda)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = self.model.neg_log_likelihood_batch(sentence_ids, label_ids, lengths)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach().item()) * sentence_ids.size(0)

            avg_loss = total_loss / max(len(examples), 1)
            print(f"Epoch {epoch + 1}/{epochs} - avg loss: {avg_loss:.4f}")

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
            raise FileNotFoundError("请先调用 train() 训练模型。")

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

    def extract_entities(self, text):
        if self.model is None:
            self.load()

        if not text:
            return []

        sentence_ids = torch.tensor(self._encode_text(text), dtype=torch.long, device=self.device)
        with torch.no_grad():
            _, best_path = self.model(sentence_ids)

        tags = [self.ix_to_tag[index] for index in best_path]
        entities = []
        current_entity = []
        current_type = None

        for char, tag in zip(text, tags):
            if tag.startswith("B-"):
                if current_entity:
                    entities.append(("".join(current_entity), current_type or "ENT"))
                current_entity = [char]
                current_type = tag.split("-", 1)[1]
            elif tag.startswith("I-") and current_entity and tag.split("-", 1)[1] == current_type:
                current_entity.append(char)
            else:
                if current_entity:
                    entities.append(("".join(current_entity), current_type or "ENT"))
                current_entity = []
                current_type = None

        if current_entity:
            entities.append(("".join(current_entity), current_type or "ENT"))

        deduped = []
        seen = set()
        for entity in entities:
            if entity in seen:
                continue
            seen.add(entity)
            deduped.append(entity)
        return deduped
