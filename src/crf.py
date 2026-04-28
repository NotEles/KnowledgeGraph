import os
import json
import joblib
import sklearn_crfsuite
from sklearn_crfsuite import metrics



class CRFNERExtractor:
    def __init__(self, model_path="ner_model.pkl"):
        self.model_path = model_path
        self.model = None
        # 类别映射字典，可以根据 DuIE 的 schema 扩展
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
        return value

    def _get_bio_labels(self, text, spo_list):
        """将 DuIE 的单条数据转为 BIO 序列"""
        labels = ['O'] * len(text)
        for spo in spo_list:
            # 处理 Subject
            sub = self._normalize_value(spo.get('subject', ''))
            sub_type = self._normalize_value(spo.get('subject_type', '实体'))
            self._label_entity(text, labels, sub, self.type_map.get(sub_type, "ENT"))
            
            # 处理 Object (DuIE 的 object 可能是字典)
            obj_val = self._normalize_value(spo.get('object', {}))
            obj_type = self._normalize_value(spo.get('object_type', '实体'))
            self._label_entity(text, labels, obj_val, self.type_map.get(obj_type, "ENT"))
        return labels

    def _label_entity(self, text, labels, entity, tag):
        start_idx = text.find(entity)
        if start_idx != -1:
            if labels[start_idx] == 'O': # 避免重复覆盖
                labels[start_idx] = f'B-{tag}'
                for i in range(1, len(entity)):
                    labels[start_idx + i] = f'I-{tag}'
    def _word2features(self, sent, i):
        """核心特征工程：滑动窗口"""
        char = sent[i]
        features = {
            'bias': 1.0,
            'char': char,
            'char.isdigit()': char.isdigit(),
        }
        # 前文特征
        if i > 0:
            features.update({
                '-1:char': sent[i-1],
                '-1:char[:2]': sent[i-1:i+1], # Bi-gram
            })
        else:
            features['BOS'] = True

        # 后文特征
        if i < len(sent) - 1:
            features.update({
                '+1:char': sent[i+1],
                '+1:char[:2]': sent[i:i+2], # Bi-gram
            })
        else:
            features['EOS'] = True
        return features
    def train(self, json_file_path, limit=5000):
        """从 DuIE 文件训练模型"""
        X, y = [], []
        with open(json_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= limit: break
                data = json.loads(line)
                text = data['text']
                labels = self._get_bio_labels(text, data.get('spo_list', []))
                
                X.append([self._word2features(text, j) for j in range(len(text))])
                y.append(labels)
        
        self.model = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100)
        self.model.fit(X, y)
        joblib.dump(self.model, self.model_path)
        print(f"模型已保存至 {self.model_path}")

    def load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError("请先调用 train() 训练模型。")
        
    def extract_entities(self, text):
        """在线推理：输入纯文本，输出实体及其类别"""
        if not self.model: self.load()
        features = [self._word2features(text, j) for j in range(len(text))]
        prediction = self.model.predict([features])[0]
        
        # 结果解析逻辑：将 BIO 还原为 实体-类别 对
        entities = []
        current_ent = ""
        current_type = ""
        
        for char, tag in zip(text, prediction):
            if tag.startswith("B-"):
                if current_ent: entities.append((current_ent, current_type))
                current_ent = char
                current_type = tag.split("-")[1]
            elif tag.startswith("I-") and current_ent:
                current_ent += char
            else:
                if current_ent: entities.append((current_ent, current_type))
                current_ent = ""
        return list(set(entities)) # 去重

if __name__ == "__main__":
    extractor = CRFNERExtractor()
    extractor.train("./KnowledgeGraph/sample/DuIE2.0/duie_train.json/duie_train.json") # 第一次运行需训练
    #extractor.load() # 之后直接加载模型进行推理   
    res = extractor.extract_entities("林俊杰考上了武汉大学。")
    print(res) # [('林俊杰', 'PER'), ('武汉大学', 'ORG')]