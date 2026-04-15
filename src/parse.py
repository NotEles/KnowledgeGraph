import json

def parse_duie_to_bio(json_line):
    """
    将单条 DuIE 数据转换为 (字符, 标签) 的列表。
    例如: [('张', 'B-PER'), ('三', 'I-PER'), ('出', 'O'), ...]
    """
    data = json.loads(json_line)
    text = data["text"]
    spo_list = data.get("spo_list", [])
    
    # 初始化所有字均为 'O' (Outside)
    labels = ['O'] * len(text)
    
    for spo in spo_list:
        # 提取主体和客体作为实体 (为了简化，这里不区分 PER/LOC 等细分类别，统称为 ENT)
        entities = [spo["subject"]]
        
        # DuIE 的 object 结构有时比较深，提取 @value
        obj = spo["object"]
        for key, val in obj.items():
            entities.append(val)
            
        for ent in entities:
            start_idx = text.find(ent)
            if start_idx != -1:
                # 标注 B-ENT (Begin)
                labels[start_idx] = 'B-ENT'
                # 标注 I-ENT (Inside)
                for i in range(1, len(ent)):
                    labels[start_idx + i] = 'I-ENT'
                    
    # 组合为 (字, 标签) 的形式
    return list(zip(list(text), labels))

class CRFNERExtractor:
    def __init__(self, model_path="ner_model.pkl"):
        self.model_path = model_path
        self.model = None
        # 类别映射字典，可以根据 DuIE 的 schema 扩展
        self.type_map = {"人物": "PER", "地点": "LOC", "机构": "ORG", "作品": "WORK"}

    def _get_bio_labels(self, text, spo_list):
        """将 DuIE 的单条数据转为 BIO 序列"""
        labels = ['O'] * len(text)
        for spo in spo_list:
            # 处理 Subject
            sub, sub_type = spo['subject'], spo['subject_type']
            self._label_entity(text, labels, sub, self.type_map.get(sub_type, "ENT"))
            
            # 处理 Object (DuIE 的 object 可能是字典)
            obj_val = spo['object']['@value']
            obj_type = spo.get('object_type', "实体")
            self._label_entity(text, labels, obj_val, self.type_map.get(obj_type, "ENT"))
        return labels

    def _label_entity(self, text, labels, entity, tag):
        start_idx = text.find(entity)
        if start_idx != -1:
            if labels[start_idx] == 'O': # 避免重复覆盖
                labels[start_idx] = f'B-{tag}'
                for i in range(1, len(entity)):
                    labels[start_idx + i] = f'I-{tag}'

# 测试一下
sample_json = '{"text": "李白出生于碎叶城", "spo_list": [{"subject": "李白", "predicate": "出生地", "object": {"@value": "碎叶城"}}]}'
print(parse_duie_to_bio(sample_json))
# 输出: [('李', 'B-ENT'), ('白', 'I-ENT'), ('出', 'O'), ('生', 'O'), ('于', 'O'), ('碎', 'B-ENT'), ('叶', 'I-ENT'), ('城', 'I-ENT')]