# KnowledgeGraph
- 这暂时是知识工程课程的作业代码库
- 配套使用了neo4j的数据库
## 目前进度
- \[✅\]连接数据库与数据库的基本操作
- \[✅\]结构化的实体与关系识别
- \[  \]半结构化的实体与关系识别
- \[  \]实体消歧

## 新增管线
- `src/bilstm_crf.py` 使用 BiLSTM+CRF 处理非结构化文本实体抽取，`src/crf.py` 作为兼容入口保留原调用方式。
- `src/relation_lstm_crf.py` 使用 BiLSTM+CRF 做关系抽取序列标注，训练数据来自外源 DuIE `spo_list`。
- `src/build_turing_kg.py` 会读取 `sample/艾伦·图灵_with_links.json` 和 `sample/艾伦·图灵_relational_corpus.jsonl`，并从 DuIE 训练集统一训练/加载实体与关系两个 BiLSTM+CRF 模型，再把半结构化属性、正文关系和内容段落一起写入 Neo4j。
