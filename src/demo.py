from neo4j import GraphDatabase

class KGBuilder:
    def __init__(self, uri, user, password):
        # 初始化数据库连接
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

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

# 测试运行
if __name__ == "__main__":
    # 1. 连接数据库
    kg = KGBuilder("bolt://localhost:7687", "neo4j", "password")
    
    # 2. 模拟从你的“抽取模块”得到了以下三元组数据
    extracted_triples = [
        {"head": "深澄真", "head_label": "Person", "rel": "FOLLOWER_OF", "tail": "月读命", "tail_label": "Deity"},
        {"head": "巴", "head_label": "Person", "rel": "SERVANT_OF", "tail": "深澄真", "tail_label": "Person"},
        {"head": "澪", "head_label": "Person", "rel": "SERVANT_OF", "tail": "深澄真", "tail_label": "Person"},
        {"head": "深澄真", "head_label": "Person", "rel": "LIVES_IN", "tail": "亚空", "tail_label": "Location"}
    ]
    
    # 3. 遍历写入图数据库
    print("开始构建图谱...")
    for tri in extracted_triples:
        kg.add_relation(
            tri["head"], tri["head_label"], 
            tri["rel"], 
            tri["tail"], tri["tail_label"]
        )
        print(f"已插入: {tri['head']} -> {tri['rel']} -> {tri['tail']}")
        
    kg.close()
    print("构建完成！请去 http://localhost:7474 运行 MATCH (n) RETURN n 查看。")