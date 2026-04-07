from neo4j import GraphDatabase
# 默认的密码与账号
user = "neo4j"
password = "password"
uri = "bolt://localhost:7687"

class KGBuilder:
    def __init__(self, uri=uri, user=user, password=password):
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
