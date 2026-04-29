import requests
from bs4 import BeautifulSoup
import re
import json
import urllib.parse
import time
import os

def extract_relational_context(base_entity_file):
    if not os.path.exists(base_entity_file):
        print(f"未找到文件: {base_entity_file}")
        return

    with open(base_entity_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    base_entity = data['entity']
    one_hop_nodes = set(data.get('one_hop_links', []))
    # 将中心节点也加入白名单，因为 1 跳页面经常会回指向“艾伦·图灵”
    entity_whitelist = one_hop_nodes | {base_entity}
    
    print(f"🚀 开始为 {len(one_hop_nodes)} 个节点提取关系上下文...")

    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'}

    output_file = f"{base_entity}_relational_corpus.jsonl"
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for i, current_node in enumerate(list(one_hop_nodes), 1):
            print(f"[{i}/{len(one_hop_nodes)}] 正在分析页面: {current_node}")
            
            url = f"https://zh.wikipedia.org/zh-cn/{urllib.parse.quote(current_node)}"
            
            try:
                time.sleep(1.2) # 礼貌抓取
                response = requests.get(url, headers=headers, timeout=15)
                response.encoding = 'utf-8'
                soup = BeautifulSoup(response.text, 'html.parser')
                content_area = soup.find(id='mw-content-text')
                
                if not content_area: continue

                # 遍历所有段落，寻找包含白名单实体的段落
                for p in content_area.find_all('p'):
                    p_text = p.get_text(strip=True)
                    if len(p_text) < 15: continue
                    
                    # 检查该段落中所有的链接
                    for a_tag in p.find_all('a', href=True):
                        href = a_tag['href']
                        if href.startswith('/wiki/') or href.startswith('/zh-'):
                            if ':' in href: continue
                            target_entity = urllib.parse.unquote(href.split('/')[-1])
                            
                            # 如果链接的目标在我们关注的实体白名单中
                            if target_entity in entity_whitelist and target_entity != current_node:
                                # 清理引用标记
                                clean_context = re.sub(r'\[\d+\]', '', p_text)
                                
                                # 构建关系元组
                                record = {
                                    "source": current_node,
                                    "target": target_entity,
                                    "context": clean_context,
                                    "url": url
                                }
                                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                                
            except Exception as e:
                print(f"跳过 {current_node}: {e}")

    print(f"\n✅ 语料提取完成！文件已保存至: {output_file}")

if __name__ == "__main__":
    extract_relational_context("艾伦·图灵_with_links.json")