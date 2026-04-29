import requests
from bs4 import BeautifulSoup
import re
import json
import urllib.parse
import os

def scrape_wikipedia_with_links(entity_name):
    encoded_entity = urllib.parse.quote(entity_name)
    url = f"https://zh.wikipedia.org/zh-cn/{encoded_entity}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept-Language': 'zh-CN,zh;q=0.9'
    }


    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"请求失败: {e}")
        return
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 1. 提取 Infobox (保持不变)
    infobox_data = {}
    infobox = soup.select_one('table.infobox, table.vcard, table.biography.vcard')
    if infobox:
        for row in infobox.find_all('tr'):
            th = row.find(['th', 'td'], class_=re.compile(r'label|fn'))
            td = row.find(['td', 'div'], class_=re.compile(r'data|nickname'))
            if not (th and td):
                th = row.find('th')
                td = row.find('td')
            if th and td:
                key = th.get_text(strip=True)
                value = re.sub(r'\[.*?\]', '', td.get_text(separator=' ', strip=True))
                if key and value:
                    infobox_data[key] = value

    # 2. 提取正文 与 1跳超链接
    paragraphs = []
    one_hop_nodes = set()
    
    # 直接找所有维基词条共有的、最外层的核心内容 ID
    content_area = soup.find(id='mw-content-text')
    
    if content_area:
        # 直接遍历该区域内的所有段落
        all_p_tags = content_area.find_all('p')
        print(f"-> 探针报告: 在正文区共发现 {len(all_p_tags)} 个 <p> 标签。")
        
        for p in all_p_tags:
            # 避开信息框、导航模板、参考资料列表等非正文区域
            if p.find_parent(['table', 'th', 'td']) or p.find_parent('div', class_=re.compile(r'navbox|infobox|metadata|reflist')):
                continue
                
            text = p.get_text(strip=True)
            
            # 放宽长度限制，只要大于10个字符就算一段
            if len(text) > 10:
                clean_text = re.sub(r'\[\d+\]|\[来源请求\]|\[注 \d+\]|\[.*?\]', '', text)
                paragraphs.append(clean_text)
                
                # 提取链接
                for a_tag in p.find_all('a', href=True):
                    href = a_tag['href']
                    # 匹配所有中文维基前缀 (如 /wiki/, /zh-cn/, /zh-tw/ 等)
                    if href.startswith('/wiki/') or href.startswith('/zh-'):
                        if ':' in href:
                            continue
                        target_entity = urllib.parse.unquote(href.split('/')[-1])
                        if target_entity and target_entity != entity_name:
                            one_hop_nodes.add(target_entity)
    else:
        print("-> 探针警告: 未找到 id='mw-content-text' 的区域！")

    # 3. 汇总并保存
    result = {
        "entity": entity_name,
        "url": url,
        "properties": infobox_data,
        "content": paragraphs,
        "one_hop_links": list(one_hop_nodes) # 转换为列表以便 JSON 序列化
    }

    file_path = f"{entity_name}_with_links.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"✅ 抓取完成！")
    print(f"   - 提取到属性数量: {len(infobox_data)}")
    print(f"   - 提取到正文段落: {len(paragraphs)}")
    print(f"   - 提取到 1跳 关联实体: {len(one_hop_nodes)} 个")
    print(f"   - 文件已保存至: {os.path.abspath(file_path)}")

if __name__ == "__main__":
    scrape_wikipedia_with_links("艾伦·图灵")