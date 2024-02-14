# -*- coding: utf-8 -*-

import json

# 从demo_data_法律咨询.jsonl文件中读取数据
# 来源：https://github.com/PKU-YuanGroup/ChatLaw/blob/main/data/demo_data_%E6%B3%95%E5%BE%8B%E5%92%A8%E8%AF%A2.jsonl
data = []
with open('./data/demo_data_法律咨询.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line))

# 转换格式变成 gpt-3.5-turbo 微调所需的数据格式
formatted_data = []
for entry in data:
    meta_instruction = entry["meta_instruction"].replace("你一个名叫ChatLAW，由北京大学团队开发的人工智能助理：", "你一个人工智能法律助理：")
    messages = []
    messages.append({
        "role": "system",
        "content": meta_instruction
    })
    for chat in entry["chat"]:
        messages.append({
            "role": "user",
            "content": chat["咨询者"]
        })
        messages.append({
            "role": "assistant",
            "content": chat["ChatLAW"]
        })
    formatted_data.append({
        "messages": messages
    })

# 将结果写入到fine_tuning.jsonl文件中
with open('./data/fine_tuning.jsonl', 'w', encoding='utf-8') as file:
    for item in formatted_data:
        file.write(json.dumps(item, ensure_ascii=False))
        file.write('\n')
