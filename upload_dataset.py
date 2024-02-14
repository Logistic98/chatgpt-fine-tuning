# -*- coding: utf-8 -*-

import openai
openai.api_key = "your_openai_key"

# 上传训练数据集
training_file = openai.File.create(
file=open("./data/fine_tuning.jsonl", "rb"),
purpose="fine-tune"
)

# file.id要复制下来，下一步开始微调要用
print(training_file.id)