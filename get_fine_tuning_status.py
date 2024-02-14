# -*- coding: utf-8 -*-

import openai
openai.api_key = "your_openai_key"

print("===列出10个微调作业")
print(openai.FineTuningJob.list(limit=10))

print("===检索微调作业的状态")
print(openai.FineTuningJob.retrieve("ftjob-5XsithSRiJ6mvf24IP9xq7eW"))

print("===列出最多10个来自微调作业的事件")
print(openai.FineTuningJob.list_events(id="ftjob-5XsithSRiJ6mvf24IP9xq7eW", limit=10))

# print("===取消作业")
# print(openai.FineTuningJob.cancel("ftjob-5XsithSRiJ6mvf24IP9xq7eW"))
#
# print("===删除经过微调的模型（必须是创建模型的组织的所有者）")
# print(openai.Model.delete("ftjob-5XsithSRiJ6mvf24IP9xq7eW"))