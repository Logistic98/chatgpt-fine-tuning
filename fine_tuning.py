# -*- coding: utf-8 -*-

# pip3 install --upgrade openai  旧版的openai没有FineTuningJob功能
import openai
openai.api_key = "your_openai_key"

# 创建微调模型
openai.FineTuningJob.create(training_file="file-LX7MRoSwB7je9yuC4FydIgV5", model="gpt-3.5-turbo")
