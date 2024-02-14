# -*- coding: utf-8 -*-

import openai
openai.api_key = "your_openai_key"

completion = openai.ChatCompletion.create(
  model="ft:gpt-3.5-turbo-0613:personal::7rJlWrzp",
  messages=[
    {"role": "user", "content": "如果有人擅自破坏水库闸门，但没有造成重大损失，是否构成决水罪？"}
  ]
)

print(completion.choices[0].message)

