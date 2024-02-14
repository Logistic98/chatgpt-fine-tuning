# -*- coding: utf-8 -*-

import json
import tiktoken
import numpy as np
from collections import defaultdict

# 指定数据路径并打开JSONL文件
data_path = "./data/fine_tuning.jsonl"

# 加载数据集
with open(data_path) as f:
    dataset = [json.loads(line) for line in f]

# 通过检查示例数量和第一项来快速查看数据
print("示例数量:", len(dataset))
print("第一个示例:")
for message in dataset[0]["messages"]:
    print(message)

# 我们需要遍历所有不同的示例，确保格式正确，并符合Chat completions消息结构
format_errors = defaultdict(int)

for ex in dataset:
    if not isinstance(ex, dict):
        format_errors["data_type"] += 1
        continue

    messages = ex.get("messages", None)
    if not messages:
        format_errors["missing_messages_list"] += 1
        continue

    for message in messages:
        if "role" not in message or "content" not in message:
            format_errors["message_missing_key"] += 1

        if any(k not in ("role", "content", "name") for k in message):
            format_errors["message_unrecognized_key"] += 1

        if message.get("role", None) not in ("system", "user", "assistant"):
            format_errors["unrecognized_role"] += 1

        content = message.get("content", None)
        if not content or not isinstance(content, str):
            format_errors["missing_content"] += 1

    if not any(message.get("role", None) == "assistant" for message in messages):
        format_errors["example_missing_assistant_message"] += 1

if format_errors:
    print("发现错误:")
    for k, v in format_errors.items():
        print(f"{k}: {v}")
else:
    print("未发现错误")

# 除了消息的结构，我们还需要确保长度不超过4096个令牌限制

# 计数令牌功能
encoding = tiktoken.get_encoding("cl100k_base")


# 不精确！简化自https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens


def print_distribution(values, name):
    print(f"\n#### {name}的分布:")
    print(f"最小值 / 最大值: {min(values)}, {max(values)}")
    print(f"平均值 / 中位数: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

# 最后，我们可以在创建微调作业之前查看不同格式操作的结果：

# 警告和令牌计数
n_missing_system = 0
n_missing_user = 0
n_messages = []
convo_lens = []
assistant_message_lens = []

for ex in dataset:
    messages = ex["messages"]
    if not any(message["role"] == "system" for message in messages):
        n_missing_system += 1
    if not any(message["role"] == "user" for message in messages):
        n_missing_user += 1
    n_messages.append(len(messages))
    convo_lens.append(num_tokens_from_messages(messages))
    assistant_message_lens.append(num_assistant_tokens_from_messages(messages))

print("缺少系统消息的示例数量:", n_missing_system)
print("缺少用户消息的示例数量:", n_missing_user)
print_distribution(n_messages, "每个示例的消息数量")
print_distribution(convo_lens, "每个示例的总令牌数量")
print_distribution(assistant_message_lens, "每个示例的助理令牌数量")
n_too_long = sum(l > 4096 for l in convo_lens)
print(f"\n{n_too_long}个示例可能超过4096个令牌限制，微调期间将被截断")

# 定价和默认n_epochs估计
MAX_TOKENS_PER_EXAMPLE = 4096
MIN_TARGET_EXAMPLES = 100
MAX_TARGET_EXAMPLES = 25000
TARGET_EPOCHS = 3
MIN_EPOCHS = 1
MAX_EPOCHS = 25

n_epochs = TARGET_EPOCHS
n_train_examples = len(dataset)
if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
    n_epochs = min(MAX_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
    n_epochs = max(MIN_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)
n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)

print(f"数据集包含约{n_billing_tokens_in_dataset}个将在训练期间收费的令牌")
print(f"默认情况下，您将对此数据集进行{n_epochs}个时代的训练")
print(f"默认情况下，您将为约{n_epochs * n_billing_tokens_in_dataset}个令牌收费")
print("请参阅定价页面以估算总成本")