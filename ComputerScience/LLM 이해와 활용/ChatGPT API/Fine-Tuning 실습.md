
---
### Fine-Tuning 실습: JSONL 형식 데이터 준비 및 검증, Fine-Tuning 과정

#### 1. JSONL 학습 데이터 준비

Fine-Tuning에 사용될 데이터는 JSONL 형식으로 준비합니다. 각 예시에는 시스템 메시지(`role: system`), 사용자 메시지(`role: user`), 그리고 챗봇의 답변 메시지(`role: assistant`)가 포함되어야 합니다.

예시:

```json
{"messages": [{"role": "system", "content": "Marv는 사실적인 정보를 제공하면서도 냉소적인 챗봇입니다."}, {"role": "user", "content": "프랑스의 수도가 무엇인가요?"}, {"role": "assistant", "content": "파리죠, 마치 모르는 사람이 있을 것 같네요."}]}
{"messages": [{"role": "system", "content": "Marv는 사실적인 정보를 제공하면서도 냉소적인 챗봇입니다."}, {"role": "user", "content": "'로미오와 줄리엣'을 쓴 사람은 누구인가요?"}, {"role": "assistant", "content": "오, 그냥 윌리엄 셰익스피어라는 사람이죠. 들어보셨나요?"}]}
```

- `role`은 `system`, `user`, `assistant`로 지정되며, 각 메시지에 적절한 내용을 입력합니다.

#### 2. JSONL 데이터 검증 및 통계

**데이터 로드 및 초기 통계** 학습 데이터를 로드하고 첫 번째 예시를 출력하여 데이터가 정상적으로 입력되었는지 확인합니다.

```python
import json

data_path = "data/toy_chat_fine_tuning_15.jsonl"

# 데이터 로드
with open(data_path, 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

# 초기 통계 출력
print("Num examples:", len(dataset))
print("First example:")
for message in dataset[0]["messages"]:
    print(message)
```

**데이터 형식 오류 검증** 각 메시지에는 `role`과 `content`가 포함되어야 하며, 역할에 맞지 않는 `role` 값이나 기타 불필요한 키가 있는지 점검합니다.

```python
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
        if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
            format_errors["message_unrecognized_key"] += 1
        if message.get("role", None) not in ("system", "user", "assistant", "function"):
            format_errors["unrecognized_role"] += 1
        if not message.get("content", "") and not message.get("function_call", ""):
            format_errors["missing_content"] += 1

    if not any(message.get("role", None) == "assistant" for message in messages):
        format_errors["example_missing_assistant_message"] += 1

# 오류 출력
if format_errors:
    print("Found errors:")
    for k, v in format_errors.items():
        print(f"{k}: {v}")
else:
    print("No errors found")
```

**토큰 계산 및 분포 출력** 각 메시지와 예시의 토큰 수를 계산하여 모델이 처리할 수 있는 범위 내에 있는지 확인합니다. 특히, 모델의 입력 토큰 수는 제한이 있기 때문에 이 검사는 매우 중요합니다.

```python
encoding = tiktoken.get_encoding("cl100k_base")

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
    return sum(len(encoding.encode(message["content"])) for message in messages if message["role"] == "assistant")

def print_distribution(values, name):
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p10 / p90: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

# 예시별 통계 출력
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

print("Num examples missing system message:", n_missing_system)
print("Num examples missing user message:", n_missing_user)
print_distribution(n_messages, "num_messages_per_example")
print_distribution(convo_lens, "num_total_tokens_per_example")
print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
```

---

#### 3. Fine-Tuning 작업 생성

Fine-Tuning을 위해 학습 데이터를 OpenAI에 업로드하고, Fine-Tuning 작업을 생성합니다.

**JSONL 파일 업로드**

```python
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

client.files.create(
    file=open("./data/toy_chat_fine_tuning_15.jsonl", "rb"),
    purpose="fine-tune"
)
```

**Fine-Tuning 작업 생성** 업로드된 파일을 사용하여 Fine-Tuning 작업을 생성합니다. 모델과 하이퍼파라미터도 지정할 수 있습니다.

```python
job = client.fine_tuning.jobs.create(
    training_file="file-9CFJgV6uZHkbi3Np95opN3",  # 업로드된 파일 ID
    model="gpt-4o-2024-08-06",
    method={
        "type": "dpo",
        "dpo": {
            "hyperparameters": {"beta": 0.1},
        },
    },
)
```

---

#### 4. Fine-Tuning 작업 상태 확인

Fine-Tuning 작업이 완료되었는지 확인할 수 있습니다.

```python
client.fine_tuning.jobs.retrieve("ftjob-...")  # Fine-Tuning Job ID를 입력
```

---

#### 5. 파인 튜닝된 모델 사용

Fine-Tuning된 모델을 사용하여 응답을 생성할 수 있습니다.

```python
completion = client.chat.completions.create(
    model="fine_tuned_model",  # 파인 튜닝된 모델 ID
    messages=[
        {"role": "system", "content": "Marv는 사실적인 정보를 제공하면서도 냉소적인 챗봇입니다."},
        {"role": "user", "content": "인터넷 없이 살 수 있을까요?"}
    ]
)

print(completion.choices[0].message)
```

---

이 과정은 OpenAI의 API를 사용하여 모델을 Fine-Tuning하는 전반적인 흐름을 다룹니다. 학습 데이터의 형식을 검증하고, 토큰 수를 계산하여 제한을 초과하지 않도록 조정한 후, 실제로 Fine-Tuning 작업을 생성하고, 파인 튜닝된 모델을 활용하여 실험을 진행할 수 있습니다.