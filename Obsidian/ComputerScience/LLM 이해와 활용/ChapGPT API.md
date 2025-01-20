
---
#### ChatGPT의 API 연동
- 요청 매개변수 설정 
	- model : 사용할 모델 지정. " gpt‐3.5‐turbo" 모델 사용 
	- content : ChatGPT에 전달할 입력 텍스트 
- API 응답 처리 
	- API 요청에 대한 응답은 JSON 형식으로 반환 됨 
	- 응답에서 `response.choices[0].content`를 사용하여 생성된 텍스트를 추출 
- 다양한 요청 및 응답 처리 
	- 다양한 유형의 대화와 작업을 수행 가능 
- 예외 처리 
	- API 요청 중에 발생할 수 있는 예외 상황을 처리하는 코드 추가 
- 보안 및 비용 관리 
	- API 키를 보안 유지하고, 사용량을 모니터링하고 비용을 관리 
- 테스트 및 디버깅 
	- 요청과 응답을 테스트하고 디버그하여 ChatGPT를 올바르게 통합
- OpenAI 계정 및 API 키 생성 
	- OpenAI 웹사이트(https://openai.com/blog/openai-api)에 가입하고 로그인 
	- 대시보드에서 API 키를 생성하고 해당 키를 안전한 곳에 보관
- 필요한 라이브러리 설치 
	- `pip install openai`
#### OpenAI API Endpoint 및 모델 호환성
- https://platform.openai.com/docs/models#model-endpoint-compatibility 
- post 요청 endpoint url → https://api.openai.com/v1/chat/completions

|**Endpoint**|**지원 모델**|
|---|---|
|`/v1/chat/completions`|GPT-4o (Realtime preview 제외), GPT-4o-mini, GPT-4, GPT-3.5 Turbo, 날짜가 명시된 릴리스 및 파인튜닝 버전들|
|`/v1/audio/transcriptions`|whisper-1|
|`/v1/audio/translations`|whisper-1|
|`/v1/audio/speech`|tts-1, tts-1-hd|
|`/v1/completions (Legacy)`|gpt-3.5-turbo-instruct, babbage-002, davinci-002|
|`/v1/embeddings`|text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002|
|`/v1/fine_tuning/jobs`|gpt-4o, gpt-4o-mini, gpt-4, gpt-3.5-turbo|
|`/v1/moderations`|text-moderation-stable, text-moderation-latest|
|`/v1/images/generations`|dall-e-2, dall-e-3|
|`/v1/realtime (beta)`|gpt-4o-realtime-preview, gpt-4o-realtime-preview-2024-10-01|

---
#### POST 요청 Assistants API 실습
- POST 요청을 https://api.openai.com/v1/assistants 엔드포인트에 보내는 것은 주로 새로 운 AI 어시스턴트를 생성하는 데 사용됩니다. 
- 어시스턴트는 다양한 기능을 수행할 수 있지만, 그 기능은 사용자가 설정하고 정의한 파라 미터와 구성에 따라 달라집니다.
```python
import requests

url = "https://api.openai.com/v1/assistants"

headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json",
    "OpenAI-Beta": "assistants=v2"
}

data = {
    "name": "Your Assistant Name",
    "description": "A brief description of what this assistant is for",
    "model": "gpt-3.5-turbo"
}

response = requests.post(url, headers=headers, json=data)

print(response.text)
```

>[!Assiatants 실습 결과]
>```json
>{
> "id": "asst_vnNwlG5P9cJcaoQiwdDGo3zN",
> "object": "assistant",
> "created_at": 1737350515,
> "name": "Your Assistant Name",
> "description": "A brief description of what this assistant is for",
> "model": "gpt-3.5-turbo",
> "instructions": null,
> "tools": [],
> "top_p": 1.0,
> "temperature": 1.0,
> "tool_resources": {},
> "metadata": {},
> "response_format": "auto"
>}
>```

---
#### Create chat completion
- post 요청 URL https://api.openai.com/v1/chat/completions 
- client.chat.completions.create () : 채팅 세션을 시작 
- messages: 대화 내역을 나타내는 메시지 리스트
	- 각 메시지는 메시지를 보내는 주체를 나타내는 role과 메시지의 내용 content 필드를 가 집니다
```python
from openai import OpenAI

OPENAI_API_KEY = "YOUR_API_KEY"
client = OpenAI(api_key=OPENAI_API_KEY)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "developer", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(completion.choices[0].message)
```

>[!Create chat completion의 response]
>```json
>{
>  "id": "chatcmpl-123", 
>  "object": "chat.completion", 
>  "created": 1677652288, 
>  "model": "gpt-4o-mini", 
>  "system_fingerprint": "fp_44709d6fcb", 
>  "choices": [
>    {
>      "index": 0, 
>      "message": {
>        "role": "assistant", 
>        "content": "\n\nHello there, how may I assist you today?"
>      }, 
>      "logprobs": null, 
>      "finish_reason": "stop"
>    }
>  ], 
>  "service_tier": "default", 
>  "usage": {
>    "prompt_tokens": 9, 
>    "completion_tokens": 12, 
>    "total_tokens": 21, 
>    "completion_tokens_details": {
>      "reasoning_tokens": 0, 
>      "accepted_prediction_tokens": 0, 
>      "rejected_prediction_tokens": 0
>    }
>  }
>}
>```

|**필드**|**설명**|
|---|---|
|**id**|응답의 고유 식별자|
|**object**|응답 객체의 유형 (`chat.completion`)|
|**created**|객체 생성 시간 (유닉스 타임스탬프)|
|**model**|응답 생성에 사용된 모델의 이름|
|**system_fingerprint**|시스템의 특정 상태나 설정을 나타내는 지문|
|**choices**|생성된 응답들 배열|
|└ **index**|선택된 응답의 인덱스|
|└ **message**|생성된 응답 내용|
|└ └ **role**|응답을 보낸 역할 (e.g., `assistant`)|
|└ └ **content**|응답 메시지|
|└ **logprobs**|로그 확률 정보 (null로 설정됨)|
|└ **finish_reason**|응답 생성 종료 이유 (`stop`: 모델이 응답을 종료함)|
|**service_tier**|사용된 서비스 등급 (e.g., `default`)|
|**usage**|요청 및 응답 생성에 사용된 토큰 수 정보|
|└ **prompt_tokens**|프롬프트에 사용된 토큰 수|
|└ **completion_tokens**|응답에 사용된 토큰 수|
|└ **total_tokens**|전체 토큰 수 (프롬프트 + 응답)|
|└ **completion_tokens_details**|추가 토큰 세부사항|
|└ └ **reasoning_tokens**|추론에 사용된 토큰 수|
|└ └ **accepted_prediction_tokens**|수용된 예측 토큰 수|
|└ └ **rejected_prediction_tokens**|거부된 예측 토큰 수|

```python
from openai import OpenAI
OPENAI_API_KEY = "YOUR_API_KEY"
client = OpenAI(api_key=OPENAI_API_KEY)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user",
	    "content": [{"type": "text", "text": "What's in this image?"},
            {"type": "image_url","image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",},},],}],
    max_tokens=300,)

print(response.choices[0])
```

>[!출력결과]
>```JSON
>Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='The image shows a wooden boardwalk leading through a green field or wetland area. The sky is blue with some clouds, and there are trees and bushes in the background. It looks like a peaceful natural landscape.', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None))
>```

---
#### 스트리밍 방식으로 대화형 AI 모델의 응답 처리
- 스트리밍 옵션 모델의 응답을 실시간으로 여러 부분으로 나뉘어서 받을 수 있으며,각 부분이 준비되는 즉시 처리할 수 있습니다.
- 특히 긴 대화나 실시간 인터랙션이 필요한 애플리케이션에 유용합니다.
```python
from openai import OpenAI

OPENAI_API_KEY = "YOUR_API_KEY"
client = OpenAI(api_key=OPENAI_API_KEY)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "developer", "content": "You are a helpful assistant."},
        {"role": "user", "content": "HELLO!"}
    ],
    stream=True  # 응답이 스트리밍 방식으로 전달
)

# 응답이 준비되는 대로 바로 받아볼 수 있습니다.
# completion은 반복 가능한 객체로, 생성된 각 응답 조각(chunk)을 순회
for chunk in completion:
    print(chunk.choices[0].delta)  # delta는 모델이 생성한 변경 사항(응답의 일부분)을 포함합니다.
```

>[!출력결과]
>```JSON
>ChoiceDelta(content='', function_call=None, refusal=None, role='assistant', tool_calls=None)
>ChoiceDelta(content='Hello', function_call=None, refusal=None, role=None, tool_calls=None)
>ChoiceDelta(content='!', function_call=None, refusal=None, role=None, tool_calls=None)
>ChoiceDelta(content=' How', function_call=None, refusal=None, role=None, tool_calls=None)
>ChoiceDelta(content=' can', function_call=None, refusal=None, role=None, tool_calls=None)
>ChoiceDelta(content=' I', function_call=None, refusal=None, role=None, tool_calls=None)
>ChoiceDelta(content=' assist', function_call=None, refusal=None, role=None, tool_calls=None)
>ChoiceDelta(content=' you', function_call=None, refusal=None, role=None, tool_calls=None)
>ChoiceDelta(content=' today', function_call=None, refusal=None, role=None, tool_calls=None)
>ChoiceDelta(content='?', function_call=None, refusal=None, role=None, tool_calls=None)
>ChoiceDelta(content=None, function_call=None, refusal=None, role=None, tool_calls=None)
>```

---
#### 특정 도구(tools)를 활용하는 채팅 세션 생성
```PYTHON
from openai import OpenAI

OPENAI_API_KEY = "YOUR_API_KEY"
client = OpenAI(api_key=OPENAI_API_KEY)

tools = [{"type": "function",
    "function": {
    "name": "get_current_weather",        
    "description": "Get the current weather in a given location",
    "parameters": {"type": "object",
        "properties": {"location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"},
            "unit": {"type": "string",
                "enum": ["celsius", "fahrenheit"]}},
            "required": ["location"]}}}]

messages = [
    {"role": "user", "content": "What's the weather like in Boston today?"}
]

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

print(completion)
```

>[!출력결과]
>```JSON
>ChatCompletion(id='chatcmpl-ArfhMUARdUH3iaivxerHbcS2G0FON', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content=None, refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_SmIyiHCcTn6LOrk1ob424YcV', function=Function(arguments='{"location":"Boston, MA"}', name='get_current_weather'), type='function')]))], created=1737355036, model='gpt-4o-2024-08-06', object='chat.completion', service_tier='default', system_fingerprint='fp_4691090a87', usage=CompletionUsage(completion_tokens=18, prompt_tokens=80, total_tokens=98, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
>```

---
#### 프로프트 엔지니어링
:모델로부터 올바른 출력을 얻기 위해 프롬프트를 제작하는 과정

>모델에 정확한 지시사항, 예시 및 필요한 맥락 정보(모델의 훈련 데이터에 포함되지 않은 개인적이거나 전문적인 정보 등)를 제공함으로써 모델의 출력 품질과 정확성을 향상시킬 수 있습니다.

- Completion.create() 매개변수

|매개변수|설명|기본값|사용 사례 및 옵션|
|---|---|---|---|
|**`model`**|사용할 모델 이름 (예: `text-davinci-003`, `gpt-3.5-turbo`)|없음|생성할 텍스트의 모델을 지정|
|**`prompt`**|프롬프트 텍스트 생성의 시작점|없음|텍스트 생성의 기준이 되는 입력 문장|
|**`temperature`**|출력 텍스트의 창의성 조정 (0~2)|`1`|창의성 조정, 값이 낮을수록 예측 가능하고, 높을수록 창의적|
|**`max_tokens`**|생성할 텍스트의 최대 길이 (토큰 수)|없음|최대 토큰 수를 설정, 응답 길이를 제한하는 데 사용|
|**`frequency_penalty`**|이미 생성된 단어의 반복 억제 (범위: -2.0 ~ 2.0)|`0`|단어 반복을 억제하거나 허용하는 정도|
|**`presence_penalty`**|새로운 단어 또는 아이디어 도입을 증가시키는 값 (범위: -2.0 ~ 2.0)|`0`|새로운 단어나 아이디어의 도입 정도|
|**`top_p`**|확률 분포 상위 p%에 해당하는 토큰만 사용 (0~1)|없음|모델이 선택할 수 있는 단어를 제한하는 방식|
|**`n`**|생성할 결과 수|`1`|여러 개의 결과를 생성하고 그 중 하나를 선택|
|**`stream`**|응답을 스트리밍 방식으로 받을지 여부|`false`|`true`로 설정하면 응답을 실시간으로 받아볼 수 있음|
|**`logprobs`**|확률 분포 상위 p%에 해당하는 토큰만 사용|없음|확률이 높은 토큰들만 선택하여 생성된 응답에 대한 확률 정보 제공|
|**`echo`**|결과와 함께 프롬프트를 에코백|`false`|`true`로 설정하면 프롬프트도 함께 출력|
|**`stop`**|텍스트 생성 중지 토큰, 이 토큰이 나오면 생성이 중단됩니다. 최대 4개까지 설정 가능|없음|문장 생성 중지 조건 설정, 예: `["\n", "stop"]`|
|**`best_of`**|서버 측에서 `best_of` 개만큼 결과를 생성하고 가장 좋은 결과를 반환|`1`|더 나은 결과를 원할 경우 `best_of` 값을 증가시켜 더 많은 결과 생성|
|**`logit_bias`**|지정한 토큰의 생성 가능성을 감소시키는 값 (토큰 ID에 대해 설정)|없음|특정 토큰을 더 적게 생성하게 하여 필터링|
|**`user`**|최종 사용자 ID, 이 값을 설정하면 특정 사용자와 관련된 데이터를 추적할 수 있음|없음|사용자 맞춤형 응답을 제공하기 위해 사용자의 ID를 설정|
##### 질의 응답 실습
```python
from openai import OpenAI

OPENAI_API_KEY = "YOUR_API_KEY"
client = OpenAI(api_key=OPENAI_API_KEY)

prompt = "인공지능에 대해 알려주세요"

response = client.completions.create(
    model="gpt-3.5-turbo-instruct", 
    prompt=prompt, 
    temperature=0, 
    max_tokens=50
)

#prompt 문자열의 끝에 모델에게 텍스트 요약 작업을 명확하게 지시
response_text = response.choices[0].text
print("응답 텍스트:", response_text.strip())
```
##### 요약 실습
```PYTHON
from openai import OpenAI

OPENAI_API_KEY = "YOUR_API_KEY"
client = OpenAI(api_key=OPENAI_API_KEY)

prompt = """인공지능(Artificial Intelligence, AI)은 인간의 지능을 모방하거나 대체하는 기술을 말합니다. 인공지능은 컴퓨터 프로그램이나 시스템을 통해 인간의 학습, 추론, 문제 해결 등의 지능적인 작업을 수행할 수 있도록 만들어진 기술입니다.인공지능은 크게 강한 인공지능과 약한 인공지능으로 나뉩니다. 강한 인공지능은 인간과 동일하거나 그 이상의 지능을 가지며, 모든 종류의 작업을 수행할 수 있습니다. 반면 약한 인공지능은 특정한 작업에만 특화된 지능을 가지며, 인간의 지능과는 차이가 있습니다. 인공지능은 다양한 분야에서 활용되고 있습니다. 예를 들어 음성 인식 기술을 이용한 인공지능 스피커, 이미지 인식 기술을 이용한 얼굴 인식 소프트웨어, 자율주행 자동차 등이 있습니다. 또한 인공지능은 의료, 금융, 교육 등 다양한 분야에서도 활용되고 있으며, 더 나은 서비스를 제공하기 위해 계속 발전하고 있습니다. 하지만 인공지능은 아직 완벽하지 않으며, 인간의 지능을 완전히 대체할 수는 없습니다. 따라서 인공지능을 개발하고 활용하는 과정에서 윤리적인 문제나 안전 문제 등을 고려해야 합니다. \n\n Summarize the above text. """

response = client.completions.create(
    model="gpt-3.5-turbo-instruct", 
    prompt=prompt, 
    temperature=0, 
    max_tokens=500
) 

# prompt 문자열의 끝에 모델에게 텍스트 요약 작업을 명확하게 지시
response_text = response.choices[0].text
print("요약 텍스트:", response_text.strip())
```
##### 번역 실습
```PYTHON
from openai import OpenAI

OPENAI_API_KEY = "YOUR_API_KEY"
client = OpenAI(api_key=OPENAI_API_KEY)

prompt = """ 한국어를 영어로 번역합니다. 한국어 : 대한민국의 민주주의는 국민이 지킵니다. 영어 : """

response = client.completions.create(
    model="gpt-3.5-turbo-instruct", 
    prompt=prompt, 
    temperature=0, 
)

response_text = response.choices[0].text
print("번역 텍스트:", response_text.strip())
```