
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
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    },
                },
            ],
        }
    ],
    max_tokens=300,
)

print(response.choices[0])
```

>[!출력결과]
>```
>Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='The image shows a wooden boardwalk leading through a green field or wetland area. The sky is blue with some clouds, and there are trees and bushes in the background. It looks like a peaceful natural landscape.', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None))
>```

