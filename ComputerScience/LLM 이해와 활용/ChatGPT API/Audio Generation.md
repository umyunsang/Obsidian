
---
### Audio Generation with GPT-4o-audio-preview: Overview

GPT-4o-audio-preview 모델은 텍스트 및 오디오 입력을 활용하여 다양한 출력 형식을 생성할 수 있습니다. 주요 기능 및 특징은 다음과 같습니다.

---
#### 1. 주요 기능

- **텍스트 본문 요약을 음성 오디오로 생성**
    
    - 입력: 텍스트
    - 출력: 요약된 텍스트 및 음성 오디오
- **녹음에서 감정 분석 수행**
    
    - 입력: 오디오
    - 출력: 텍스트 및 감정 분석 결과
- **비동기 음성 대 음성 상호작용**
    
    - 입력: 오디오
    - 출력: 음성 오디오

---
#### 2. 입출력 조합

| **입력 형식**    | **출력 형식**    |
| ------------ | ------------ |
| 텍스트 입력       | 텍스트 + 오디오 출력 |
| 오디오 입력       | 텍스트 + 오디오 출력 |
| 오디오 입력       | 텍스트 출력       |
| 텍스트 + 오디오 입력 | 텍스트 + 오디오 출력 |
| 텍스트 + 오디오 입력 | 텍스트 출력       |

---
#### 3. 사용 방법

- **REST API**: HTTP 클라이언트를 통해 OpenAI Audio API 호출
- **OpenAI 공식 SDK**: Python 및 기타 언어에서 SDK를 사용해 간편한 API 호출

---
#### 4. 주요 라이브러리

- **`base64` 모듈**
    - 바이너리 데이터와 ASCII 문자열 간의 인코딩 및 디코딩 처리
    - 오디오 데이터를 Base64 형식으로 변환하여 API로 전송

---
#### 5. 주요 사례

1. **텍스트 → 텍스트 + 오디오 출력**  
    텍스트 질문에 대해 GPT-4o-audio-preview가 텍스트와 오디오 형식으로 응답 생성.
    
2. **오디오 → 텍스트 출력**  
    녹음 데이터를 텍스트로 변환 및 감정 분석 수행.
    
3. **오디오 → 텍스트 + 오디오 출력**  
    음성 입력을 텍스트로 변환하며, 추가적으로 음성으로도 응답.
    
---
#### **참고 링크**

- OpenAI Audio API 가이드:  
    [https://platform.openai.com/docs/guides/audio](https://platform.openai.com/docs/guides/audio)

---
#### url 오디오 파일 예제 코드
```python
import base64
import requests
from openai import OpenAI

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

# 오디오 파일 가져오기 및 Base64 인코딩
url = "https://openaiassets.blob.core.windows.net/$web/API/docs/audio/alloy.wav"
response = requests.get(url)
response.raise_for_status()  # 요청 상태 확인
wav_data = response.content  # 바이너리 형태의 오디오 데이터

# 오디오 데이터를 Base64 문자열로 인코딩
encoded_string = base64.b64encode(wav_data).decode('utf-8')

# OpenAI API 호출
completion = client.chat.completions.create(
    model="gpt-4o-audio-preview",
    modalities=["text", "audio"],
    audio={"voice": "alloy", "format": "wav"},
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this recording?"},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": encoded_string,
                        "format": "wav"
                    }
                }
            ]
        },
    ]
)

# 응답 메시지 출력
print(completion.choices[0].message)
```

#### 한국어 음성 입력 > transcript한 후 응답(음성과 텍스트)
```python
import base64
import requests
from openai import OpenAI

client = OpenAI(api_key= OPENAI_API_KEY)
# 로컬 디렉토리에서 wav 파일 읽기
file_path = "korean.mp3"   
with open(file_path, "rb") as wav_file:
    wav_data = wav_file.read()

# base64로 인코딩
encoded_string = base64.b64encode(wav_data).decode("utf-8")

completion = client.chat.completions.create(
    model="gpt-4o-audio-preview",
    modalities=["text", "audio"],  #모델이 사용할 입력/출력 형식을 정의
    audio={"voice": "alloy", "format": "mp3"},
    messages=[
        {
            "role": "user",
            "content": [
                { 
                    "type": "text",
                    "text": "음성 녹음 내용이 무엇인가요?"
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": encoded_string,
                        "format": "mp3"
                    }
                }
            ]
        },
    ]
)

print(completion.choices[0].message)
```