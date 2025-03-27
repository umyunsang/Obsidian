
---
### Text to Speech (TTS) 및 Speech to Text (STT) 기능 요약

---

### 1. Text to Speech (TTS)

**TTS 모델**은 텍스트를 음성으로 변환하며, 다양한 언어 및 음성 스타일을 제공합니다.  
즉각적인 오디오 피드백과 실시간 스트리밍 기능도 지원합니다.

#### 주요 특징

- **지원 음성**: `alloy`, `ash`, `coral`, `echo`, `fable`, `onyx`, `nova`, `sage`, `shimmer`.
- **지원 출력 형식**: 기본은 `mp3`, 추가적으로 `opus`, `aac`, `flac`, `pcm` 가능.
- **지원 언어**: Whisper 모델과 동일한 언어 지원.

#### TTS 실습 코드

```python
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="2025년은 푸른 뱀의 해입니다."
)

# 음성 파일 저장
response.stream_to_file("output.mp3")
```

---

### 2. Speech to Text (STT

**STT 모델**은 Whisper를 기반으로 하며, 오디오를 텍스트로 변환(Transcription)하거나 번역(Translation)합니다.

#### 주요 특징

- **지원 모델**: `large-v2 Whisper`.
- **파일 크기 제한**: 최대 25MB.
- **지원 입력 파일 형식**: `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `wav`, `webm`.
- **고급 기능**:
    - 타임스탬프를 포함한 JSON 구조화된 출력 가능.
    - PyDub 라이브러리를 통해 큰 파일을 분할 처리.

---

### 2.1. Transcriptions (오디오 -> 텍스트 변환

#### 실습 코드

```python
from openai import OpenAI

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

# 오디오 파일 열기
audio_file = open("output.mp3", "rb")

# Whisper 모델로 텍스트 변환
transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="text"
)

# 변환된 텍스트 출력
print(transcription)
```

---

### 2.2. Translations (오디오 -> 번역 텍스트 변환

#### 실습 코드

```python
from openai import OpenAI

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

# 오디오 파일 열기
audio_file = open("output.mp3", "rb")

# Whisper 모델로 번역 수행
translation = client.audio.translations.create(
    model="whisper-1",
    file=audio_file,
)

# 번역된 텍스트 출력
print(translation.text)
```

---

### 활용 시나리오

1. **TTS 활용**:
    
    - 콘텐츠 음성화(예: 뉴스 읽기, 챗봇 음성 응답).
    - 스트리밍 기반의 실시간 피드백 서비스.
2. **STT 활용**:
    
    - 음성 기록 텍스트화(회의 기록, 강의 녹음).
    - 멀티언어 오디오 번역(국제 컨퍼런스).

**참고**: [OpenAI TTS/STT Documentation](https://platform.openai.com/docs/guides/text-to-speech)