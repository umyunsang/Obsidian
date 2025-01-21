
---
#### Text to speech
TTS 모델을 활용하여 텍스트 기반 콘텐츠를 음성으로 변환  https://platform.openai.com/docs/guides/text-to-speech  다양한 언어로 제공하며, 스트리밍 기능을 통해 즉각적인 오디오 피드백을 사용자에게 제 공  6개의 내장된 목소리를 제공 (alloy, ash, coral, echo, fable, onyx, nova, sage and shimmer) 실시간 오디오 스트리밍을 지원합니다. 지원되는 출력 형식 기본 응답 형식은 "mp3"이지만, "opus", "aac", "flac", "pcm"과 같은 다 른 형식도 사용할 수 있습니다. 지원되는 언어 TTS 모델은 일반적으로 Whisper 모델의 언어 지원을 따릅니다.

#### Text to speech 실습
```python
from openai import OpenAI

client = OpenAI(api_key= OPENAI_API_KEY)

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="2025년은 푸른 뱀의 해입니다.",
)

response.stream_to_file("output.mp3")
```
#### Speech to text
'large-v2 Whisper 모델'을 기반으로 'transcriptions’(오디오 파일의 내용을 듣고 그 내용을 텍스트로 변환)과 'translations’ ( 오디오를 듣고 그 내용을 다른 언어로 변환) 제공  파일 업로드는 현재 최대 25MB로 제한되어 있으며, 지원하는 입력 파일 유형에는 mp3, mp4, mpeg, mpga, m4a, wav, webm이 포함됩니다. 기본적으로 Whisper API는 제공된 오디오를 텍스트로 transcript 합니다  비디오 편집에 단어 수준의 정밀도를 가능하게 하며, 개별 단어에 연결된 특정 프레임의 제거를 허용합니다. timestamp_granularities[] 파라미터를 사용하면 더 구조화되고 타임스탬프가 찍힌 JSON 출 력 형식을 활성화할 수 있습니다 Whisper API는 25MB 이하의 파일만 지원합니다. 만약 25MB보다 큰 오디오 파일이 있다면, 파일을 25MB 이하로 나누거나 압축된 오디오 형식을 사용해야 합니다  PyDub이라는 오픈 소스 파이썬 패키지를 사용하여 오디오를 분할 https://platform.openai.com/docs/guides/speech-to-text

#### Speech to text Ranscriptions 실습