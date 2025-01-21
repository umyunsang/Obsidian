
---
Moderation  텍스트나 이미지가 잠재적으로 해로운지 여부를 확인하는 데 사용할 수 있는 도구 콘텐츠를 필터링하거나 문제를 일으키는 사용자 계정에 개입하는 등의 시정 조치를 취할 수 있습니다. Moderation 엔드포인트는 무료로 사용할 수 있습니다. omni-moderation-latest 모델 : 모든 스냅샷은 더 많은 분류 옵션과 멀티 모달 입력을 지원 합니다. text-moderation-latest (Legacy) 모델 : 오직 텍스트 입력만 지원하고 입력 분류가 더 적은 구형 모델입니다. https://platform.openai.com/docs/guides/moderation
#### Moderation 실습
```python
from openai import OpenAI

# OpenAI 클라이언트 초기화
client = OpenAI(api_key= OPENAI_API_KEY)

# 콘텐츠 검토 요청
response = client.moderations.create(
    model="omni-moderation-latest",
    input="...text to classify goes here...",
)

# 응답 출력
print(response)
```
