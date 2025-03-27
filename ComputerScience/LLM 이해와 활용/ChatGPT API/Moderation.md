
---
### Moderation 요약 정리

Moderation은 텍스트나 이미지가 잠재적으로 해로운지 여부를 판별하여 적절한 조치를 취하는 데 사용하는 도구입니다.  
이 도구는 콘텐츠 필터링, 문제 사용자 계정 관리 등에 활용할 수 있습니다.

---

### 주요 특징

1. 텍스트 및 이미지 콘텐츠 검토 지원.
2. 콘텐츠 필터링과 문제 발생 방지를 위한 시정 조치 수행.
3. **Moderation 엔드포인트**는 무료로 제공.

---

### 모델 종류

1. **omni-moderation-latest**
    
    - 최신 모델로, 더 많은 분류 옵션 제공.
    - 멀티 모달 입력(텍스트와 이미지)을 지원.
2. **text-moderation-latest (Legacy)**
    
    - 구형 모델로, 텍스트 입력만 지원.
    - 분류 옵션이 제한적.

---

### Moderation 엔드포인트 활용 실습

```python
from openai import OpenAI

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

# 콘텐츠 검토 요청
response = client.moderations.create(
    model="omni-moderation-latest",  # 최신 Moderation 모델 사용
    input="...text to classify goes here...",  # 검토할 텍스트 입력
)

# 응답 출력
print(response)
```

---

### 참고 링크

- Moderation 가이드: [OpenAI Moderation Documentation](https://platform.openai.com/docs/guides/moderation)