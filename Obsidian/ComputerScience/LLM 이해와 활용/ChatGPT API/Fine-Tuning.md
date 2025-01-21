
---
### Fine-Tuning이란

Fine-Tuning은 사전 학습된 LLM(Large Language Model)을 특정 작업이나 요구 사항에 맞게 추가 학습시켜 최적화하는 **전이 학습(Transfer Learning)** 방법입니다.

- 적은 데이터로도 특정 작업에서 높은 성능 발휘 가능.
- 사용자 요구에 맞는 스타일, 톤, 출력 형식을 제공.

---

### Fine-Tuning의 주요 특징

1. **정의**
    - LLM을 특정 작업에 적합하도록 추가 학습.
    - 주로 **질문-답변 형태**의 고품질 데이터셋으로 학습 진행.
2. **장점**
    - 적은 프롬프트 예제로도 정확한 결과 생성.
    - 요청 지연 시간 감소 및 비용 절감.
3. **OpenAI 권장 사항**
    - Fine-Tuning 적용 전 **프롬프트 엔지니어링** 또는 **프롬프트 체이닝**을 시도.

---

### Fine-Tuning의 단계

1. **데이터 준비**
    - 고품질 데이터 수집 및 전처리(레이블링, 클리닝, 토큰화).
    - JSON Lines(`.jsonl`) 형식으로 작성:
        
        ```json
        {"prompt": "입력 프롬프트", "completion": "출력 값"}
        ```
        
    - 최소 10개의 예제 필요, 권장 50~100개 제공.
2. **모델 준비**
    - 사전 학습된 모델 로드(GPT, BERT 등).
    - 필요한 경우 모델 구조 조정.
3. **하이퍼파라미터 설정**
    - 학습률, 배치 크기, 에폭 수 설정.
4. **학습 진행**
    - 기존 가중치를 유지하며 추가 학습(Fine-Tuning).
5. **평가 및 테스트**
    - 검증 데이터로 성능 평가, 오버피팅 방지(Early Stopping).
6. **모델 사용**
    - Fine-Tuned 모델 배포 및 실제 사용.

---

### Fine-Tuning 기법 비교

|기법|설명|특징|
|---|---|---|
|**Full Fine-Tuning**|모델 전체 가중치 업데이트|대규모 데이터, 높은 컴퓨팅 자원 필요.|
|**Partial Fine-Tuning**|모델 일부 계층만 학습|하위 계층 고정, 상위 계층 업데이트.|
|**Prompt Tuning**|입력 프롬프트를 조정해 출력 변화|대규모 모델에 적합, 효율적인 방식.|
|**Adapter Tuning**|Adapter 계층만 학습, 기존 가중치는 유지|추가 메모리 소모가 적음.|
|**LoRA**|저차원 매개변수 학습 기법|파라미터 효율적 업데이트.|

---

### Fine-Tuning 가능한 모델

- `gpt-4o-2024-08-06`
- `gpt-4o-mini-2024-07-18`
- `gpt-4-0613`
- `gpt-3.5-turbo-0125`
- `gpt-3.5-turbo-1106`
- `gpt-3.5-turbo-0613`

---

### Fine-Tuning 적용 사례

1. 스타일, 톤, 형식 설정.
2. 출력 신뢰성 향상.
3. 프롬프트 오류 수정.
4. 복잡한 엣지 케이스 처리.
5. 새로운 작업 또는 기술 적용.

---

### Fine-Tuning의 한계

- 고품질의 레이블링된 데이터가 필요 
- 높은 학습 비용과 시간이 필요 
- 소량의 데이터로 파인튜닝 시 모델이 특정 데이터에 과적합될 수 있음 
- 모델의 범용성 저하

---
### Fine-Tuning API 사용법

#### FineTune 클래스

Fine-Tuning API와 상호작용하며 학습 작업 생성, 상태 모니터링, 결과 확인을 쉽게 수행할 수 있는 클래스.

|메서드|설명|
|---|---|
|`FineTune.create()`|Fine-Tuning 작업 생성.|
|`list()`|모든 Fine-Tuning 작업 나열.|
|`retrieve(file_id)`|특정 Fine-Tuning 작업 세부 정보 조회.|
|`delete(file_id)`|진행 중인 Fine-Tuning 작업 취소.|
|`download(file_id)`|서버에서 파일 다운로드.|
|`events()`|작업 이벤트 로그 확인(상태, 오류 등).|

#### `FineTune.create()` 주요 파라미터

- `training_file`: 학습 데이터 파일 ID.
- `validation_file`: 검증 데이터 파일 ID(선택 사항).
- `model`: 기본 모델 이름(e.g., `gpt-3.5-turbo`).
- `n_epochs`: 학습 반복 횟수(기본값: 4).

---

### **추천 참고 링크**

1. [Preparing Your Dataset](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset)
2. [Fine-Tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
