
---
### Embedding: 개요 및 실습

---

#### 1. Embedding이란?

- **정의**: 텍스트 데이터를 고차원 벡터(부동소수점 배열)로 변환하는 기술.
- **특징**:
    - 유사한 의미를 가진 단어나 문장은 벡터 거리가 가깝게 설계.
    - 유사하지 않은 의미는 벡터 간 거리가 멀어지도록 구성.
- **활용 사례**:
    - **데이터 분석**: 클러스터링, 추천 시스템, 검색 최적화.
    - **AI 응용**: 유사도 비교, 주제 분석, 군집화.
- **참고 자료**:  
    [Embedding 공식 문서](https://platform.openai.com/docs/guides/embeddings)

---

#### 2. Embedding 클래스 및 주요 함수

**A. Embedding 클래스**

- **역할**: 입력 텍스트에 대해 고차원 벡터 표현 생성.
- **응용 사례**:
    - 텍스트를 의미론적으로 유사한 텍스트끼리 가까운 위치로 매핑.
    - 텍스트 의미와 구조를 벡터로 변환해 컨텍스트 정보 캡처.

**B. Embedding.create() 함수**

- **설명**: 주어진 입력 텍스트에 대한 임베딩 생성.
- **매개변수**:
    - `model` (필수): 사용할 모델 이름 (예: `"text-embedding-3-small"`).
    - `input` (필수): 임베딩 생성 대상 (텍스트 또는 텍스트 리스트).
    - `user` (선택적): 사용자 정의 데이터 제공 시 사용.
    - `**kwargs` (선택적): 추가 설정이나 API 기능 구성.

---

#### 3. Embedding 기반 유사도 검색 실습

**목적**: Facebook의 오픈소스 벡터 데이터베이스 **Faiss**를 활용하여 유사도 검색 수행.

```python
import openai
import numpy as np
import faiss

# OpenAI API Key 설정
OPENAI_API_KEY = "sk-proj-hbsi47ndwg_2I1zuYAa0p9ODzEM 4vApXF27cJkcA............."
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 입력 텍스트에 대한 임베딩 생성
in_text = "오늘은 눈이 오지 않아서 다행입니다"
response = client.embeddings.create(
    input=in_text,
    model="text-embedding-3-sm"
)

# 응답에서 임베딩 추출 및 변환
in_embeds = [record.embedding for record in response.data]
ndarray_embeds = np.array(in_embeds).astype("float32")  # 효율적 계산을 위해 numpy 배열 변환

# 비교 대상 텍스트 리스트
target_texts = [
    "좋아하는 음식은 무엇인가요?", 
    "어디에 살고 계신가요?", 
    "아침 전철은 혼잡하네요", 
    "오늘 날씨가 화창합니다", 
    "요즘 경기가 좋지 않습니다."
]

# 비교 대상 텍스트에 대한 임베딩 생성
response2 = client.embeddings.create(
    input=target_texts,
    model="text-embedding-3-small"
)
target_embeds = [record.embedding for record in response2.data]
ndarray_embeds2 = np.array(target_embeds).astype("float32")

# Faiss 인덱스 생성 및 벡터 추가
index = faiss.IndexFlatL2(1536)  # 벡터 차원 지정 (예: 1536차원 모델)
index.add(ndarray_embeds2)  # 벡터 추가 (ndarray는 float32 형식이어야 함)

# 유사도 검색 수행
D, I = index.search(ndarray_embeds, 1)  # 가장 가까운 벡터 1개 검색

# 결과 출력
print(D)  # D: 쿼리 벡터와 가장 가까운 이웃 벡터 간 거리
print(I)  # I: 가장 가까운 이웃 벡터의 인덱스
print(target_texts[I[0][0]])  # 가장 가까운 이웃 텍스트 출력
```

---

#### 4. 주요 실행 결과

- **`D` (거리)**: 입력 벡터와 가장 가까운 이웃 벡터 간의 거리 정보.
- **`I` (인덱스)**: 입력 벡터와 가장 가까운 이웃 텍스트의 인덱스.
- **출력 텍스트**: 가장 유사한 의미를 가진 텍스트를 반환.

---

**활용**: 추천 시스템, 유사 텍스트 검색, 주제 기반 군집화 등 다양한 머신러닝 및 NLP 응용.  
**참고**: `faiss`는 대규모 벡터 데이터를 처리할 때 효율적이며, GPU 가속도 가능.