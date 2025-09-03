
---
# Architecture
## **1. 시스템 개요**  

카페나 상점에서 고객의 **구매 기록**과 **AI 분석**을 활용하여 맞춤형 추천을 제공하고, **결제까지 한 번에 진행**할 수 있도록 하는 시스템을 설계합니다. 또한, **판매 관리자가 판매하는 메뉴 정보를 직접 입력**하면, 해당 데이터가 **벡터 저장소(Vector Database)** 에 저장되고, **RAG(Retrieval-Augmented Generation) 방식**을 활용하여 AI 챗봇이 더욱 정밀한 응답을 생성하도록 합니다.

### **🎯 주요 목표**  
- 고객의 취향과 구매 패턴을 분석하여 **개인 맞춤형 메뉴 추천**  
- 자연어 처리(NLP) 기반의 **대화형 AI 챗봇**을 활용한 주문 인터페이스  
- **결제 기능 통합**으로 원스톱 구매 프로세스 구축  
- LLM(RAG 또는 Fine-Tuning)을 활용한 고객 응대 및 추천 개선  
- **판매 관리자가 메뉴 데이터를 입력하여 AI 응답을 최적화**  
- 접근성 향상(디지털 소외계층 고려)  

---

## **2. 시스템 구성 요소**  

### **🛠 기술 스택**  
| 구성 요소 | 기술 선택 |
|-----------|------------|
| 백엔드 API | Python (Flask / FastAPI) |
| 데이터베이스 | PostgreSQL / Firebase Firestore |
| AI 모델 | LLM (GPT-4, Llama2, Mistral) + RAG or Fine-Tuning |
| 추천 시스템 | 협업 필터링, 콘텐츠 기반 필터링 (Scikit-learn, TensorFlow) |
| 음성 인터페이스 | OpenAI Whisper, Google Speech-to-Text |
| 프론트엔드 | React (Next.js) |
| 결제 시스템 | Stripe, Toss Payments, Kakao Pay |
| 벡터 저장소 | ChromaDB, Weaviate, Pinecone |

---

## **3. 시스템 아키텍처**  

<div class="mermaid-container" style="transform: scale(0.6); transform-origin: top left;">

```mermaid
sequenceDiagram
    판매관리자->>벡터저장소: 메뉴 데이터 입력
    벡터저장소-->>AI챗봇: 데이터 활용
    사용자->>AI챗봇: 자연어 입력
    AI챗봇-->>사용자: 메뉴 추천
    사용자->>결제시스템: 주문 요청
    결제시스템-->>사용자: 결제 진행
    사용자->>데이터저장: 주문 완료
```
</div>

1️⃣ **판매 관리자가 메뉴 정보를 입력하면 벡터 저장소에 저장**  
2️⃣ **고객이 AI 챗봇을 통해 음성/텍스트 입력으로 메뉴를 검색**  
3️⃣ **AI가 벡터 저장소의 정보를 기반으로 최적의 추천을 제공**  
4️⃣ **고객이 메뉴를 선택하고 주문을 진행**  
5️⃣ **결제 모듈과 연동하여 결제 진행 (QR 결제, 카드, 간편결제 지원)**  
6️⃣ **주문 완료 후 고객 피드백 수집 및 데이터 저장**  

---

## **4. AI 모델 활용 방안**  

### **🔹 LLM을 활용한 자연어 기반 추천**  
- 고객의 **자연어 입력 분석** (예: "오늘 단 게 땡겨", "커피 말고 다른 음료 있어?")  
- **카페 메뉴 정보**를 벡터 저장소에 저장하고, RAG 또는 Fine-Tuning하여 LLM이 자연스럽게 추천  

### **🔹 추천 시스템 (AI 기반 추천)**  
1️⃣ **협업 필터링 (CF, Collaborative Filtering)**  
   - 비슷한 고객의 구매 패턴을 분석하여 추천  
   - "A 고객이 선호하는 메뉴를 B 고객도 선호할 가능성이 높음"  

2️⃣ **콘텐츠 기반 필터링 (CBF, Content-Based Filtering)**  
   - 고객의 과거 구매한 메뉴와 유사한 특징을 가진 메뉴 추천  
   - "고객이 바닐라 라떼를 자주 주문했다면, 헤이즐넛 라떼 추천"  

3️⃣ **하이브리드 모델 (CF + CBF + LLM)**  
   - LLM이 자연어 입력을 해석하고, 추천 알고리즘과 결합하여 정밀한 추천 제공  

---

## **5. 데이터베이스 및 벡터 저장소 설계 (예시)**  

### **🔹 주요 테이블**  

#### **① 고객 테이블 (`customers`)**  
| Column | Type | Description |
|--------|------|-------------|
| customer_id | INT | 고객 고유 ID |
| name | VARCHAR | 고객 이름 |
| phone | VARCHAR | 전화번호 |
| preferences | JSON | 선호하는 메뉴 및 태그 |

#### **② 주문 테이블 (`orders`)**  
| Column | Type | Description |
|--------|------|-------------|
| order_id | INT | 주문 고유 ID |
| customer_id | INT | 고객 ID (FK) |
| menu_id | INT | 주문한 메뉴 ID |
| total_price | DECIMAL | 총 결제 금액 |
| order_date | TIMESTAMP | 주문 시간 |

#### **③ 메뉴 테이블 (`menu`)**  
| Column | Type | Description |
|--------|------|-------------|
| menu_id | INT | 메뉴 고유 ID |
| name | VARCHAR | 메뉴명 |
| category | VARCHAR | 메뉴 카테고리 (커피, 디저트 등) |
| price | DECIMAL | 가격 |
| tags | JSON | 태그 정보 (예: {"달콤한": true, "고소한": false}) |

#### **④ 벡터 저장소 (`vector_menu`)**  
| Column | Type | Description |
|--------|------|-------------|
| menu_id | INT | 메뉴 고유 ID (FK) |
| embedding | VECTOR | 메뉴 설명 및 특징을 임베딩한 벡터 |
| metadata | JSON | 추가적인 메타데이터 (예: 재료, 맛 등) |

---

# DEV

## 🚀 **개발 진행 현황 및 우선순위**  

### ✅ **현재까지 개발된 부분**  

#### 1️⃣ **사용자 관리**  
- ✅ **회원가입** (일반 사용자, 관리자)  
- ✅ **로그인/인증** (JWT 토큰 기반)  
- ✅ **사용자 선호도 저장** (단맛, 신맛, 쓴맛)  

#### 2️⃣ **메뉴 관리**  
- ✅ **메뉴 데이터베이스 구축** (SQLite)  
- ✅ **메뉴 조회 기능**  

#### 3️⃣ **장바구니 & 주문 시스템**  
- ✅ **장바구니 계산 기능**  
- ✅ **주문 생성 및 조회**  
- ✅ **주문 상태 관리** (Pending, Paid, Completed, Cancelled)  

#### 4️⃣ **AI 추천 시스템**  
- ✅ **LLM 기반 자연어 처리** (Mistral)  
- ✅ **개인화된 메뉴 추천**  
- ✅ **협업 필터링 & 콘텐츠 기반 필터링**  
- ✅ **벡터 스토어 기반 검색** (RAG + 파인튜닝)  

#### 5️⃣ **관리자 기능**  
- ✅ **메뉴 관리 (CRUD)**  
- ✅ **주문 관리 및 상태 업데이트**  

---

### ❗ **개발이 필요한 부분**  

#### 🔹 **1. 결제 시스템**  
- ❌ **결제 모듈 연동** (Naver Pay, Toss, Kakao Pay)  
- ❌ **결제 상태 관리**  
- ❌ **영수증 발행**  

#### 🔹 **2. 음성 인터페이스**  
- ❌ **음성 인식 기능** (OpenAI Whisper)  
- ❌ **음성 출력 기능**  

#### 🔹 **3. 프론트엔드 (웹 UI/UX 개발)**  
- ❌ **웹 인터페이스 구축** (Next.js)  
- ❌ **반응형 디자인 적용**  
- ❌ **접근성 고려 (디지털 소외 계층 배려)**  

#### 🔹 **4. 관리자 기능 확장**  
- ❌ **매출 통계 대시보드**  
- ❌ **고객 데이터 분석 기능**  

#### 🔹 **5. 보안 강화**  
- ❌ **결제 정보 암호화**  
- ❌ **API 접근 제한**  
- ❌ **로그 관리 시스템**  

#### 🔹 **6. 기타 기능**  
- ❌ **알림 시스템 (주문 완료 알림 등)**  
- ❌ **리뷰/평점 시스템**  
- ❌ **포인트/적립 시스템**  

---

# TEST

# 카페 추천 AI 시스템 API 테스트 결과
테스트 시작 시간: 2024-03-30 15:57:32

## 1. 발견된 문제점 및 해결
1. URL 리다이렉트 문제
   - 문제: 307 Temporary Redirect 에러 발생
   - 해결: URL 끝에 슬래시(/) 추가 (예: /api/v1/menus/ )

2. 관리자 권한 문제
   - 문제: 잘못된 관리자 계정 정보 사용
   - 해결: config.py에 설정된 올바른 관리자 계정 사용
     - 이메일: admin@example.com
     - 비밀번호: admin123

## 2. API 테스트 결과

### 2.1 회원가입 테스트 (일반 사용자)
**요청 (Request):**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/v1/auth/register' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "email": "test2@example.com",
    "name": "테스트2",
    "password": "test1234!",
    "preferences": {
      "taste": {
        "sweet": 3,
        "sour": 2,
        "bitter": 4
      }
    }
  }'
```

**응답 (Response):**
```json
{
  "email": "test2@example.com",
  "name": "테스트2",
  "preferences": {
    "taste": {
      "sweet": 3,
      "sour": 2,
      "bitter": 4
    }
  },
  "is_active": true,
  "is_admin": false,
  "id": 6,
  "created_at": "2025-03-30T15:48:06",
  "updated_at": null
}
```

### 2.2 관리자 로그인 테스트
**요청 (Request):**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/v1/auth/login' \
  -H 'Content-Type: application/json' \
  -d '{
    "email": "admin@example.com",
    "password": "admin123"
  }'
```

**응답 (Response):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NDQwNDE0NDAsInN1YiI6ImFkbWluQGV4YW1wbGUuY29tIn0.R2_7-87Nd3b6KMeleKKSZqnJYRQVS_VdU0feT9wPQgo",
  "token_type": "bearer"
}
```

### 2.3 메뉴 추가 테스트 (관리자)
**요청 (Request):**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/v1/menus/' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NDQwNDE0NDAsInN1YiI6ImFkbWluQGV4YW1wbGUuY29tIn0.R2_7-87Nd3b6KMeleKKSZqnJYRQVS_VdU0feT9wPQgo' \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "카푸치노",
    "description": "에스프레소와 스팀 밀크, 우유 거품의 완벽한 조화",
    "price": 5000,
    "category": "커피",
    "attributes": {
      "taste": {
        "sweet": 2,
        "sour": 3,
        "bitter": 3
      },
      "temperature": "hot",
      "size": "regular"
    }
  }'
```

**응답 (Response):**
```json
{
  "name": "카푸치노",
  "description": "에스프레소와 스팀 밀크, 우유 거품의 완벽한 조화",
  "price": 5000.0,
  "category": "커피",
  "image_url": null,
  "sweetness": 0.0,
  "sourness": 0.0,
  "bitterness": 0.0,
  "temperature": "both",
  "properties": null,
  "id": 5,
  "order_count": 0,
  "created_at": "2025-03-30T15:57:32.432824",
  "updated_at": "2025-03-30T15:57:32.432830",
  "rating": 0.0
}
```

## 3. 최종 테스트 결과
1. 회원가입 (일반 사용자) - ✅ 성공
2. 일반 사용자 로그인 - ✅ 성공
3. 메뉴 목록 조회 - ✅ 성공
4. 일반 사용자 메뉴 추가 - ❌ 실패 (권한 없음, 예상된 동작)
5. 관리자 로그인 - ✅ 성공
6. 관리자 메뉴 추가 - ✅ 성공



