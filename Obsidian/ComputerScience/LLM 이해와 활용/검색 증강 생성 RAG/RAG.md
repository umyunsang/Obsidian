
---
### RAG (Retrieval-Augmented Generation) 개요

![[Pasted image 20250121150551.png]]

**RAG**는 생성형 AI 모델에 실시간 정보 검색 능력을 결합한 접근 방식입니다. 대형 언어 모델(LLM)이 외부 지식 소스와 연계하여 모델의 범용성과 적응력을 유지하면서도 정확하고 신뢰할 수 있는 답변을 생성할 수 있게 합니다. 주요 응용 분야로는 질문 응답 시스템(QA), 지식 검색, 고객 지원 등이 있습니다.

---

### 챗GPT의 한계와 RAG의 해결책

#### 챗GPT의 한계

1. **고정된 지식**: GPT-3.5까지의 챗GPT는 학습 데이터의 기준 시점에 고정되어 최신 정보를 반영하지 못함.
2. **한정된 도메인 지식**: 특정 분야의 깊이 있는 전문 지식을 제공하는 데 한계가 있음.
3. **맥락 이해의 한계**: 사용자의 특정 상황이나 맥락을 완벽히 이해하고 반영하기 어려움.
4. **환각(Hallucination)**: 잘못된 정보를 사실인 것처럼 제시하는 경우가 있음.

#### RAG의 해결책

1. **최신 정보 접근**: 실시간으로 외부 데이터베이스를 검색하여 최신 정보를 반영 가능.
2. **전문 지식 강화**: 특정 도메인의 전문 데이터베이스를 연결하여 깊이 있는 전문 지식 제공.
3. **맥락 인식 향상**: 사용자의 질문과 관련된 구체적인 정보를 검색하여 더 정확한 맥락 이해 가능.
4. **정보의 신뢰성 향상**: 검색된 실제 데이터를 바탕으로 답변을 생성하여 환각 현상 감소.

---

### RAG 파이프라인

1. **로딩 (Loading)**: 다양한 소스(텍스트 파일, PDF, 웹사이트, 데이터베이스, API 등)에서 데이터를 가져와 파이프라인에 입력.
    - LlamaHub에서 제공하는 다양한 커넥터를 활용할 수 있음.
    
2. **인덱싱 (Indexing)**: 데이터를 쿼리 가능한 구조로 변환.
    - 주로 벡터 임베딩을 생성하여 데이터의 의미를 수치화하고, 관련 메타데이터와 함께 저장.
    
3. **저장 (Storing)**: 생성된 인덱스와 메타데이터를 저장하여 재사용 가능하게 함.
    
4. **쿼리 (Querying)**: LLM과 LlamaIndex 데이터 구조를 활용하여 데이터를 검색.
    - 서브쿼리, 다단계 쿼리, 하이브리드 전략 등 다양한 방식으로 데이터를 검색.
    
5. **평가 (Evaluation)**: 파이프라인의 효과성을 객관적으로 측정.
    - 응답의 정확성, 충실도, 속도 등을 평가.

---

### RAG 작동 원리

1. **색인 작업 (Indexing)**
    - 다양한 외부 데이터 소스(예: 코드 파일, PDF, 텍스트 문서, 이미지 등)에서 정보를 추출.
    - **로드** → **분할** → **임베딩**: 분할된 데이터를 벡터 형태로 변환하고, 벡터 스토어에 저장.
    
2. **Retrieval (검색 단계)**
    - **질문 분석 (Question Analysis)**: 사용자의 질문을 벡터 형태로 변환.
    - **벡터 검색(Vector Search)**: 입력 문장과 관련 있는 문서를 검색 (유사도 검색, TF-IDF, BM25 등).
    
3. **Augmentation (정보 보강 단계)**
    - 검색된 정보를 언어 모델의 입력으로 제공.
    - LLM이 기본적으로 이해하는 언어적 지식에 검색된 데이터를 보강.
    
4. **Generation (생성 단계)**
    - LLM이 보강된 데이터를 바탕으로 자연스러운 텍스트 생성.
    - 응답은 단순히 검색된 내용을 복사하는 것이 아니라, 문맥을 바탕으로 생성.

---

### RAG 장점

1. **시간과 비용 절감**: Fine-tuning에 비해 적은 시간과 비용이 소요됨. 외부 데이터베이스를 활용하므로 별도의 학습 데이터 준비가 필요 없음.
2. **모델의 일반성 유지**: 특정 도메인에 국한되지 않고 다양한 분야에 대한 질문에 답변할 수 있음.
3. **답변의 근거 제공**: 답변과 함께 정보 출처를 제공하여 신뢰도를 높일 수 있음.
4. **할루시네이션 가능성 감소**: 외부 데이터를 기반으로 답변을 생성하여 모델 자체의 오류나 편향을 줄일 수 있음.

---

### Data Load 클래스 (참고)

**데이터 소스에서 정보를 읽어 들여 처리하는 다양한 클래스들**

- **WebBaseLoader**: 웹 기반 데이터 소스로부터 데이터를 로드 (웹 페이지나 API에서 사용).
- **TextLoader**: 일반 텍스트 데이터를 읽어서 LangChain에서 사용할 수 있는 형태로 변환.
- **DirectoryLoader**: 프로젝트 폴더나 지정된 경로 내의 다수 파일을 일괄적으로 처리.
- **CSVLoader**: 테이블 형태의 데이터를 처리하거나 데이터 분석 작업에 사용.
- **PyPDFLoader**: PDF 파일에서 텍스트를 추출 (PyPDF2 라이브러리 사용).
- **UnstructuredPDFLoader**: 복잡하거나 일정하지 않은 레이아웃을 가진 PDF에서 텍스트 추출.
- **PyMuPDFLoader**: PyMuPDF (Fitz) 라이브러리를 사용하여 PDF 파일로부터 텍스트 추출.
- **OnlinePDFLoader**: 웹 URL을 통해 접근 가능한 PDF 문서 처리.
- **PyPDFDirectoryLoader**: 디렉토리 내의 모든 PDF 파일을 로드하여 데이터를 추출.

---

### RAG: 데이터 로딩 및 텍스트 분할

#### 데이터 로딩 (Data Load)

- **WebBaseLoader** 클래스를 사용하여 웹페이지의 텍스트 데이터를 추출하고 Document 객체 리스트로 변환.

```python
# RAG : Load Data
# pip install langchain_community

from langchain_community.document_loaders import WebBaseLoader
url = 'https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EC%A0%95%EC%B1%85%EA%B3%BC_%EC%A7%80%EC%B9%A8'
loader = WebBaseLoader(url)
docs = loader.load()
print(len(docs))
print(len(docs[0].page_content))
print(docs[0].page_content[5000:6000])
```

#### 텍스트 분할 (Text Split)

- **RecursiveCharacterTextSplitter**를 사용하여 긴 문서를 작은 청크로 분할.
- 청크 크기: 1000자, 200자 중복하여 문맥 유지.

```python
# RAG : 텍스트 분할(Text Split)
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
											   chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print(len(splits))
print(splits[10].page_content)
print(splits[10].metadata)
```

---

### RAG: 인덱싱 및 임베딩 생성

#### 인덱싱 (Indexing)

- 텍스트를 임베딩으로 변환하고 벡터 저장소에 저장 후 유사성 검색을 수행.

```python
# RAG : 인덱싱(Indexing)
# pip install langchain_openai
# pip install chromadb

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(documents=splits,
									embedding=OpenAIEmbeddings())

# Vector Database에 저장된 내용을 유사도 검색으로 확인
docs = vectorstore.similarity_search("격하 과정에 대해서 설명해주세요.")
print(len(docs))
print(docs[0].page_content)
```

#### **벡터 임베딩 전략**

- **모델 선택**: 고성능 모델(text-embedding-ada-002 등) 활용.
- **차원 최적화**: 1536 차원이 표준.
- **정규화 및 배치 처리**: 대규모 데이터의 효율적 처리.

---

### RAG: 검색 및 생성

#### 검색 및 생성 (Retrieval & Generation)

- 사용자 질문에 관련된 정보를 검색하고 LLM에 입력하여 답변을 생성.

```python
template = '''Answer the question based only on the following context: 
{context} 


Question: {question}'''
# Template기반의 Prompt객체 (사용자의 질문 포함)
prompt = ChatPromptTemplate.from_template(template)   
# 검색결과 기반으로 응답 텍스트 생성할 LLM모델 생성
model = ChatOpenAI(model='gpt-4o-mini', temperature=0) 
# VectorDB의 검색 엔진 객체 생성
retriever = vectorstore.as_retriever()                 
rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt 
    | model 
    | StrOutputParser()
)
rag_chain.invoke("격하 과정에 대해서 설명해주세요.")
```

---

### RAG: 검색기(Retriever) 최적화

#### 검색기 최적화 전략

- **BM25 Retriever**: 정확한 단어 매칭에 강점.
- **Vector Store Retriever**: 의미론적 유사성을 기반으로 검색.
- **MultiQuery Retriever**: 여러 질의 생성으로 더 많은 관련 문서 검색.
- **Ensemble Retriever**: 다양한 검색 방법 결합으로 더 정확한 검색.
- **MMR (Maximal Marginal Relevance)**: 결과의 다양성과 관련성 균형.

---

### RAG: 순위 재조정 (Rerank)

- **Rerank**: 검색된 문서들의 순위를 재조정하여 더 관련성 높은 문서를 상위에 배치.
- **한국어 특화 Reranker**: 한국어 데이터에 특화된 'Dongjin-kr/ko-reranker' 모델 사용.

---
### RAG 기반 Q&A 웹서비스 개발

#### 1. 필요한 라이브러리 설치

RAG (Retrieval-Augmented Generation) 기반 Q&A 웹서비스를 개발하려면 필요한 라이브러리들을 설치해야 합니다. 아래 명령어로 필요한 라이브러리들을 설치합니다:

```bash
pip install langchain langchain_openai chromadb streamlit Wikipedia langchain_community
```

- `WikipediaLoader` : Wikipedia에서 문서를 로드합니다. 
- `RecursiveCharacterTextSplitter` : 긴 문서를 작은 청크로 분할합니다. 
- `OpenAIEmbeddings` : 텍스트를 벡터로 변환합니다. 
- `Chroma` : 벡터 데이터베이스로 사용됩니다. 
- `ChatOpenAI` : OpenAI의 GPT 모델을 사용합니다. 
- `RetrievalQA` : 검색-질문 응답 체인을 생성합니다. 
- `PromptTemplate` : 사용자 정의 프롬프트를 생성합니다.

---

#### 2. Streamlit 예제

Streamlit을 사용하면 파이썬 코드를 이용해 간단하게 웹 애플리케이션을 만들 수 있습니다. 예를 들어, 아래와 같은 코드를 사용하여 랜덤 데이터를 표시하고, 간단한 차트를 그릴 수 있습니다.

```python
import streamlit as st  # streamlit 라이브러리 임포트
import pandas as pd  # 데이터 처리를 위한 pandas 임포트
import numpy as np  # 숫자 계산을 위한 numpy 임포트

# Streamlit 애플리케이션의 제목 설정
st.title('Streamlit Example')

# 랜덤 데이터를 생성하여 DataFrame으로 변환
data = pd.DataFrame(np.random.randn(50, 3), columns=['a', 'b', 'c'])

# 생성된 데이터 표시
st.write("Here's our random data:")
st.write(data)

# 생성된 데이터로 라인 차트 그리기
chart = st.line_chart(data)
```

- `st.title('Streamlit Example')`: 애플리케이션의 제목을 설정합니다.
- `pd.DataFrame(np.random.randn(50, 3), columns=['a', 'b', 'c'])`: 랜덤한 숫자 데이터 50개, 3개의 열을 생성하고 DataFrame 형식으로 저장합니다.
- `st.write(data)`: DataFrame을 웹 페이지에 표시합니다.
- `st.line_chart(data)`: DataFrame의 데이터를 기반으로 라인 차트를 그립니다.

---

#### 3. Streamlit 애플리케이션 실행

Streamlit 애플리케이션을 실행하려면 아래 명령어를 사용합니다. `your_script.py`는 사용자가 작성한 파이썬 스크립트 파일을 의미합니다.

```bash
streamlit run your_script.py
```

위 명령어를 입력하면, Streamlit 웹 애플리케이션이 자동으로 실행되어 웹 브라우저에서 애플리케이션을 확인할 수 있습니다.

---
