
---
#### LlamaIndex란?
LlamaIndex는 LLM에서 학습되지 않은 데이터를 활용하여 질의응답 AI를 쉽게 개발할 수 있는 오픈소스 라이브러리입니다. 외부 데이터를 LLM에 주입해 더 정확하고 최신의 응답을 생성할 수 있으며, RAG(Retrieval-Augmented Generation) 작업 흐름을 간단한 Python 코드로 구현할 수 있도록 지원합니다. 이 라이브러리는 내부적으로 LangChain을 활용하며, 다음과 같은 특징과 기능을 제공합니다:

- **다양한 데이터 소스 활용**  
    텍스트, PDF, API 등 다양한 형태의 데이터를 처리할 수 있어, 고객 지원 챗봇, 학술 연구, 의료 정보 제공 등 다양한 응용 분야에 적용 가능합니다.
    
- **효율적인 데이터 처리 및 쿼리 기능**  
    데이터를 적재, 구조화, 쿼리, 통합하는 과정을 지원하며, 고급 검색 및 쿼리 인터페이스를 제공합니다. 복잡한 RAG 파이프라인도 간단히 구현 가능하여 사용자 요구에 맞춘 솔루션을 쉽게 개발할 수 있습니다.
    
- **유연성과 확장성**  
    다양한 모델 및 프레임워크와 통합 가능하도록 설계되어, 개인 데이터를 가져오고 구조화하거나 고급 검색 인터페이스를 생성하는 데 유용합니다. ChatGPT와의 통합도 지원합니다.
    
- **참고 자료**  
    관련 내용은 [GitHub](https://github.com/run-llama/llama_index)와 [공식 문서](https://docs.llamaindex.ai/en/stable/examples/data_connectors/simple_directory_reader//) 에서 확인할 수 있습니다.
---
#### LlamaIndex 주요 기능

- **다양한 데이터 소스 지원**
    - 텍스트, PDF, ePub, 워드, 파워포인트, 오디오를 비롯한 다양한 파일 형식과 트위터, 슬랙, 위키피디아 등 웹 서비스를 자체 데이터로 지정 가능
    
- **벡터 임베딩 및 인덱싱**
    - 로드한 데이터를 벡터 임베딩으로 변환하고 효율적인 검색을 위해 인덱싱
    
- **효율적인 검색 알고리즘**
    - 사용자의 쿼리에 대해 가장 관련성이 높은 문서나 데이터 조각을 검색
    
- **LLM 기반 응답 생성**
    - 검색된 관련 문서를 바탕으로 LLM을 활용하여 사용자 쿼리에 대한 정확하고 상세한 응답 생성
    
- **모듈화된 구조 설계**
    - 각 컴포넌트를 필요에 따라 커스터마이즈하거나 교체 가능
    
- **다양한 데이터베이스 및 모델 지원**
    - 벡터 데이터베이스, 임베딩 모델, LLM 등을 유연하게 지원
    
- **쿼리 최적화**
    - 복잡한 쿼리를 자동으로 분해하고 최적화하여 더 정확한 응답 생성
    
- **멀티모달 데이터 처리**
    - 텍스트뿐만 아니라 이미지, 오디오 등 다양한 형태의 데이터를 처리 가능

---
#### LlamaIndex 아키텍처

- **다양한 데이터 소스 지원**
    1. 데이터는 "인덱스" 형태로 변환되어 쿼리에 사용될 수 있도록 준비됨
    2. 사용자가 질문을 입력하면, 쿼리는 인덱스에 전달됨
    3. 인덱스는 사용자의 쿼리와 가장 관련성 높은 데이터를 필터링
    4. 필터링된 관련 데이터, 원래 쿼리, 적절한 프롬프트가 LLM에 전달
    5. 정보를 바탕으로 LLM이 응답 생성
    6. 생성된 응답이 사용자에게 전달됨

---

#### LlamaIndex 주요 구성 요소

- **Loading**
    - 텍스트 파일, PDF 파일, 웹 사이트, 데이터베이스, API 등의 데이터를 파이프라인에 넣는 단계
    - LlamaHub에 다양한 **Connector**가 존재
- **Document**
    - 데이터 소스를 담는 컨테이너 역할
- **Node**
    - LlamaIndex 데이터의 한 단위이며 Document를 작은 청크로 나눔
    - Node에는 메타데이터 포함
- **Connectors**
    - **Reader**라고도 불리며, 다양한 소스에서 데이터를 받아 Document와 Node를 생성
- **Indexing**
    - 데이터를 쿼리할 수 있는 자료 구조를 생성
    - LLM에서는 주로 벡터 임베딩을 생성
- **Indexes**
    - 데이터를 ingest한 뒤 검색(retrieve)하기 쉬운 구조로 변경
    - 주로 벡터 임베딩을 생성하며, 데이터에 대한 메타데이터를 포함할 수 있음
- **Embeddings**
    - LLM은 숫자로 표현된 데이터(임베딩)를 생성
- **Storing**
    - 생성된 인덱스를 저장하여 재사용 가능
- **Querying**
    - 저장된 인덱스를 활용해 효율적인 데이터 검색 및 쿼리 수행
---
