
---
### LLM 프레임워크: LangChain

- **LangChain**은 LLM(Language Model)을 쉽게 구현하고 통합할 수 있도록 도와주는 오픈 소스 프레임워크입니다.
- **언어**: Python과 JavaScript 라이브러리를 제공
- **출시**: 2022년 10월, 해리슨 체이스가 출시
- **목표**: 표준 인터페이스를 제공하도록 설계
- **링크**:
    - [LangChain GitHub](https://github.com/hwchase17/langchain)
    - [LangChain Python Docs](https://python.langchain.com/docs/introduction/)

---

### LangChain 구성 요소

1. **LangChain 라이브러리**:
    
    - Python과 JavaScript 라이브러리를 포함하며, 다양한 컴포넌트의 인터페이스와 통합을 제공합니다.
    - 이 컴포넌트들을 체인(Chain)과 에이전트(Agent)로 결합할 수 있는 기본 런타임.
2. **LangChain 템플릿**:
    
    - 다양한 작업을 위한 참조 아키텍처 모음으로, 쉽게 배포할 수 있습니다.
3. **LangServe**:
    
    - LangChain 체인을 REST API로 배포할 수 있는 라이브러리입니다.
4. **LangSmith**:
    
    - LLM 프레임워크에서 구축된 체인을 디버깅, 테스트, 평가, 모니터링할 수 있는 개발자 플랫폼입니다.

---

### LangChain 모듈

- **LLM 추상화 (Large Language Model Abstraction)**:
    
    - LLM을 캡슐화하여 일관된 인터페이스를 제공하고, 모델을 쉽게 교체하거나 업그레이드할 수 있습니다. 성능 최적화를 위한 여러 전략을 적용할 수 있습니다.
- **Prompt Template**:
    
    - LLM에 전달되는 입력 텍스트의 구조를 정의하며, 쿼리와 컨텍스트를 구조화합니다.
- **Chain**:
    
    - 여러 LLM 작업을 조합하여 복잡한 작업 흐름을 구성합니다. 입출력을 연결하여 다양한 작업을 수행할 수 있습니다.
- **Index**:
    
    - 정보 검색을 위한 구조로, 학습 데이터 세트에 포함되지 않은 외부 데이터 소스(내부 문서, 이메일 등)에 액세스할 수 있습니다. 이를 위해 Document Loaders, Vector Database, Text Splitters를 사용합니다.
- **Memory**:
    
    - 과거 대화나 상호작용에서 얻은 정보를 저장하는 시스템입니다. 대화 전체를 기억하거나, 지금까지의 대화 요약만 기억하는 옵션이 있습니다.
- **Agent**:
    
    - 사용자의 요청에 따라 어떤 기능을 어떤 순서로 실행할지 결정하는 모듈입니다. 여러 기능과 데이터 소스를 결합하여 사용합니다.

---

### Agent: 기능 수행 결정

- **Agent**는 사용자의 요청에 따라 수행할 기능을 결정합니다.
    - **입력 (Input)**: 사용자가 에이전트에게 작업을 부여하는 방식입니다.
    - **추론 (Thought)**: 에이전트가 무엇을 해야 할지 생각하는 과정입니다.
    - **행동 (Action/Action Input)**: 에이전트가 사용할 도구와 도구에 대한 입력을 결정합니다.
    - **관찰 (Observation)**: 도구의 출력 결과를 관찰합니다.

---

### Agent 종류

1. **ReAct (Reason + Act)**:
    
    - **ReAct**는 추론(Thought)과 행동(Action)을 번갈아가며 반복하는 방식입니다.
    - [ReAct: Reasoning and Acting in Language Models](https://research.google/blog/react-synergizing-reasoning-and-acting-in-language-models/)
2. **대화형 에이전트 (Conversational Agents)**:
    
    - 대화를 기반으로 사용자의 요청에 적절한 반응을 생성합니다.
3. **Zero-shot ReAct**:
    
    - 도구에 대한 설명만으로 사용할 도구를 결정하는 에이전트입니다.
4. **React-docstore**:
    
    - **ReAct**를 사용하여 문서를 보관하는 데이터베이스인 문서 저장소와 상호작용하는 에이전트입니다.
5. **Self-ask-with-search**:
    
    - **Intermediate Answer**라는 도구를 사용하는 에이전트입니다.

---
### LangChain의 Memory 시스템

- **Memory**는 Agent가 사용자의 대화를 기억하고, 추론과 행동을 결정하는 데 필요한 정보를 저장하는 시스템입니다.
- **목적**: Agent와 사용자 간의 대화 내용을 기억하여, 미래에 무엇을 말해야 할지 결정하는 데 활용됩니다.
- **메모리 초기화**: Agent 초기화 시 메모리 시스템을 설정하여 정보를 장기간에 걸쳐 사용할 수 있도록 지원합니다.

---

### Memory 유형

1. **Simple Memory**:
    
    - 간단한 키-값 저장소를 사용하여 정보를 저장합니다.
    - 복잡한 관계나 문맥을 효과적으로 처리하는 데는 한계가 있습니다.
2. **Persistent Memory**:
    
    - 데이터베이스나 파일 시스템을 활용하여 영구적으로 정보를 저장합니다.
    - 장기간 학습 및 상호작용에 유용합니다.
3. **Contextual Memory**:
    
    - 현재 대화의 문맥을 저장하고 관리합니다.
    - 실시간 대화에서 유용하며, 동적 변화에 빠르게 적응할 수 있습니다.
4. **Distributed Memory**:
    
    - 클라우드 기반 또는 분산 데이터베이스를 사용하여 대규모 데이터를 관리합니다.

---

### LangChain에서 제공하는 Memory 모듈

- **ConversationBufferMemory**: 모든 대화 기록을 사용하는 메모리
- **ConversationBufferWindowMemory**: 최근 K번의 대화 기록을 사용하는 메모리
- **ConversationTokenBufferMemory**: 최신 K개 토큰의 대화 기록을 사용하는 메모리
- **ConversationSummaryMemory**: 대화 기록의 요약을 사용하는 메모리
- **ConversationSummaryBufferMemory**: 최신 K개 토큰의 대화 내역 요약을 사용하는 메모리
- **ConversationEntityMemory**: 대화 내 엔티티 정보를 저장하고 필요에 따라 사용하는 메모리
- **ConversationKGMemory**: 지식 그래프 정보를 사용하는 메모리
- **VectorStoreRetrieveMemory**: 대화 기록을 벡터 데이터베이스에 저장하고 상위 K개의 유사한 정보를 사용하는 메모리

[LangChain Memory 문서](https://python.langchain.com/docs/versions/migrating_memory/)

---

### LangSmith: LLM 애플리케이션 개발 및 모니터링

- **목적**: LLM 애플리케이션 개발, 모니터링 및 테스트를 위한 플랫폼
- **Trace 기능**:
    - LLM 애플리케이션의 동작을 추적하여 예상치 못한 결과, 에이전트의 루프, 체인 속도 저하, 토큰 수 등을 분석하는 데 유용합니다.
    - 실행 카운트, 오류 발생률, 토큰 사용량, 과금 정보를 프로젝트 단위로 확인할 수 있습니다.
    - 문서 검색 결과, GPT의 입출력 내용을 기록하여, 검색 알고리즘 및 프롬프트 변경 여부를 판단하는 데 도움을 줍니다.

---

### Runnable 프로토콜

- **목적**: 사용자 정의 체인을 쉽게 생성하고 관리할 수 있도록 설계된 핵심 개념
- **기능**: 다양한 타입의 컴포넌트를 조합하고 복잡한 데이터 처리 파이프라인을 구성할 수 있습니다.

---

### Runnable 메소드

1. **invoke**:
    
    - 주어진 입력에 대해 체인의 모든 처리 단계를 한 번에 실행하고 결과를 반환합니다.
    - 단일 입력에 대해 동기적으로 작동합니다.
2. **batch**:
    
    - 입력 리스트에 대해 체인을 호출하고, 각 입력에 대한 결과를 리스트로 반환합니다.
    - 여러 입력에 대해 동기적으로 작동하며, 배치 처리에 적합합니다.
3. **stream**:
    
    - 입력 데이터를 스트림으로 처리하고, 반복적으로 데이터를 체인을 통해 보내고 결과를 받습니다.
    - 연속적인 데이터 흐름, 대용량 데이터 처리, 실시간 데이터 처리에 유용합니다.
4. **비동기 메소드 (ainvoke, abatch, astream)**:
    
    - 각각의 동기 메소드에 대한 비동기 실행을 지원합니다.

---

### Runnable 객체 유형

1. **RunnablePassthrough**:
    
    - 조건 분기 없이 입력을 받아 다음 단계로 전달합니다.
2. **RunnableMap**:
    
    - 입력 데이터에 함수를 매핑하여 새로운 데이터를 생성합니다.
3. **RunnableFilter**:
    
    - 주어진 조건에 따라 입력 데이터를 필터링합니다.
    - 데이터 스트림에서 특정 조건을 만족하는 데이터만을 추출할 때 사용합니다.
4. **RunnableReduce**:
    
    - 여러 데이터 조각을 하나로 통합하는 리듀스 작업을 수행합니다.
    - 여러 입력값을 받아 하나의 결과로 합치는 과정에서 사용됩니다.
5. **RunnableInvoke**:
    
    - 외부 함수나 메소드를 호출하고 결과를 반환합니다.
    - 외부 시스템과의 통합을 구현할 수 있습니다.
6. **RunnableSequence**:
    
    - 여러 Runnable 객체를 순차적으로 실행하는 시퀀스를 구성합니다.
    - 복잡한 처리 로직을 단계별로 구성할 때 유용합니다.
7. **RunnableParallel**:
    
    - 여러 Runnable 객체를 병렬로 실행하여 동시에 여러 작업을 수행하고 결과를 동시에 수집합니다.

---

### LCEL (LangChain Expression Language)

LCEL은 언어 모델과 상호작용하는 프로세스를 간소화하고 구조화하기 위해 설계된 스크립팅 언어입니다. 여러 구성 요소를 단일 체인으로 결합하여 복잡한 데이터 파이프라인을 구축하고, 데이터 흐름을 명확하게 관리할 수 있습니다.

- **특징**:
    - `|` 기호는 Unix 파이프 연산자처럼 구성 요소들을 연결하고, 한 구성 요소의 출력을 다음 구성 요소의 입력으로 전달합니다.
    - 조건문과 반복문을 지원하여 동적인 데이터 처리에 유연하게 대응합니다.
    - 다른 시스템이나 라이브러리와 쉽게 통합될 수 있도록 설계되었습니다.

---

### LCEL 인터페이스

- **목표**: 사용자 정의 체인을 쉽게 만들 수 있도록, `Runnable` 프로토콜을 구현하여 표준 인터페이스로 사용자 정의 체인을 정의하고 호출할 수 있게 합니다.

#### 주요 메소드

1. **stream**:
    
    - 주어진 토픽에 대한 데이터 스트림을 생성하고, 각 데이터의 내용을 즉시 출력합니다.
2. **invoke**:
    
    - 입력에 대해 체인을 호출하여 처리합니다.
3. **batch**:
    
    - 여러 개의 딕셔너리를 포함하는 리스트를 인자로 받아, 각 딕셔너리의 `topic` 키 값을 사용하여 일괄 처리를 수행합니다.
    - `config` 딕셔너리의 `max_concurrency` 키를 통해 동시에 처리할 수 있는 최대 작업 수를 설정합니다.
4. **astream**:
    
    - 비동기 스트림을 생성하여 주어진 토픽에 대한 메시지를 비동기적으로 처리합니다.
5. **ainvoke**:
    
    - 비동기적으로 입력에 대해 체인을 호출합니다.
6. **abatch**:
    
    - 비동기적으로 입력 목록에 대해 체인을 호출합니다.
7. **astream_log**:
    
    - 최종 응답뿐만 아니라 발생하는 중간 단계를 스트리밍하여 기록합니다.

---

### Prompt

**Prompt**는 사용자와 언어 모델 간의 대화에서 질문이나 요청의 형태로 제시되는 입력문입니다. 모델이 어떤 유형의 응답을 제공할지 결정하는 데 중요한 역할을 합니다.

#### 좋은 프롬프트 구성

- **명확하고 구체적이어야 합니다**:
    
    - 예시: "다음 주 주식 시장에 영향을 줄 수 있는 예정된 이벤트들은 무엇일까요?"는 "주식 시장에 대해 알려주세요."보다 더 구체적입니다.
- **모델이 문맥을 이해할 수 있도록 배경 정보를 제공**:
    
    - 예시: "2020년 미국 대선의 결과를 바탕으로 현재 정치 상황에 대한 분석을 해주세요."
- **핵심 정보에 초점을 맞추고 불필요한 정보는 배제**:
    
    - 예시: "2021년에 발표된 삼성전자의 ESG 보고서를 요약해주세요."
- **열린 질문을 사용하여 자세한 답변을 유도**:
    
    - 예시: "신재생에너지에 대한 최신 연구 동향은 무엇인가요?"
- **정확한 정보나 결과 유형을 정의**:
    
    - 예시: "AI 윤리에 대한 문제점과 해결 방안을 요약하여 설명해주세요."
- **대화의 맥락에 적합한 언어와 문체 선택**:
    
    - 예시: 공식적인 보고서를 요청하는 경우, "XX 보고서에 대한 전문적인 요약을 부탁드립니다."와 같은 정중한 문체 사용.

#### LLM 모델에 입력할 프롬프트 구성 요소

1. **지시**: 언어 모델에게 수행할 작업을 구체적으로 요청하는 지시
2. **예시**: 작업을 수행하는 방법에 대한 예시
3. **맥락**: 작업을 수행하기 위한 추가적인 정보
4. **질문**: 답변을 요구하는 구체적인 질문

---

### PromptTemplate

`PromptTemplate`은 문자열 템플릿을 기반으로 프롬프트를 생성하는 클래스입니다.

#### 예시 코드

```python
from langchain_core.prompts import PromptTemplate

template = "{country}의 수도는 어디인가요?"
prompt = PromptTemplate(template=template, input_variables=["country"])

# 템플릿 생성
print(prompt)

# 템플릿을 특정 값으로 포맷
prompt.format(country="대한민국")
```

`PromptTemplate` 클래스는 문자열을 기반으로 프롬프트를 생성하며, `+` 연산자를 사용하여 템플릿을 결합할 수 있습니다.

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 문자열 템플릿 결합
combined_prompt = (prompt + PromptTemplate.from_template("\n\n아버지를 아버지라 부를 수 없습니다.") + "\n\n{language}로 번역해주세요.")
print(combined_prompt)

# 체인 생성 및 실행
llm = ChatOpenAI(model="gpt-4o-mini")
chain = combined_prompt | llm | StrOutputParser()
chain.invoke({"age": 30, "language": "영어", "name": "홍길동"})
```

---

### ChatPromptTemplate

대화형 상황에서 여러 메시지 입력을 기반으로 단일 메시지 응답을 생성하는 데 사용됩니다.

#### 예시 코드

```python
from langchain_core.prompts import ChatPromptTemplate

# 튜플 형태의 메시지 목록으로 프롬프트 생성
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "이 시스템은 천문학 질문에 답변할 수 있습니다."),
    ("user", "{user_input}")
])

messages = chat_prompt.format_messages(user_input="태양계에서 가장 큰 행성은 무엇인가요?")
print(messages)

# 체인 생성 및 실행
from langchain_core.output_parsers import StrOutputParser
chain = chat_prompt | llm | StrOutputParser()
chain.invoke({"user_input": "태양계에서 가장 큰 행성은 무엇인가요?"})
```

#### SystemMessagePromptTemplate와 HumanMessagePromptTemplate 사용

```python
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("이 시스템은 천문학 질문에 답변할 수 있습니다."),
    HumanMessagePromptTemplate.from_template("{user_input}")
])

messages = chat_prompt.format_messages(user_input="태양계에서 가장 큰 행성은 무엇인가요?")
# print(messages)
chain = chat_prompt | llm | StrOutputParser()
chain.invoke({"user_input": "태양계에서 가장 큰 행성은 무엇인가요?"})
```

---
### LangChain 개념 정리

#### MessagePlaceholder

- **목적**: 메시지 목록에 변수를 동적으로 삽입하려고 할 때 유용
- **사용 예시**:
    
    ```python
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 요약 전문 AI 어시스턴트입니다."),
        MessagesPlaceholder(variable_name="conversation"),
        ("human", "지금까지의 대화를 {word_count} 단어로 요약합니다.")
    ])
    formatted_chat_prompt = chat_prompt.format(word_count=5, conversation=[
        ("human", "안녕하세요! 저는 오늘 새로 입사한 테디 입니다."),
        ("ai", "반가워요! 앞으로 잘 부탁 드립니다."),
    ])
    ```
    

#### Few-shotPromptTemplate

- **목적**: Few-shot 학습을 통해 모델에 예시를 제공하여 정확하고 일관된 응답 생성
- **구성 요소**:
    - `examples`: 예시 데이터
    - `prefix`: 접두사
    - `suffix`: 접미사
    - `input_variables`: 입력 변수 지정
- **사용 예시**:
    
    ```python
    example_prompt = PromptTemplate.from_template("질문: {question}\n{answer}")
    examples = [
        {"question": "지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?", "answer": "지구 대기의 약 78%를 차지하는 질소입니다."},
        {"question": "광합성에 필요한 주요 요소들은 무엇인가요?", "answer": "광합성에 필요한 주요 요소는 빛, 이산화탄소, 물입니다."}
    ]
    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix="질문: {input}",
        input_variables=["input"]
    )
    ```
    

#### FewShotChatMessagePromptTemplate

- **목적**: 고정 예제를 사용한 Few-shot 프롬프팅 기법
- **사용 예시**:
    
    ```python
    examples = [
        {"input": "지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?", "output": "질소입니다."},
        {"input": "광합성에 필요한 주요 요소들은 무엇인가요?", "output": "빛, 이산화탄소, 물입니다."}
    ]
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ])
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )
    ```
    

#### 동적 Few-shot 프롬프팅

- **목적**: ExampleSelector를 사용하여 입력에 맞는 예제를 동적으로 선택
- **사용 예시**:
    
    ```python
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 과학과 수학에 대해 잘 아는 교육자입니다."),
        few_shot_prompt,
        ("human", "{input}")
    ])
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    chain = final_prompt | model
    result = chain.invoke({"input": "지구의 자전 주기는 얼마인가요?"})
    print(result.content)
    ```
    

#### ExampleSelector

- **목적**: 여러 개의 답변 예시 중 가장 적합한 예시를 선택하는 클래스
- **종류**:
    - `LengthBasedExampleSelector`: 문자열 길이를 기준으로 예시 선택
    - `SemanticSimilarityExampleSelector`: 입력과 의미적 유사성을 기준으로 예시 선택
    - `MaxMarginalRelevanceExampleSelector`: 유사성을 최적화하면서 다양한 예시 선택
- **사용 예시**:
    
    ```python
    from langchain_core.example_selectors import SemanticSimilarityExampleSelector
    example_selector = SemanticSimilarityExampleSelector(
        examples=examples,
        embeddings=OpenAIEmbeddings(),
        vectorstore_cls=Chroma,
        k=3
    )
    ```
    
---
### Output Parser

- **PydanticOutputParser**: 언어 모델의 출력을 더 구조화된 정보로 변환. 단순 텍스트 응답 대신, 필요한 정보를 명확하고 체계적인 형태로 제공.
    
- **CommaSeparatedListOutputParser**: 쉼표로 구분된 항목을 리스트 형태로 변환.
    
    ```python
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import CommaSeparatedListOutputParser
    
    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()
    
    prompt = PromptTemplate(
        template="List five {subject}.\n{format_instructions}",
        input_variables=["subject"],
        partial_variables={"format_instructions": format_instructions},
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | output_parser
    chain.invoke({"subject": "popular Korean cusine"})
    ```
    
- **JsonOutputParser**: 모델의 출력을 JSON으로 해석하고, 자료구조를 `Pydantic`을 사용해 정의.
    
    ```python
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.pydantic_v1 import BaseModel, Field
    
    class CusineRecipe(BaseModel):
        name: str = Field(description="name of a cusine")
        recipe: str = Field(description="recipe to cook the cusine")
    
    output_parser = JsonOutputParser(pydantic_object=CusineRecipe)
    format_instructions = output_parser.get_format_instructions()
    
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": format_instructions},
    )
    
    chain = prompt | model | output_parser
    chain.invoke({"query": "Let me know how to cook Bibimbap"})
    ```
    
- **PandasDataFrameOutputParser**: 구조화된 데이터를 다루기 위한 도구 세트를 제공하여, 데이터 정제, 변환, 분석에 유용.
    
- **DatetimeOutputParser**: 출력을 `datetime` 형식으로 파싱.
    
- **EnumOutputParser**: 열거형 값을 처리하는 파서.
    
- **OutputFixingParser**: 출력 파싱 중 발생할 수 있는 오류를 자동으로 수정.
    

---

### Chain

여러 개의 LLM(언어 모델)이나 프롬프트의 입출력을 연결할 수 있는 모듈입니다.

- **LLMChain**: 사용자 입력을 기반으로 프롬프트 템플릿을 생성하고, 이를 사용하여 LLM을 호출합니다.
    
- **SimpleSequentialChain**: 하나의 입출력에 대해 여러 개의 체인을 순차적으로 연결합니다.
    
- **SequentialChain**: 여러 개의 입출력을 가진 체인을 연결합니다.
    
- **RetrievalQA**: 질의응답을 수행하는 체인.
    
- **RetrievalQAWaithSourceChain**: 소스가 포함된 질의응답을 수행합니다.
    
- **SummarizeChain**: 텍스트 요약을 수행하는 체인.
    
- **PALChain**: 입력 질문을 파이썬 코드로 변환하여, 파이썬 REPL을 통해 실행합니다.
    
- **SQLDatabaseChain**: 데이터베이스 질문을 SQL 쿼리로 변환하고 실행합니다.
    
- **LLMMathChain**: 수학 문제를 파이썬 코드로 변환하여 파이썬 REPL로 실행합니다.
    
- **LLMBashChain**: 질문을 bash 명령어로 변환하여 터미널에서 실행합니다.
    
- **LLMCheckerChain**: 질문에 대한 답변을 다른 LLMChain을 통해 확인하고, 그 정확성을 검증합니다.
    
- **LLMRequestsChain**: URL과 파라미터를 입력받아, 이를 기반으로 웹 요청을 생성하고 실행합니다.
    
- **OpenAIModerationChain**: OpenAI의 콘텐츠 모더레이션 API를 사용하여 콘텐츠를 모더레이션합니다.
    
---
