
---
이 코드는 LangChain을 활용하여 PDF 문서를 읽고, 해당 문서의 내용을 기반으로 질문에 답변하는 RAG (Retrieval-Augmented Generation) 구조를 구현한 실습 코드입니다. 

---
## **1. `docs_load()` 함수**
- **역할**: PDF 문서를 읽어들입니다.
- **세부 내용**:
  - `PyPDFLoader`를 사용하여 PDF 문서를 로드하고, `loader`라는 변수에 문서 내용을 저장합니다.
  - 출력된 문서 내용은 `loader`에 저장되어 나중에 다른 함수에서 사용됩니다.

```python
def docs_load():  
    # PDF 문서를 로드하는 함수  
  
    from langchain.document_loaders import PyPDFLoader  
  
    # PDF 문서를 로드하고, 'loader'에 저장  
    loader = PyPDFLoader("Corpus/치과교정용 스마트 페이스마스크를 활용한 스마트 교정 관리.pdf").load()  
  
    # 로드된 문서 출력 (디버깅 용도)  
    print(loader)  
  
    # 로드된 문서를 반환  
    return loader
```

---

## **2. `rc_text_split(corpus)` 함수**
- **역할**: 문서를 분할하여 청크(작은 단위)로 나눕니다.
- **세부 내용**:
  - `RecursiveCharacterTextSplitter`를 사용하여 문서를 분할합니다.
  - 각 청크의 크기는 2000자로 설정하고, 청크 간 500자의 오버랩을 설정합니다.
  - `split_documents` 함수를 통해 문서를 청크로 나눠 반환합니다.

```python
def rc_text_split(corpus):  
    """  
    문서를 청크 단위로 분할하는 함수  
    :param corpus: 분할할 문서 데이터  
    :return: 분리된 청크 리스트  
    """  
    from langchain.text_splitter import RecursiveCharacterTextSplitter  
  
    # 청크 분할을 위한 텍스트 분할기 설정  
    # 문서를 구분할 때 "\n\n", "\n", " ", "" 등을 기준으로 청크 단위로 분할  
    rc_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(  
        separators=["\n\n", "\n", " ", ""],  
        chunk_size=2000,  # 각 청크의 최대 길이 (문자 수)  
        chunk_overlap=500,  # 청크 간 중첩되는 부분의 길이  
        model_name="gpt-4o"  # 텍스트 인코딩에 사용할 모델 지정  
    )  
  
    # 분할된 문서 청크 생성  
    text_documents = rc_text_splitter.split_documents(corpus)  
  
    # 분할된 청크 반환  
    return text_documents
```

---

## **3. `embedding_model()` 함수**
- **역할**: 문서를 벡터화하기 위한 임베딩 모델을 생성합니다.
- **세부 내용**:
  - Hugging Face에서 제공하는 한국어 임베딩 모델(`ko-sroberta-multitask`)을 사용합니다.
  - CPU에서 작동하도록 설정하고, 임베딩을 정규화하여 반환합니다.

```python
def embedding_model():  
    """  
    문서 임베딩을 위한 모델을 생성하는 함수  
    :return: HuggingFace 임베딩 모델  
    """  
    from langchain.embeddings import HuggingFaceEmbeddings  
  
    # 임베딩 모델 설정  
    model_name = "jhgan/ko-sroberta-multitask"  # 한국어 임베딩 모델 선택  
    model_kwargs = {'device': 'cpu'}  # CPU에서 모델 실행  
    encode_kwargs = {'normalize_embeddings': True}  # 임베딩 값을 정규화하여 일관성을 유지  
  
    # 설정된 HuggingFace 임베딩 모델 생성  
    model = HuggingFaceEmbeddings(  
        model_name=model_name,  
        model_kwargs=model_kwargs,  
        encode_kwargs=encode_kwargs  
    )  
  
    # 생성된 임베딩 모델 반환  
    return model
```

---

## **4. `document_embedding(docs, model, save_directory)` 함수**
- **역할**: 청크로 나눈 문서를 임베딩한 후, 벡터저장소에 저장합니다.
- **세부 내용**:
  - 벡터저장소로 `Chroma`를 사용합니다.
  - 기존에 같은 경로에 저장된 벡터저장소가 있으면 삭제하고, 새로 생성된 벡터저장소에 임베딩된 데이터를 저장합니다.

```python
def document_embedding(docs, model, save_directory):  
    """  
    문서를 임베딩하고, Chroma 벡터저장소에 저장하는 함수  
    :param docs: 분할된 문서 청크 리스트  
    :param model: 사용할 임베딩 모델  
    :param save_directory: 벡터저장소 저장 경로  
    :return: 생성된 벡터저장소 데이터베이스 객체  
    """  
    from langchain_community.vectorstores import Chroma  
    import os  
    import shutil  
  
    print("\n잠시만 기다려주세요.\n\n")  
  
    # 기존에 동일 경로에 벡터저장소가 있으면 삭제  
    if os.path.exists(save_directory):  
        shutil.rmtree(save_directory)  
        print(f"디렉토리 {save_directory}가 삭제되었습니다. \n")  
  
    print("문서 벡터화를 시작합니다.")  
  
    # Chroma 벡터저장소에 문서를 임베딩하여 저장  
    db = Chroma.from_documents(docs, model, persist_directory=save_directory)  
    print("새로운 Chroma 데이터베이스가 생성되었습니다.\n")  
  
    # 생성된 데이터베이스 반환  
    return db
```

---

## **5. `chat_llm()` 함수**
- **역할**: OpenAI API를 통해 거대 언어 모델(LLM)을 설정합니다.
- **세부 내용**:
  - `ChatOpenAI` 클래스를 사용하여 GPT-4 모델(`gpt-4o-mini`)을 초기화합니다.
  - API 키는 `.env` 파일에서 불러오며, 스트리밍을 통한 실시간 응답을 설정합니다.

```python
def chat_llm():  
    """  
    OpenAI의 GPT 모델을 사용하여 채팅 LLM (거대 언어 모델)을 생성하는 함수  
    :return: 채팅 LLM 객체  
    """  
    import os  
    from dotenv import load_dotenv  
    from langchain.chat_models import ChatOpenAI  
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  
  
    # .env 파일에서 환경 변수 로드 (API 키 가져오기 위함)  
    load_dotenv('.env')  
  
    # OpenAI API를 사용하여 GPT-4o 모델 초기화  
    llm = ChatOpenAI(  
        model="gpt-4o-mini",  # 사용할 모델  
        api_key=os.getenv("OPENAI_API_KEY"),  # API 키 설정  
        temperature=0,  # 모델의 출력 변동성을 제어 (0은 고정된 출력)  
        streaming=True,  # 스트리밍 방식으로 답변  
        callbacks=[StreamingStdOutCallbackHandler()]  # 실시간 스트리밍 출력을 위한 핸들러  
    )  
  
    # 생성된 LLM 반환  
    return llm
```

---

## **6. `qna(llm, db)` 함수**
- **역할**: 사용자와의 질의응답 루프를 수행합니다.
- **세부 내용**:
  - `Y`를 입력하면 계속 질문을 받을 수 있고, `N`을 입력하면 종료됩니다.
  - 각 질문에 대해 `db_qna` 함수를 호출하여 답변을 받습니다.

```python
def qna(llm, db):  
    """  
    사용자 질문을 처리하는 함수  
    :param llm: 채팅 LLM 객체  
    :param db: 벡터저장소 데이터베이스 객체  
    :return: 질의응답 결과 리스트  
    """  
    qna_result = []  # 질의응답 결과를 저장할 리스트  
  
    check = 'Y'  # 사용자로부터 계속 질문할지 여부를 입력받음  
  
    while check == 'Y' or check == 'y':  # 'Y'나 'y'를 입력하면 계속 질문  
        query = input("질문을 입력하세요 : ")  # 사용자로부터 질문을 입력받음  
        response = db_qna(llm, db, query)  # 입력된 질문을 처리하여 답변 생성  
  
        qna_result.append({'query': query, 'response': response})  # 질문과 답변을 리스트에 저장  
  
        check = input("\n\nY: 계속 질문한다.\nN: 프로그램 종료\n입력: ")  # 사용자에게 계속 질문할지 묻는 입력  
  
    return qna_result  # 질의응답 결과 반환
```

---

## **7. `db_qna(llm, db, query)` 함수**
- **역할**: 벡터저장소에서 관련 문서를 검색하여 거대 언어 모델이 적절한 답변을 하도록 합니다.
- **세부 내용**:
  - `mmr` 방식으로 검색을 수행하며, 검색된 문서들을 바탕으로 답변합니다.
  - `ChatPromptTemplate`를 사용하여 시스템과 사용자의 메시지를 정의한 후, 문서 내용(`context`)과 질문을 기반으로 답변을 생성합니다.

```python
def db_qna(llm, db, query):  
    """  
    데이터베이스에서 검색된 정보를 바탕으로 LLM이 답변을 생성하는 함수  
    :param llm: 채팅 LLM 객체  
    :param db: 벡터저장소 데이터베이스 객체  
    :param query: 사용자 질문  
    :return: LLM이 생성한 답변  
    """  
    from langchain.prompts import ChatPromptTemplate  
    from langchain.schema.runnable import RunnableLambda, RunnablePassthrough  
    from langchain_core.output_parsers import StrOutputParser  
  
    # 데이터베이스에서 검색한 내용을 가져올 설정  
    db = db.as_retriever(  
        search_type="mmr",  # 다중 증강 검색 방식 사용  
        search_kwargs={'k': 3, 'fetch_k': 5}  # 최종 검색 결과 3개 반환, 5개 문서 검색  
    )  
  
    # 프롬프트 템플릿을 설정하여 시스템과 사용자의 메시지를 정의  
    prompt = ChatPromptTemplate.from_messages(  
        [  
            (  
                "system",  
                """  
                You are a specialized AI for question-and-answer tasks.                You must answer questions based solely on the Contest provided.  
                Context: {context}                """            ),  
            ("human", "Question: {question}"),  
        ]  
    )  
  
    # 검색된 문서를 컨텍스트로, 사용자 질문을 입력으로 LLM에 전달하여 답변 생성  
    chain = {  
                "context": db | RunnableLambda(format_docs),  # 검색된 문서 포맷팅  
                "question": RunnablePassthrough()  # 사용자 질문 그대로 전달  
            } | prompt | llm | StrOutputParser()  # 프롬프트, LLM, 출력 파서로 연결된 체인  
  
    response = chain.invoke(query)  # 질문에 대한 답변 생성  
  
    return response  # 생성된 답변 반환
```

---

## **8. `format_docs(docs)` 함수**
- **역할**: 검색된 문서를 문자열로 변환합니다.
- **세부 내용**:
  - 각 문서의 내용을 페이지 단위로 이어붙여 하나의 문자열로 만듭니다.

```python
def format_docs(docs):  
    """  
    검색된 문서들을 하나의 문자열로 포맷팅하는 함수  
    :param docs: 검색된 문서 리스트  
    :return: 하나의 문자열로 결합된 문서 내용  
    """  
    # 각 문서의 페이지 내용을 "\n\n"으로 구분하여 결합한 문자열로 반환  
    return "\n\n".join(document.page_content for document in docs)
```

---

## **9. `run()` 함수**
- **역할**: 전체 프로세스를 실행하는 메인 함수입니다.
- **세부 내용**:
  - 문서 로드, 텍스트 분할, 문서 임베딩, LLM 생성, 질의응답을 차례로 수행합니다.
  - 마지막으로 사용자의 질문에 대해 답변을 합니다.

```python
def run():  
    """  
    전체 프로세스를 실행하는 메인 함수  
    문서 로드 -> 텍스트 분할 -> 문서 임베딩 -> 벡터저장소 저장 -> 질문 응답  
    """  
    # 1. 문서 로드  
    loader = docs_load()  
  
    # 2. 문서 분할  
    chunk = rc_text_split(loader)  
  
    print(chunk)  # 분할된 청크 출력 (디버깅 용도)  
    print(len(chunk))  # 청크의 개수 출력  
  
    # 3. 임베딩 모델 생성  
    model = embedding_model()  
  
    # 4. 문서 임베딩 및 벡터저장소 저장  
    db = document_embedding(chunk, model, save_directory="./chroma_db")  
  
    # 5. 채팅에 사용할 거대언어모델(LLM) 생성  
    llm = chat_llm()  
  
    # 6. 질의응답 처리  
    qna_list = qna(llm, db)  
  
    print(qna_list)  # 질의응답 결과 출력

if __name__ == "__main__":
    run()
```

---
