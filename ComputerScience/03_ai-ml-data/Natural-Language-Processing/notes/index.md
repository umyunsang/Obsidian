# Index

## lecture

* [04. BoW와 TF-IDF - 문서 벡터화, 이진·빈도 단어가방 모델과 두 가지 TF-IDF 계산법](./04.%20BoW%EC%99%80%20TF-IDF%20-%20%EB%AC%B8%EC%84%9C%20%EB%B2%A1%ED%84%B0%ED%99%94%2C%20%EC%9D%B4%EC%A7%84%C2%B7%EB%B9%88%EB%8F%84%20%EB%8B%A8%EC%96%B4%EA%B0%80%EB%B0%A9%20%EB%AA%A8%EB%8D%B8%EA%B3%BC%20%EB%91%90%20%EA%B0%80%EC%A7%80%20TF-IDF%20%EA%B3%84%EC%82%B0%EB%B2%95.md) - 문서를 수치 벡터로 바꾸는 단어가방 모델(BoW)의 두 형태(binary: 등장 여부, count: 등장 횟수)와 흔한 단어의 가중치를 깎는 TF-IDF의 두 계산법(방법1 - 문서 길이 반영 TF와 IDF = ln(N/df), 방법2 - 싸이킷런 TfidfVectorizer의 IDF = ln((N+1)/(df+1))+1 및 L2 정규화)을 강의 예제·과제용 벡터라이저 4개로 직접 조작하며 확인한다. TF는 문서마다 다른 반면 IDF는 단어당 값이 하나라는 구분을 핵심으로 다룬다.
