# Index

## lecture

* [04. BoW와 TF-IDF - 문서 벡터화, 이진·빈도 단어가방 모델과 두 가지 TF-IDF 계산법](./04.%20BoW%EC%99%80%20TF-IDF%20-%20%EB%AC%B8%EC%84%9C%20%EB%B2%A1%ED%84%B0%ED%99%94%2C%20%EC%9D%B4%EC%A7%84%C2%B7%EB%B9%88%EB%8F%84%20%EB%8B%A8%EC%96%B4%EA%B0%80%EB%B0%A9%20%EB%AA%A8%EB%8D%B8%EA%B3%BC%20%EB%91%90%20%EA%B0%80%EC%A7%80%20TF-IDF%20%EA%B3%84%EC%82%B0%EB%B2%95.md) - 텍스트를 컴퓨터가 다룰 수 있는 수치 벡터로 바꾸는 문서 벡터화(Vectorization)의 출발점인 단어가방 모델(BoW, Bag of Words)의 두 형태(binary BoW: 등장 여부, count BoW: 등장 횟수), 단어사전(vocabulary) 구축 시 순서의 중요성과 차원의 저주·희소 행렬 한계, 그리고 흔한 단어의 가중치를 깎는 TF-IDF의 두 가지 계산법(방법1 - 문서 길이 반영 TF와 IDF = ln(N/df), 방법2 - 싸이킷런 TfidfVectorizer의 IDF = ln((N+1)/(df+1))+1 및 L2 정규화)을 3문서 BoW 과제와 단계별 TF-IDF 계산기로 학습한다.
