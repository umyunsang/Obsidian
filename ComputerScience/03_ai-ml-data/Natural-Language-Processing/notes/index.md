# Index

## assignment

* [04-1. BoW 벡터화 과제 - 6차원 단어사전과 binary·count 벡터 풀이](./04-1.%20BoW%20%EB%B2%A1%ED%84%B0%ED%99%94%20%EA%B3%BC%EC%A0%9C%20-%206%EC%B0%A8%EC%9B%90%20%EB%8B%A8%EC%96%B4%EC%82%AC%EC%A0%84%EA%B3%BC%20binary%C2%B7count%20%EB%B2%A1%ED%84%B0%20%ED%92%80%EC%9D%B4.md) - NLP 4장 BoW 과제 풀이. 세 개 영문 문서(i love machine learning / i love deep deep learning learning / i like deep learning)에서 정렬된 6개 단어사전(deep, i, learning, like, love, machine)을 도출해 벡터 차원을 결정하고, 등장 여부만 보는 binary BoW와 등장 횟수까지 반영하는 count BoW로 각각 6차원 벡터를 구한다. 중복 단어(deep·learning 각 2회)가 있는 문서2만 두 방식의 결과가 달라지는 점을 인터랙티브 벡터라이저로 확인한다.

## lecture

* [04. BoW와 TF-IDF - 문서 벡터화, 이진·빈도 단어가방 모델과 두 가지 TF-IDF 계산법](./04.%20BoW%EC%99%80%20TF-IDF%20-%20%EB%AC%B8%EC%84%9C%20%EB%B2%A1%ED%84%B0%ED%99%94%2C%20%EC%9D%B4%EC%A7%84%C2%B7%EB%B9%88%EB%8F%84%20%EB%8B%A8%EC%96%B4%EA%B0%80%EB%B0%A9%20%EB%AA%A8%EB%8D%B8%EA%B3%BC%20%EB%91%90%20%EA%B0%80%EC%A7%80%20TF-IDF%20%EA%B3%84%EC%82%B0%EB%B2%95.md) - 텍스트를 컴퓨터가 다룰 수 있는 수치 벡터로 바꾸는 문서 벡터화(Vectorization)의 출발점인 단어가방 모델(BoW, Bag of Words)의 두 형태(binary BoW: 등장 여부, count BoW: 등장 횟수), 단어사전(vocabulary) 구축 시 순서의 중요성과 차원의 저주·희소 행렬 한계, 그리고 흔한 단어의 가중치를 깎는 TF-IDF의 두 가지 계산법(방법1 - 문서 길이 반영 TF와 IDF = ln(N/df), 방법2 - 싸이킷런 TfidfVectorizer의 IDF = ln((N+1)/(df+1))+1 및 L2 정규화)을 3문서 BoW 과제와 단계별 TF-IDF 계산기로 학습한다.
