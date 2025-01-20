
---
#### OpenAI API
```
OpenAI API 주요 기능 : 
	텍스트 생성 
	이미지 생성 
	임베딩 생성 
	파인튜닝 
	모델링 
	음성 텍스트 변환(STT)
```
- Elon Musk, Sam Altman, Greg Brockman 등의 창업자들에 의해 2015년에 설립 
- 자연어 처리, 강화학습, 생성 모델링 등의 기술과 알고리즘을 개발하는 데 중점을 둠
- 자연어 처리 API 
- https://platform.openai.com
- 애플리케이션에 GPT-4, GPT-3.5의 기능을 통합할 수 있습니다. 
- OpenAI API 라이브러리가 지원하는 프로그래밍 언어는 Python, Node.js
- https://platform.openai.com/docs/libraries
#### ChatGPT 모델 이해
##### GPT (Generative Pre-trained Transformer)
- Transformer의 디코더(Decoder) 아키텍처를 기반으로 설계 
	- Encoder는 입력 문장을 종합적으로 이해하고 고차원 표현으로 변환하는 데 적합 
	- Decoder는 이미 주어진 문맥에서 새로운 텍스트를 생성하는 데 더 적합 
- 자기회귀적 모델 (Autoregressive) 
	- 이전의 입력 토큰들을 사용해 다음 토큰을 예측 
- Causal Masking (인과적 마스킹) 
	- 입력 시퀀스에서 미래 정보를 보지 않도록 제한 
	- 모델은 현재 시점 이전의 정보만 이용하여 다음 단어를 예측 
- 단방향(순방향) Attention
#### GPT 모델 Layer
- Input Embedding Layer 
	- 입력 텍스트를 고정된 크기의 벡터로 변환 
	- GPT는 BPE(Byte Pair Encoding) 같은 토크나이저를 사용하여 단어를 더 작은 단위로 분 해 
	- 토큰 임베딩과 위치 임베딩(순서 정보가 유지되도록)을 포함
- Transformer Block (Decoder Block) 
	- 입력 시퀀스 내의 모든 토큰 간의 관계를 모델링하여 각 토큰의 문맥적 의미를 파악 
	- Multi-Head Self-Attention Mechanism : 여러 개의 어텐션 헤드를 사용하여 다양한 표현 을 동시에 학습하고, 각 토큰이 다른 모든 토큰과 어떻게 관련되는지를 평가합니다. 
	- Position-wise Feed-Forward Neural Networks : 각 위치의 토큰에 대해 독립적으로 작동 하는 두 개의 선형 변환과 하나의 비선형 활성화 함수(ReLU)로 구성
		- 모델의 표현력을 향상시킵니다. 
		- Feed-Forward Layer는 Attention 출력 결과를 고차원 공간에서 추가로 변환 
		- 각 계층에 활성화 함수(ReLU 등)가 포함됩니다
	- Residual Connections and Layer Normalization : 각 서브 레이어(Attention과 Feed- Forward Network)의 출력에 대해 레이어 정규화를 수행하고, 각 서브 레이어의 입력에 대한 잔차(오차) 연결을 추가하여 깊은 네트워크에서도 안정적인 학습을 도모합니다. 각 계층의 출력이 입력에 더해져 정보 손실을 방지하고, 학습 안정성을 높입니다
-  Output Layer 
	- 선형 레이어로 구성되며, 소프트맥스 함수를 적용하여 각 토큰의 발생 확률을 계산 
	- 높은 확률을 가진 단어가 출력으로 생성

>[!In-Context Learning]
>- 학습 과정 없이, 주어진 입력 내에서 직접 문맥 정보를 활용하여 작업을 수행하는 방식 
>- 모델이 주어진 예시들을 바탕으로 새로운 입력에 대한 적절한 반응을 생성 
>- 입력된 예시의 질과 양에 따라 모델의 성능이 크게 영향을 받습니다
- Zero-shot Learning 
	- 모델이 한 번도 학습하지 않은 새로운 작업이나 개념을 해결하는 능력 
	- 생성 AI에게 필요한 데이터나 설명을 덧붙이지 않고 답변을 생성하게 하는 방법 
	- 모델의 사전 학습된 능력을 바탕으로 결과를 예측하며, 가중치 업데이트(gradient update)는 수행되지 않습니다 
- One-shot Learning 
	- 몇모델은 단일 예시를 기반으로 작업을 추론합니다. 
	- 가중치 업데이트는 여전히 수행되지 않습니다. 
- Few-shot Learning 
	- 여러 개의 예시를 통해 힌트를 주면서 답변을 생성하는 방법 
	- 모델이 새로운 작업을 학습할 때, 소량의 데이터(몇 개의 예제) 만으로도 문제를 해결하는 능력 
	- 가중치 업데이트 없이 제공된 문맥 정보만 활용됩니다

>[!Fine-Tuning]
>- 사전 훈련된 모델을 특정 작업에 맞게 추가적으로 학습시키는 과정 
>- 모델은 초기에 학습된 일반적인 지식을 바탕으로, 더 구체적이고 특화된 작업을 더 효과적 으로 수행할 수 있습니다. 
>- 사전 훈련된 모델의 파라미터를 특정 작업의 데이터에 맞게 조정합니다. 
>- 특정 작업에 대해 모델의 정확도와 효율성을 크게 개선합니다

