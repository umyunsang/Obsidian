
---
### Bootstrap 방법론

부트스트랩은 통계량의 분포를 이해하고 p-값을 계산하기 위한 최신 통계 기법입니다(p-값은 과학적 주장이 잘못되었을 확률을 의미합니다). 1979년 스탠포드에서 수학자들이 컴퓨터와 컴퓨터 시뮬레이션을 사용하여 확률을 더 잘 이해할 수 있는 방법을 연구하던 중에 발명되었습니다.

#### 주요 통찰

1. **기저 분포(𝐹)에 대한 접근**:
   - 만약 우리가 기저 분포(𝐹)에 접근할 수 있다면, 통계치의 정확도에 대한 거의 모든 질문에 쉽게 답할 수 있습니다.
   - 예를 들어, 이전 섹션에서 표본 크기 n의 샘플에서 샘플 분산을 계산하는 공식을 제공했습니다. 기대값에서 우리의 샘플 분산은 실제 분산과 동일합니다. 그러나 계산된 값이 일정 범위 내에 있을 확률을 알고 싶다면 어떻게 할까요? 이 질문은 과학적 주장을 평가하는 데 중요합니다.
   - 기저 분포(𝐹)를 알고 있다면, 크기 n의 샘플을 여러 번 추출하고 샘플 분산을 계산하여 특정 범위 내에 속하는 비율을 테스트할 수 있습니다.

2. **샘플로부터 기저 분포 추정**:
   - 부트스트랩의 다음 통찰은 샘플 자체가 𝐹에 대한 최고의 추정치라는 것입니다.
   - 가장 간단한 방법은 샘플에서 k가 나타난 비율을 𝑃(𝑋=𝑘)으로 가정하는 것입니다. 이는 추정된 분포 𝐹^의 확률 질량 함수를 정의합니다.

```python
def bootstrap(sample):
   N = number of elements in sample
   pmf = estimate the underlying pmf from the sample
   stats = []
   repeat 10,000 times:
      resample = draw N new samples from the pmf
      stat = calculate your stat on the resample
      stats.append(stat)
   stats can now be used to estimate the distribution of the stat
```

#### 부트스트랩의 유효성

- 부트스트랩은 샘플이 모집단 분포와 비슷하게 생겼다면 합리적인 방법입니다. 무작위로 선택된 대부분의 샘플은 모집단과 유사할 것입니다.
- 예를 들어, $Var(S^2)$를 계산하려면 각 리샘플 i에 대해 $S_i^2$를 계산하고 10,000번 반복 후 $S_i^2$의 샘플 분산을 계산할 수 있습니다.
- 리샘플의 크기가 원래 샘플(n)과 동일한 이유는 계산하는 통계량의 변동이 샘플 크기에 따라 달라질 수 있기 때문입니다. 통계량의 분포를 정확하게 추정하려면 동일한 크기의 리샘플을 사용해야 합니다.

부트스트랩은 강력한 이론적 보장을 가지고 있으며, 과학계에서 널리 인정받고 있습니다. 그러나 기저 분포에 긴 꼬리가 있거나 샘플이 I.I.D가 아닌 경우에는 문제가 발생할 수 있습니다.

## p-값 계산 예시

우리는 부탄과 네팔의 사람들의 행복도를 비교하고자 합니다. 부탄에서 n1=200명, 네팔에서 n2=300명을 샘플로 추출하여 행복도를 1에서 10까지 평가하도록 합니다. 두 샘플의 평균을 측정한 결과, 네팔의 사람들이 약간 더 행복한 것으로 나타났습니다. 네팔 샘플 평균과 부탄 샘플 평균의 차이는 0.5점입니다.

이 주장을 과학적으로 만들기 위해 p-값을 계산해야 합니다. p-값은 귀무가설이 참일 때, 측정된 통계치가 보고된 값 이상일 확률입니다. 귀무가설은 두 그룹 사이에 차이가 없다는 가설입니다.

네팔과 부탄을 비교할 때, 귀무가설은 네팔과 부탄의 행복도 분포에 차이가 없다는 것입니다. 샘플을 추출할 때 네팔의 평균이 부탄보다 0.5점 높게 나왔다는 것은 우연의 일치입니다.

부트스트랩을 사용하여 p-값을 계산할 수 있습니다. 먼저, 부탄과 네팔의 모든 샘플로부터 확률 질량 함수를 만들어 귀무가설의 기저 분포를 추정합니다.

```python
def pvalue_bootstrap(bhutan_sample, nepal_sample):
   N = size of the bhutan_sample
   M = size of the nepal_sample
   universal_sample = combine bhutan_samples and nepal_samples
   universal_pmf = estimate the underlying pmf of the universalSample
   count = 0
   observed_difference = mean(nepal_sample) - mean(bhutan_sample)
   repeat 10,000 times:
      bhutan_resample = draw N new samples from the universalPmf
      nepal_resample = draw M new samples from the universalPmf
      mu_bhutan = sample mean of the bhutanResample
      mu_nepal = sample mean of the nepalResample
      mean_difference = |muNepal - muBhutan|
      if mean_difference > observed_difference:
         count += 1
   pvalue = count / 10,000
```

이 방법의 장점은 샘플이 나온 모수적 분포에 대한 가정을 할 필요가 없다는 것입니다. t-검정과 같은 다른 p-값 계산 방법은 샘플이 가우시안 분포를 따르며 같은 분산을 가진다는 가정을 합니다. 현대의 컴퓨터 파워를 고려할 때, 부트스트랩은 더 정확하고 다재다능한 도구입니다.