# 05. PySpark - Apache Spark를 활용한 대규모 분산 처리

## 📚 개요

PySpark는 Apache Spark의 Python API로, 대규모 데이터를 효율적으로 처리하기 위한 분산 컴퓨팅 프레임워크입니다. 이 실습에서는 PySpark의 핵심 개념과 실제 활용법을 학습합니다.

### 🎯 학습 목표
- Apache Spark의 기본 개념과 아키텍처 이해
- RDD(Resilient Distributed Dataset)의 활용법
- Spark의 Transformations과 Actions 이해
- 실제 빅데이터 처리 작업에 PySpark 적용

## 🚀 Apache Spark 소개

![Logo](images/apache_spark_logo.png)

### 📋 **Spark의 역사**
- **2014년 첫 릴리즈**: Apache Spark 공식 출시
- **창시자**: [Matei Zaharia](http://people.csail.mit.edu/matei)의 수업과제로 시작
- **학술적 배경**: 박사학위 논문으로 제시된 혁신적인 기술
- **개발 언어**: [Scala](https://www.scala-lang.org)로 작성되어 고성능과 안정성 확보
- **이미지 크레딧**: [Databricks](https://databricks.com/product/getting-started-guide)

### 🔧 **Spark의 핵심 특징**

#### 🎯 **빠르고 범용적인 클러스터 컴퓨팅**
- **고성능**: 메모리 기반 처리로 Hadoop MapReduce보다 10-100배 빠름
- **범용성**: 다양한 데이터 처리 작업에 활용 가능
- **확장성**: 수백 대의 서버로 확장 가능한 분산 처리

#### 🌐 **다양한 언어 지원**
- **Java, Scala, Python**: 고수준 API 제공
- **최적화된 엔진**: 일반적인 실행 그래프를 지원하는 효율적인 엔진
- **통합 환경**: 하나의 프레임워크에서 다양한 작업 처리

#### 📊 **빅데이터 처리 연산자**
- **기본 연산**: `map`, `filter`, `groupby`, `join`
- **복잡한 계산**: 구조화된 패턴으로 복잡한 데이터 처리
- **최적화**: 자동으로 최적화되는 분산 처리

### 🛠️ **Spark 생태계**

#### 📋 **핵심 컴포넌트**
- **[Spark SQL](https://spark.apache.org/docs/latest/sql-programming-guide.html)**: 구조화된 데이터 처리를 위한 SQL API
- **[MLlib](https://spark.apache.org/docs/latest/ml-guide.html)**: 머신러닝을 위한 라이브러리
- **[GraphX](https://spark.apache.org/docs/latest/graphx-programming-guide.html)**: 그래프 처리를 위한 라이브러리
- **Structured Streaming**: 실시간 스트림 데이터 처리

## 🔄 RDD (Resilient Distributed Dataset) - Spark의 핵심

RDD는 Apache Spark의 기본적인 데이터 추상화로, 대규모 분산 데이터 처리를 위한 핵심 개념입니다.

### 📚 **RDD의 핵심 특징**

#### 🎯 **RDD의 정의**
- **읽기 전용 (Read-only)**: 생성 후 수정할 수 없는 불변 데이터 구조
- **병렬 (Parallel)**: 여러 코어에서 동시에 처리 가능
- **분산 (Distributed)**: 클러스터의 여러 노드에 분산 저장
- **오류 감내 (Fault-tolerant)**: 노드 장애 시 자동으로 복구

#### 🔧 **RDD의 동작 원리**

##### 📋 **데이터 분산**
- **클러스터 분산**: 데이터가 클러스터 내 여러 노드에 분산 저장
- **자동 할당**: Spark 프레임워크가 자동으로 데이터와 작업을 노드에 할당
- **투명성**: 프로그래머는 분산 처리를 신경 쓸 필요 없음

##### ⚡ **병렬 처리**
- **함수 적용**: 콜렉션의 모든 요소에 함수를 병렬로 적용
- **새로운 RDD 생성**: 변환 작업을 통해 새로운 RDD 생성
- **최적화**: Spark가 자동으로 최적의 실행 계획 수립

##### 🛡️ **오류 복구**
- **자동 재생성**: 노드 장애 시 RDD를 자동으로 재생성
- **데이터 복제**: 중요한 데이터를 여러 노드에 복제하여 안정성 확보
- **체크포인트**: 주기적으로 중간 결과를 저장하여 복구 시간 단축

### 💡 **RDD의 장점**
- **단순성**: 복잡한 분산 처리를 간단한 API로 처리
- **효율성**: 메모리 기반 처리로 빠른 성능
- **안정성**: 자동 오류 복구로 안정적인 처리
- **확장성**: 수백 대의 서버로 확장 가능

## 🔄 Spark 프로그램 생명주기

Spark 프로그램은 체계적인 단계를 거쳐 데이터를 처리합니다. 각 단계는 최적화와 성능 향상을 위해 설계되었습니다.

### 📋 **4단계 생명주기**

#### 1️⃣ **데이터 입력 (Data Input)**
- **외부 데이터**: 파일, 데이터베이스, 스트림에서 RDD 생성
- **메모리 데이터**: 드라이버 프로그램의 컬렉션을 병렬화
- **분산 저장**: 데이터를 클러스터의 여러 노드에 분산

#### 2️⃣ **데이터 변환 (Transformations)**
- **변환 함수**: `filter()`, `map()`, `flatMap()` 등 사용
- **새로운 RDD**: 기존 RDD로부터 새로운 RDD 정의
- **지연 실행**: 실제 계산은 Action이 호출될 때까지 지연

#### 3️⃣ **캐싱 (Caching)**
- **중간 결과 저장**: 재사용될 RDD를 메모리에 캐시
- **성능 최적화**: 반복 계산을 피하여 처리 속도 향상
- **메모리 관리**: 중요한 데이터만 선택적으로 캐시

#### 4️⃣ **액션 실행 (Actions)**
- **계산 실행**: `count()`, `collect()`, `saveAsTextFile()` 등 실행
- **최적화**: Spark가 자동으로 최적의 실행 계획 수립
- **결과 반환**: 최종 결과를 드라이버 프로그램에 반환

### 💡 **생명주기의 장점**
- **지연 실행**: 불필요한 계산을 피하여 효율성 증대
- **자동 최적화**: Spark가 자동으로 최적의 실행 계획 수립
- **메모리 효율성**: 필요한 데이터만 메모리에 유지
- **오류 복구**: 각 단계에서 오류 발생 시 자동으로 복구

## ⚡ 분산 데이터 연산

Spark에서 데이터 처리는 두 가지 주요 연산 유형으로 구분됩니다. 각각의 특징과 사용법을 이해하는 것이 중요합니다.

### 📋 **연산의 두 가지 유형**

#### 🔄 **Transformations (변환)**
- **특징**: 데이터를 변환하지만 즉시 실행되지 않음
- **지연 실행 (Lazy)**: Action이 호출될 때까지 실제 계산을 지연
- **새로운 RDD 생성**: 기존 RDD로부터 새로운 RDD를 정의
- **최적화**: 여러 변환을 하나의 작업으로 최적화 가능

#### 🎯 **Actions (액션)**
- **특징**: 실제 계산을 수행하고 결과를 반환
- **즉시 실행**: 호출되는 순간 계산이 시작됨
- **결과 반환**: 드라이버 프로그램에 최종 결과 전달
- **트리거**: Transformations의 실행을 트리거하는 역할

### 💡 **Lazy Evaluation의 장점**
- **최적화**: 불필요한 중간 계산을 피하여 효율성 증대
- **메모리 절약**: 필요한 데이터만 메모리에 유지
- **성능 향상**: 여러 변환을 하나의 작업으로 통합하여 성능 향상
- **유연성**: 복잡한 변환 체인을 쉽게 구성 가능

## 🔄 Transformations - 데이터 변환 함수

Transformations는 RDD를 변환하여 새로운 RDD를 생성하는 함수들입니다. 모든 변환은 지연 실행(lazy)되므로 실제 계산은 Action이 호출될 때까지 수행되지 않습니다.

### 📋 **주요 Transformation 함수들**

#### 🎯 **기본 변환 함수**
- **`map()`**: 각 요소에 함수를 적용하여 새로운 RDD 생성
- **`flatMap()`**: 각 요소를 여러 요소로 확장하여 평면화
- **`filter()`**: 조건을 만족하는 요소만 필터링

#### 🔧 **고급 변환 함수**
- **`mapPartitions()`**: 각 파티션에 함수를 적용
- **`mapPartitionsWithIndex()`**: 파티션 인덱스와 함께 함수 적용
- **`sample()`**: 데이터 샘플링

#### 📊 **집계 및 정렬 함수**
- **`groupBy()`**: 키를 기준으로 그룹화
- **`groupByKey()`**: 키-값 쌍에서 키 기준 그룹화
- **`reduceByKey()`**: 키별로 값들을 집계
- **`sortBy()`**: 지정된 키로 정렬
- **`sortByKey()`**: 키 기준 정렬

#### 🔗 **결합 함수**
- **`union()`**: 두 RDD를 결합
- **`intersection()`**: 교집합 계산
- **`distinct()`**: 중복 제거
- **`join()`**: 두 RDD를 조인

### 💡 **Transformation의 특징**
- **지연 실행**: Action이 호출될 때까지 실제 계산 지연
- **불변성**: 기존 RDD를 변경하지 않고 새로운 RDD 생성
- **최적화**: 여러 변환을 하나의 작업으로 최적화
- **분산 처리**: 클러스터의 여러 노드에서 병렬 실행
#### 🔧 **추가 Transformation 함수들**
- **`cogroup()`**: 여러 RDD를 키별로 그룹화
- **`cartesian()`**: 두 RDD의 카르테시안 곱 계산
- **`pipe()`**: 외부 프로그램과 데이터 파이프라인 연결
- **`coalesce()`**: 파티션 수를 줄여서 최적화
- **`repartition()`**: 파티션을 재분배하여 균형 조정
- **`partitionBy()`**: 파티션 함수를 사용하여 데이터 재분배
- **기타 함수들**: 다양한 특수 목적 변환 함수들

## 🎯 Actions - 실제 계산 실행

Actions는 RDD에서 실제 계산을 수행하고 결과를 반환하는 함수들입니다. Action이 호출되면 모든 이전 Transformations이 실행됩니다.
### 📋 **주요 Action 함수들**

#### 🎯 **기본 집계 함수**
- **`reduce()`**: 모든 요소를 하나의 값으로 집계
- **`count()`**: RDD의 요소 개수 반환
- **`first()`**: 첫 번째 요소 반환
- **`take(n)`**: 처음 n개 요소 반환

#### 📊 **샘플링 및 정렬 함수**
- **`takeSample()`**: 무작위 샘플링
- **`takeOrdered()`**: 정렬된 순서로 요소 반환
- **`countByKey()`**: 키별 요소 개수 계산

#### 💾 **저장 함수**
- **`saveAsTextFile()`**: 텍스트 파일로 저장
- **`saveAsSequenceFile()`**: 시퀀스 파일로 저장
- **`saveAsObjectFile()`**: 객체 파일로 저장
- **`saveToCassandra()`**: Cassandra 데이터베이스에 저장

#### 🔄 **반복 및 수집 함수**
- **`collect()`**: 모든 요소를 드라이버에 수집
- **`foreach()`**: 각 요소에 함수를 적용

### 💡 **Action의 특징**
- **즉시 실행**: 호출되는 순간 계산 시작
- **결과 반환**: 드라이버 프로그램에 최종 결과 전달
- **트리거**: 모든 이전 Transformations 실행
- **최적화**: Spark가 자동으로 최적의 실행 계획 수립

## 🐍 Python API - PySpark의 핵심

PySpark는 Python에서 Apache Spark를 사용할 수 있게 해주는 API로, Python의 간결함과 Spark의 강력한 분산 처리 능력을 결합합니다.

### 📚 **PySpark의 기술적 특징**

#### 🔧 **Py4J 기반 아키텍처**
- **Py4J 활용**: Java 객체에 동적으로 접근할 수 있는 Python 프로그램 구현
- **JVM 연동**: Python과 Java Virtual Machine 간의 효율적인 통신
- **동적 바인딩**: 런타임에 Java 객체와 메소드에 접근

#### 🎯 **PySpark의 장점**
- **Python 친화적**: Python 개발자에게 익숙한 API 제공
- **풍부한 생태계**: NumPy, Pandas 등 Python 라이브러리와 연동
- **간결한 코드**: 복잡한 분산 처리를 간단한 코드로 구현
- **빠른 프로토타이핑**: 데이터 과학자들이 빠르게 실험 가능

### 💡 **PySpark의 활용 분야**
- **데이터 분석**: 대규모 데이터셋의 탐색적 분석
- **머신러닝**: MLlib를 활용한 분산 머신러닝
- **스트림 처리**: 실시간 데이터 스트림 처리
- **ETL 파이프라인**: 데이터 추출, 변환, 로드 작업

![PySpark Internals](images/YlI8AqEl.png)

## 🔧 SparkContext 클래스 - Spark의 핵심

SparkContext는 Spark 애플리케이션의 진입점으로, 클러스터와의 연결을 관리하고 RDD를 생성하는 핵심 클래스입니다.

### 📋 **SparkContext의 역할**

#### 🎯 **핵심 기능**
- **클러스터 연결**: Spark 클러스터와의 연결을 관리
- **RDD 생성**: 외부 데이터로부터 RDD를 생성
- **리소스 관리**: 메모리, CPU 등 클러스터 리소스 할당
- **작업 스케줄링**: 작업을 클러스터의 여러 노드에 분배

#### 🔧 **사용 방법**
- **인스턴스 생성**: `pyspark.SparkContext` 객체 생성
- **메소드 호출**: SparkContext 인스턴스에서 메소드 호출
- **자동 할당**: 일반적으로 `sc` 변수에 자동으로 할당
- **생명주기**: 애플리케이션 시작부터 종료까지 유지

### 💡 **SparkContext의 중요성**
- **진입점**: 모든 Spark 작업의 시작점
- **리소스 관리**: 클러스터 리소스의 효율적 활용
- **작업 조율**: 분산 작업의 조율과 관리
- **성능 최적화**: 자동으로 최적의 실행 계획 수립

### 🔧 **RDD 생성 방법**

#### 📋 **parallelize 메소드**
- **기능**: Python 컬렉션을 RDD로 변환
- **용도**: 작은 데이터셋을 분산 처리하기 위해 사용
- **제한**: 메모리에 모든 데이터를 로드해야 하므로 대용량 데이터에는 부적합

#### 🌐 **실제 데이터 소스**
- **대용량 파일**: HDFS, S3 등에서 대용량 파일 읽기
- **데이터베이스**: HBase, Cassandra 등 NoSQL 데이터베이스
- **스트림**: 실시간 데이터 스트림 처리
- **클라우드**: AWS, Azure 등 클라우드 스토리지

## 🚀 첫 번째 예제 - 기본 RDD 연산

### 📋 **PySpark 설정 및 환경 구성**

#### 🔧 **라이브러리 경로 문제**
- **문제**: PySpark가 기본적으로 `sys.path`에 없어서 라이브러리를 찾을 수 없음
- **해결책**: `site-package`에 pyspark를 심볼릭 링크하거나 런타임에 `sys.path`에 추가
- **추천 도구**: [findspark](https://github.com/minrk/findspark) 라이브러리 활용

#### 🚀 **기본 설정 코드**

```python
import os, sys
sys.executable

~~~

```python
# Spark 홈 디렉토리 설정 (필요시)
#os.environ["SPARK_HOME"] = "/opt/spark-3.0.1-bin-hadoop2.7"
os.environ["PYSPARK_PYTHON"] = sys.executable
```

```python
import pyspark

# SparkContext 생성
sc = pyspark.SparkContext(master="local[*]", appName="FirstExample")
sc.setLogLevel("ERROR")  # 로그 레벨을 ERROR로 설정하여 불필요한 출력 제거
```

```python
print(sc)  # SparkContext 객체 확인 (Pool Processor executor와 유사)
```

## 🔄 첫 번째 RDD 생성

### 📋 **RDD 생성 과정**

```python
# 데이터 준비
data = list(range(8))
print("원본 데이터:", data)

# RDD 생성
rdd = sc.parallelize(data)  # Python 컬렉션을 RDD로 변환
print("RDD 객체:", rdd)
```

### 💡 **RDD 생성의 특징**
- **분산 저장**: 데이터가 클러스터의 여러 노드에 분산 저장
- **병렬 처리**: 각 요소를 독립적으로 병렬 처리 가능
- **지연 실행**: 실제 계산은 Action이 호출될 때까지 지연
- **최적화**: Spark가 자동으로 최적의 파티션 수 결정

## 🎯 연습문제 - 기본 RDD 연산

### 📋 **연습문제 5.1: RDD 기본 연산**

**목표**: 생성된 RDD에 대해 기본적인 Transformation과 Action 연산을 수행해보기

#### 🔧 **요구사항**
1. **map() 함수**: 각 요소에 제곱 연산 적용
2. **filter() 함수**: 짝수만 필터링
3. **collect() 함수**: 결과를 드라이버에 수집
4. **count() 함수**: 요소 개수 계산

#### 💡 **구현 힌트**
```python
# 1. map() - 각 요소에 제곱 연산 적용
squared_rdd = rdd.map(lambda x: x ** 2)
print("제곱 결과:", squared_rdd.collect())

# 2. filter() - 짝수만 필터링
even_rdd = rdd.filter(lambda x: x % 2 == 0)
print("짝수만:", even_rdd.collect())

# 3. count() - 요소 개수 계산
print("총 요소 개수:", rdd.count())
print("짝수 개수:", even_rdd.count())
```

### 🎯 **예상 결과**
- **제곱 결과**: [0, 1, 4, 9, 16, 25, 36, 49]
- **짝수만**: [0, 2, 4, 6]
- **총 요소 개수**: 8
- **짝수 개수**: 4

### 💡 **학습 포인트**
- **지연 실행**: Transformation은 Action이 호출될 때까지 실행되지 않음
- **분산 처리**: 각 연산이 클러스터의 여러 노드에서 병렬로 실행
- **메모리 효율성**: 필요한 데이터만 메모리에 유지
- **자동 최적화**: Spark가 자동으로 최적의 실행 계획 수립

## 📝 연습문제 5.2: 텍스트 파일 처리

### 📋 **요구사항**
- **faker 패키지**: 임의 텍스트를 생성하는 라이브러리
- **파일 생성**: `sample.txt` 파일을 생성
- **RDD 로드**: `textFile` 함수를 사용하여 파일을 RDD로 로드

### 🔧 **faker 패키지 활용**

#### 📚 **faker의 기능**
- **임의 텍스트**: 다양한 형태의 가짜 데이터 생성
- **다양한 데이터**: 이름, 주소, 프로필 등 생성 가능
- **다국어 지원**: 한국어 데이터 생성 시 `ko_KR` 매개변수 사용

#### 💡 **구현 방법**
```python
from faker import Faker

# 한국어 faker 인스턴스 생성
fake = Faker('ko_KR')

# 샘플 텍스트 파일 생성
with open('sample.txt', 'w', encoding='utf-8') as f:
    for _ in range(100):
        f.write(fake.text() + '\n')

# Spark로 텍스트 파일 읽기
text_rdd = sc.textFile('sample.txt')
print("파일에서 읽은 라인 수:", text_rdd.count())
```

### 🎯 **학습 목표**
- **텍스트 처리**: 대용량 텍스트 파일의 분산 처리
- **RDD 생성**: 외부 파일로부터 RDD 생성
- **데이터 탐색**: 텍스트 데이터의 기본 통계 확인

```python
from faker import Faker
fake = Faker()
Faker.seed(0)  # 재현 가능한 결과를 위한 시드 설정

# 샘플 텍스트 파일 생성
with open("sample.txt","w") as f:
    f.write(fake.text(max_nb_chars=1000))
    
# Spark로 텍스트 파일 읽기
rdd = sc.textFile("sample.txt")
```

## 📥 Collect - 데이터 수집

### 📋 **Collect의 특징**
- **Action**: 실제 계산을 수행하는 액션 함수
- **드라이버 반환**: RDD의 모든 요소를 드라이버에 단일 리스트로 반환
- **메모리 주의**: 대용량 데이터의 경우 메모리 부족 위험

![Collect 동작 원리](images/DUO6ygB.png)

*출처: https://i.imgur.com/DUO6ygB.png*

### 🎯 **연습문제 5.3: Collect 연산**

**목표**: `sample.txt` 파일에서 읽은 텍스트를 collect() 함수로 수집하기

#### 💡 **구현 방법**
```python
# 텍스트 파일의 모든 라인을 드라이버에 수집
collected_data = rdd.collect()
print("수집된 데이터:", collected_data)
print("총 라인 수:", len(collected_data))
```

### ⚠️ **주의사항**
- **메모리 사용량**: 모든 데이터가 드라이버 메모리에 로드됨
- **네트워크 트래픽**: 클러스터의 모든 노드에서 드라이버로 데이터 전송
- **대용량 데이터**: 큰 데이터셋의 경우 `take()` 함수 사용 권장

## 🗺️ Map - 데이터 변환

### 📋 **Map의 특징**
- **Transformation**: 데이터를 변환하지만 즉시 실행되지 않음
- **Narrow 연산**: 각 파티션 내에서 독립적으로 실행
- **새로운 RDD**: 기존 RDD의 각 요소에 함수를 적용하여 새로운 RDD 생성
- **지연 실행**: Action이 호출될 때까지 실제 계산 지연

![Map 연산 동작 원리](images/PxNJf0U.png)

*출처: http://i.imgur.com/PxNJf0U.png*

### 💡 **Map 연산의 동작 원리**
- **1:1 매핑**: 각 입력 요소에 대해 정확히 하나의 출력 요소 생성
- **병렬 처리**: 각 파티션에서 독립적으로 함수 적용
- **데이터 변환**: 원본 데이터를 변환하여 새로운 데이터 생성
- **성능 최적화**: Narrow 연산으로 네트워크 통신 최소화

### 🎯 **Map 연산 예제**

```python
# RDD 생성
rdd = sc.parallelize(list(range(8)))
print("원본 데이터:", rdd.collect())

# Map 연산: 각 요소에 제곱 연산 적용
squared_rdd = rdd.map(lambda x: x ** 2)
print("제곱 결과:", squared_rdd.collect())
```

### 💡 **Map 연산의 활용**
- **데이터 변환**: 각 요소에 수학적 연산 적용
- **타입 변환**: 문자열을 숫자로 변환
- **필드 추출**: 복잡한 객체에서 특정 필드 추출
- **데이터 정리**: 불필요한 문자 제거, 형식 통일

## 🎯 연습문제 5.4: Map 연산 활용

### 📋 **연습문제 5.4: 텍스트 데이터 처리**

**목표**: `sample.txt` 파일의 텍스트 데이터에 대해 Map 연산을 활용하여 데이터 처리하기

#### 🔧 **요구사항**
1. **텍스트 길이 계산**: 각 라인의 문자 수 계산
2. **단어 수 계산**: 각 라인의 단어 수 계산
3. **대문자 변환**: 모든 텍스트를 대문자로 변환
4. **결과 확인**: 처리된 결과를 확인

#### 💡 **구현 방법**
```python
# 1. 텍스트 길이 계산
line_lengths = rdd.map(lambda line: len(line))
print("각 라인의 길이:", line_lengths.collect())

# 2. 단어 수 계산
word_counts = rdd.map(lambda line: len(line.split()))
print("각 라인의 단어 수:", word_counts.collect())

# 3. 대문자 변환
uppercase_lines = rdd.map(lambda line: line.upper())
print("대문자 변환 결과:", uppercase_lines.collect())
```

### 🎯 **예상 결과**
- **라인 길이**: 각 텍스트 라인의 문자 수
- **단어 수**: 각 라인의 단어 개수
- **대문자**: 모든 텍스트가 대문자로 변환된 결과

### 💡 **학습 포인트**
- **Map의 활용**: 다양한 데이터 변환 작업에 Map 연산 사용
- **지연 실행**: Transformation은 Action이 호출될 때까지 실행되지 않음
- **병렬 처리**: 각 라인이 독립적으로 병렬 처리됨
- **성능 최적화**: Narrow 연산으로 효율적인 처리

### 🔧 **고급 연습: 병렬 처리 확인**

**목표**: `sleep(1)`을 포함한 함수로 Map 연산의 병렬 처리 확인

#### 💡 **구현 방법**
```python
import time

def slow_function(x):
    time.sleep(1)  # 1초 대기
    return x ** 2

# 병렬 처리 시간 측정
start_time = time.time()
result = rdd.map(slow_function).collect()
end_time = time.time()

print("결과:", result)
print("총 소요 시간:", end_time - start_time, "초")
```

### 💡 **병렬 처리 확인**
- **순차 처리**: 8개 요소 × 1초 = 8초 소요
- **병렬 처리**: 여러 코어에서 동시 실행으로 8초보다 훨씬 빠름
- **성능 향상**: CPU 코어 수에 비례한 성능 향상

## 🔍 Filter - 데이터 필터링

### 📋 **Filter의 특징**
- **Transformation**: 조건을 만족하는 요소만 필터링
- **Narrow 연산**: 각 파티션 내에서 독립적으로 실행
- **새로운 RDD**: 조건을 만족하는 요소들로 구성된 새로운 RDD 생성
- **지연 실행**: Action이 호출될 때까지 실제 계산 지연

### 💡 **Filter 연산의 동작 원리**
- **조건 검사**: 각 요소에 대해 조건(predicate) 함수 적용
- **선택적 포함**: 조건을 만족하는 요소만 새로운 RDD에 포함
- **병렬 처리**: 각 파티션에서 독립적으로 필터링 수행
- **성능 최적화**: Narrow 연산으로 네트워크 통신 최소화

![Filter 연산 동작 원리](images/GFyji4U.png)

*출처: http://i.imgur.com/GFyji4U.png*

### 🎯 **Filter 연산 예제**

```python
# 짝수만 선택
even_numbers = rdd.filter(lambda x: x % 2 == 0)
print("짝수만:", even_numbers.collect())

# 5보다 큰 수만 선택
large_numbers = rdd.filter(lambda x: x > 5)
print("5보다 큰 수:", large_numbers.collect())

# 3의 배수만 선택
multiples_of_3 = rdd.filter(lambda x: x % 3 == 0)
print("3의 배수:", multiples_of_3.collect())
```

### 💡 **Filter 연산의 활용**
- **데이터 정리**: 불필요한 데이터 제거
- **조건부 선택**: 특정 조건을 만족하는 데이터만 선택
- **데이터 분할**: 데이터를 여러 그룹으로 분할
- **품질 관리**: 데이터 품질 검사 및 필터링

## 🔄 FlatMap - 데이터 평면화

### 📋 **FlatMap의 특징**
- **Transformation**: 각 요소를 여러 요소로 확장하여 평면화
- **Narrow 연산**: 각 파티션 내에서 독립적으로 실행
- **새로운 RDD**: 확장된 요소들로 구성된 새로운 RDD 생성
- **지연 실행**: Action이 호출될 때까지 실제 계산 지연

### 💡 **FlatMap 연산의 동작 원리**
- **함수 적용**: 각 요소에 함수를 적용하여 리스트 생성
- **평면화**: 생성된 리스트들을 하나의 평면 리스트로 변환
- **병렬 처리**: 각 파티션에서 독립적으로 처리 수행
- **성능 최적화**: Narrow 연산으로 네트워크 통신 최소화

![FlatMap 연산 동작 원리](images/TsSUex8.png)

### 🎯 **FlatMap 연산 예제**

```python
# RDD 생성
rdd = sc.parallelize([1, 2, 3])
print("원본 데이터:", rdd.collect())

# FlatMap 연산: 각 요소를 3개로 확장
expanded_rdd = rdd.flatMap(lambda x: (x, x*100, 42))
print("확장된 데이터:", expanded_rdd.collect())
```

### 💡 **FlatMap vs Map의 차이점**
- **Map**: 1:1 매핑 (각 입력 → 1개 출력)
- **FlatMap**: 1:N 매핑 (각 입력 → 여러 출력, 평면화)
- **용도**: 중첩된 구조를 평면화할 때 유용

### 🎯 **FlatMap의 활용**
- **텍스트 처리**: 문장을 단어로 분할
- **데이터 확장**: 하나의 레코드를 여러 레코드로 분할
- **중첩 구조 처리**: 리스트의 리스트를 평면 리스트로 변환
- **데이터 정규화**: 복잡한 데이터 구조를 단순화

## 🎯 연습문제 5.5: FlatMap 연산 활용

### 📋 **연습문제 5.5: 텍스트 데이터 평면화**

**목표**: `sample.txt` 파일의 텍스트 데이터에 대해 FlatMap 연산을 활용하여 단어 단위로 분할하기

#### 🔧 **요구사항**
1. **단어 분할**: 각 라인을 단어로 분할
2. **평면화**: 모든 단어를 하나의 평면 리스트로 변환
3. **중복 제거**: 고유한 단어만 추출
4. **결과 확인**: 처리된 결과를 확인

#### 💡 **구현 방법**
```python
# 1. 단어로 분할 및 평면화
words_rdd = rdd.flatMap(lambda line: line.split())
print("모든 단어:", words_rdd.collect())

# 2. 고유한 단어만 추출
unique_words = words_rdd.distinct()
print("고유한 단어:", unique_words.collect())

# 3. 단어 개수 계산
word_count = words_rdd.count()
unique_count = unique_words.count()
print(f"총 단어 수: {word_count}")
print(f"고유 단어 수: {unique_count}")
```

### 🎯 **예상 결과**
- **모든 단어**: 텍스트에서 추출된 모든 단어의 리스트
- **고유한 단어**: 중복이 제거된 고유한 단어들
- **통계**: 총 단어 수와 고유 단어 수

### 💡 **학습 포인트**
- **FlatMap의 활용**: 텍스트 데이터를 단어 단위로 분할
- **데이터 평면화**: 중첩된 구조를 평면 구조로 변환
- **중복 제거**: `distinct()` 함수를 사용한 중복 제거
- **성능 최적화**: Narrow 연산으로 효율적인 처리

### 🔧 **고급 연습: 텍스트 정리**

**목표**: `sample.txt` 파일의 텍스트를 정리하여 단어로 분할하기

#### 💡 **구현 방법**
```python
# 텍스트 정리: 소문자 변환, 구두점 제거, 단어 분할
cleaned_words = rdd.flatMap(lambda line: 
    line.lower()
        .replace('.', '')
        .replace(',', '')
        .replace('!', '')
        .replace('?', '')
        .split()
)
print("정리된 단어:", cleaned_words.collect())
```

## 📊 GroupBy - 데이터 그룹화

### 📋 **GroupBy의 특징**
- **Transformation**: 데이터를 키별로 그룹화
- **Wide 연산**: 여러 파티션 간 데이터 이동 필요
- **새로운 RDD**: 키별로 그룹화된 데이터로 구성된 새로운 RDD 생성
- **지연 실행**: Action이 호출될 때까지 실제 계산 지연

### 💡 **GroupBy 연산의 동작 원리**
- **키 생성**: 사용자 정의 함수를 사용하여 각 요소에 대해 키 생성
- **그룹화**: 같은 키를 가진 요소들을 하나의 그룹으로 묶음
- **데이터 이동**: 여러 파티션 간 데이터 이동이 필요 (Wide 연산)
- **성능 고려**: 네트워크 통신이 많이 발생하므로 신중한 사용 필요

![GroupBy 연산 동작 원리](images/gdj0Ey8.png)

### 🎯 **GroupBy 연산 예제**

```python
# 이름 리스트 생성
names = sc.parallelize(['John', 'Fred', 'Anna', 'James'])
print("원본 이름:", names.collect())

# 첫 글자로 그룹화
grouped_names = names.groupBy(lambda w: w[0])
print("그룹화된 결과:", [(k, list(v)) for (k, v) in grouped_names.collect()])
```

### 💡 **GroupBy의 활용**
- **데이터 분류**: 특정 기준으로 데이터를 분류
- **통계 분석**: 그룹별 통계 계산
- **데이터 집계**: 그룹별로 데이터 집계
- **패턴 분석**: 그룹별 패턴 분석

### ⚠️ **GroupBy 사용 시 주의사항**
- **성능 영향**: Wide 연산으로 인한 네트워크 통신 증가
- **메모리 사용**: 그룹화된 데이터의 메모리 사용량 증가
- **대안**: `reduceByKey()` 함수 사용 권장 (더 효율적)

## 🔑 GroupByKey - 키별 데이터 그룹화

### 📋 **GroupByKey의 특징**
- **Transformation**: 키-값 쌍에서 키별로 그룹화
- **Wide 연산**: 여러 파티션 간 데이터 이동 필요
- **새로운 RDD**: 키별로 그룹화된 값들로 구성된 새로운 RDD 생성
- **지연 실행**: Action이 호출될 때까지 실제 계산 지연

### 💡 **GroupByKey 연산의 동작 원리**
- **키별 그룹화**: 같은 키를 가진 값들을 하나의 그룹으로 묶음
- **데이터 이동**: 여러 파티션 간 데이터 이동이 필요 (Wide 연산)
- **성능 고려**: 네트워크 통신이 많이 발생하므로 신중한 사용 필요
- **대안**: `reduceByKey()` 함수 사용 권장 (더 효율적)

![GroupByKey 연산 동작 원리](images/TlWRGr2.png)

### 🎯 **GroupByKey 연산 예제**

```python
# 키-값 쌍 데이터 생성
data = sc.parallelize([('B', 5), ('B', 4), ('A', 3), ('A', 2), ('A', 1)])
print("원본 데이터:", data.collect())

# 키별로 그룹화
grouped_data = data.groupByKey()
result = [(key, list(values)) for key, values in grouped_data.collect()]
print("그룹화된 결과:", result)
```

### 💡 **GroupByKey의 활용**
- **데이터 집계**: 키별로 데이터를 그룹화하여 집계
- **통계 분석**: 그룹별 통계 계산
- **데이터 분류**: 키별로 데이터를 분류
- **패턴 분석**: 그룹별 패턴 분석

### ⚠️ **GroupByKey 사용 시 주의사항**
- **성능 영향**: Wide 연산으로 인한 네트워크 통신 증가
- **메모리 사용**: 그룹화된 데이터의 메모리 사용량 증가
- **대안**: `reduceByKey()` 함수 사용 권장 (더 효율적)

## 🔗 Join - 데이터 조인

### 📋 **Join의 특징**
- **Transformation**: 두 RDD를 키를 기준으로 조인
- **Wide 연산**: 여러 파티션 간 데이터 이동 필요
- **새로운 RDD**: 조인된 데이터로 구성된 새로운 RDD 생성
- **지연 실행**: Action이 호출될 때까지 실제 계산 지연

### 💡 **Join 연산의 동작 원리**
- **키 매칭**: 두 RDD에서 같은 키를 가진 요소들을 찾아 조인
- **데이터 이동**: 여러 파티션 간 데이터 이동이 필요 (Wide 연산)
- **성능 고려**: 네트워크 통신이 많이 발생하므로 신중한 사용 필요
- **조인 유형**: Inner Join, Left Join, Right Join, Outer Join 지원

![Join 연산 동작 원리](images/YXL42Nl.png)

### 🎯 **Join 연산 예제**

```python
# 첫 번째 RDD 생성
x = sc.parallelize([("a", 1), ("b", 2)])
print("첫 번째 RDD:", x.collect())

# 두 번째 RDD 생성
y = sc.parallelize([("a", 3), ("a", 4), ("b", 5)])
print("두 번째 RDD:", y.collect())

# Inner Join 수행
joined = x.join(y)
print("조인 결과:", joined.collect())
```

### 💡 **Join의 활용**
- **데이터 결합**: 두 데이터셋을 키를 기준으로 결합
- **관계 분석**: 관련된 데이터 간의 관계 분석
- **데이터 보강**: 한 데이터셋에 다른 데이터셋의 정보 추가
- **통계 분석**: 결합된 데이터를 이용한 통계 분석

### ⚠️ **Join 사용 시 주의사항**
- **성능 영향**: Wide 연산으로 인한 네트워크 통신 증가
- **메모리 사용**: 조인된 데이터의 메모리 사용량 증가
- **키 분포**: 키의 분포가 균등하지 않으면 성능 저하
- **데이터 크기**: 큰 데이터셋의 조인은 시간이 오래 걸림

## 🔍 Distinct - 중복 제거

### 📋 **Distinct의 특징**
- **Transformation**: 중복된 요소를 제거하여 고유한 요소만 유지
- **Wide 연산**: 여러 파티션 간 데이터 이동 필요
- **새로운 RDD**: 중복이 제거된 데이터로 구성된 새로운 RDD 생성
- **지연 실행**: Action이 호출될 때까지 실제 계산 지연

### 💡 **Distinct 연산의 동작 원리**
- **중복 검사**: 모든 요소를 비교하여 중복된 요소를 찾음
- **데이터 이동**: 여러 파티션 간 데이터 이동이 필요 (Wide 연산)
- **성능 고려**: 네트워크 통신이 많이 발생하므로 신중한 사용 필요
- **메모리 사용**: 중복 제거를 위한 추가 메모리 사용

![Distinct 연산 동작 원리](images/Vqgy2a4.png)

### 🎯 **Distinct 연산 예제**

```python
# 중복이 포함된 데이터 생성
data = sc.parallelize([1, 2, 3, 3, 4])
print("원본 데이터:", data.collect())

# 중복 제거
unique_data = data.distinct()
print("중복 제거된 데이터:", unique_data.collect())
```

### 💡 **Distinct의 활용**
- **데이터 정리**: 중복된 데이터를 제거하여 데이터 품질 향상
- **고유값 분석**: 데이터의 고유한 값들만 분석
- **메모리 절약**: 중복 제거로 메모리 사용량 감소
- **성능 향상**: 중복이 제거된 데이터로 처리 속도 향상

### ⚠️ **Distinct 사용 시 주의사항**
- **성능 영향**: Wide 연산으로 인한 네트워크 통신 증가
- **메모리 사용**: 중복 제거를 위한 추가 메모리 사용
- **데이터 크기**: 큰 데이터셋의 중복 제거는 시간이 오래 걸림
- **대안**: 가능한 경우 `groupByKey()` 후 `mapValues()` 사용 권장

## 🔑 KeyBy - 키 생성

### 📋 **KeyBy의 특징**
- **Transformation**: 각 요소에 대해 키를 생성하여 키-값 쌍으로 변환
- **Narrow 연산**: 각 파티션 내에서 독립적으로 실행
- **새로운 RDD**: 키-값 쌍으로 구성된 새로운 RDD 생성
- **지연 실행**: Action이 호출될 때까지 실제 계산 지연

### 💡 **KeyBy 연산의 동작 원리**
- **키 생성**: 사용자 정의 함수를 사용하여 각 요소에 대해 키 생성
- **키-값 쌍**: 생성된 키와 원본 값을 키-값 쌍으로 변환
- **병렬 처리**: 각 파티션에서 독립적으로 처리 수행
- **성능 최적화**: Narrow 연산으로 네트워크 통신 최소화

![KeyBy 연산 동작 원리](images/nqYhDW5.png)

### 🎯 **KeyBy 연산 예제**

```python
# 이름 리스트 생성
names = sc.parallelize(['John', 'Fred', 'Anna', 'James'])
print("원본 이름:", names.collect())

# 첫 글자를 키로 하는 키-값 쌍 생성
keyed_names = names.keyBy(lambda w: w[0])
print("키-값 쌍:", keyed_names.collect())
```

### 💡 **KeyBy의 활용**
- **키 생성**: 데이터에 대해 의미 있는 키 생성
- **그룹화 준비**: 그룹화 작업을 위한 키-값 쌍 생성
- **조인 준비**: 조인 작업을 위한 키-값 쌍 생성
- **데이터 변환**: 단순한 데이터를 키-값 구조로 변환

### 🎯 **KeyBy의 장점**
- **유연성**: 사용자 정의 함수로 다양한 키 생성 가능
- **성능**: Narrow 연산으로 효율적인 처리
- **확장성**: 대용량 데이터에 대해서도 빠른 처리
- **호환성**: 다른 키 기반 연산과 쉽게 연동

## 🎯 Actions - 실제 계산 실행

### 📋 **Actions의 특징**
- **실제 계산**: Transformation의 지연 실행을 트리거하여 실제 계산 수행
- **결과 반환**: 드라이버 프로그램에 최종 결과 반환
- **즉시 실행**: 호출되는 순간 계산이 시작됨
- **최적화**: Spark가 자동으로 최적의 실행 계획 수립

### 🔄 Map-Reduce 연산

#### 📋 **Map-Reduce의 특징**
- **Action**: 실제 계산을 수행하는 액션 함수
- **드라이버 반환**: 모든 요소를 집계하여 드라이버에 단일 결과 반환
- **쌍별 연산**: 요소들을 쌍별로 연산하여 부분 결과 생성
- **최종 집계**: 부분 결과들을 최종적으로 집계

#### 💡 **Map-Reduce의 동작 원리**
- **Map 단계**: 각 요소에 함수를 적용하여 변환
- **Reduce 단계**: 변환된 결과들을 쌍별로 연산하여 집계
- **병렬 처리**: 여러 파티션에서 동시에 처리 수행
- **결과 반환**: 최종 집계 결과를 드라이버에 반환

![Map-Reduce 연산 동작 원리](images/R72uzwX.png)

### 🎯 **Map-Reduce 연산 예제**

```python
from operator import add

# RDD 생성
rdd = sc.parallelize(list(range(8)))
print("원본 데이터:", rdd.collect())

# Map-Reduce 연산: 각 요소를 제곱한 후 합계 계산
result = rdd.map(lambda x: x ** 2).reduce(add)
print("제곱의 합:", result)
```

### 💡 **Map-Reduce의 활용**
- **집계 연산**: 데이터의 합계, 평균, 최대값, 최소값 계산
- **통계 분석**: 데이터의 통계적 특성 분석
- **데이터 변환**: 데이터를 변환한 후 집계
- **성능 최적화**: 여러 연산을 하나의 작업으로 통합

### 🎯 **Map-Reduce의 장점**
- **효율성**: 여러 연산을 하나의 작업으로 통합하여 성능 향상
- **병렬 처리**: 여러 파티션에서 동시에 처리 수행
- **메모리 효율성**: 중간 결과를 메모리에 유지하지 않음
- **자동 최적화**: Spark가 자동으로 최적의 실행 계획 수립

### 📊 통계 함수들

#### 📋 **통계 함수의 특징**
- **Action**: 실제 계산을 수행하는 액션 함수
- **드라이버 반환**: 통계 결과를 드라이버에 반환
- **수치 연산**: 숫자형 RDD에 대해서만 사용 가능
- **즉시 실행**: 호출되는 순간 계산이 시작됨

#### 🔧 **주요 통계 함수들**
- **`max()`**: 최대값 계산
- **`min()`**: 최소값 계산
- **`sum()`**: 합계 계산
- **`mean()`**: 평균 계산
- **`variance()`**: 분산 계산
- **`stdev()`**: 표준편차 계산

![통계 함수 동작 원리](images/HUCtib1.png)

### 🔢 CountByKey - 키별 개수 계산

#### 📋 **CountByKey의 특징**
- **Action**: 실제 계산을 수행하는 액션 함수
- **드라이버 반환**: 키별 개수를 드라이버에 반환
- **키-값 쌍**: 키-값 쌍 RDD에 대해서만 사용 가능
- **즉시 실행**: 호출되는 순간 계산이 시작됨

#### 💡 **CountByKey의 동작 원리**
- **키별 집계**: 같은 키를 가진 요소들의 개수를 계산
- **결과 반환**: 키와 개수의 매핑을 드라이버에 반환
- **병렬 처리**: 여러 파티션에서 동시에 처리 수행
- **메모리 효율성**: 중간 결과를 메모리에 유지하지 않음

![CountByKey 연산 동작 원리](images/jvQTGv6.png)

### 🎯 **CountByKey 연산 예제**

```python
# 키-값 쌍 데이터 생성
data = sc.parallelize([('J', 'James'), ('F', 'Fred'), 
                      ('A', 'Anna'), ('J', 'John')])
print("원본 데이터:", data.collect())

# 키별 개수 계산
key_counts = data.countByKey()
print("키별 개수:", key_counts)
```

### 💡 **CountByKey의 활용**
- **빈도 분석**: 각 키의 출현 빈도 분석
- **데이터 분포**: 키별 데이터 분포 확인
- **통계 분석**: 키별 통계적 특성 분석
- **데이터 검증**: 키별 데이터의 일관성 검증

### 🎯 **CountByKey의 장점**
- **효율성**: 키별 개수를 한 번에 계산
- **병렬 처리**: 여러 파티션에서 동시에 처리 수행
- **메모리 효율성**: 중간 결과를 메모리에 유지하지 않음
- **자동 최적화**: Spark가 자동으로 최적의 실행 계획 수립

# 키별 개수 계산
key_counts = rdd.countByKey()
print("키별 개수:", key_counts)
```

## 📚 학습 요약

### 🎯 **PySpark 핵심 개념**
- **RDD**: 분산 데이터 처리를 위한 기본 추상화
- **Transformations**: 데이터 변환 (지연 실행)
- **Actions**: 실제 계산 실행 (즉시 실행)
- **Lazy Evaluation**: 지연 실행을 통한 성능 최적화

### 🔧 **주요 연산들**
- **Narrow 연산**: `map()`, `filter()`, `flatMap()`, `keyBy()`
- **Wide 연산**: `groupBy()`, `groupByKey()`, `join()`, `distinct()`
- **Actions**: `collect()`, `count()`, `reduce()`, `countByKey()`

### 💡 **성능 최적화**
- **Narrow vs Wide**: 네트워크 통신 최소화
- **Lazy Evaluation**: 불필요한 계산 방지
- **자동 최적화**: Spark의 자동 실행 계획 수립
- **메모리 효율성**: 필요한 데이터만 메모리에 유지

### 🎯 **다음 단계**
- **Spark DataFrames**: 구조화된 데이터 처리
- **Pandas**: 데이터 분석을 위한 고급 기능
- **실제 프로젝트**: 대용량 데이터 처리 프로젝트 적용

### 🔚 **SparkContext 종료**

```python
# 로컬 Spark 클러스터 종료
sc.stop()
```

### ⚠️ **중요 사항**
- **리소스 해제**: SparkContext를 사용한 후 반드시 `stop()` 호출
- **메모리 정리**: 사용한 메모리와 리소스를 정리
- **클러스터 종료**: 로컬 클러스터를 안전하게 종료
- **에러 방지**: 리소스 누수 방지 및 안정적인 프로그램 종료

## 🎯 최종 연습문제: Apache Spark WordCount

### 📋 **연습문제 5.6: Spark WordCount 구현**

**목표**: Apache Spark를 사용하여 완전한 WordCount 프로그램을 구현하기

#### 🔧 **요구사항**
1. **텍스트 파일 읽기**: `sample.txt` 파일을 RDD로 로드
2. **단어 분할**: 각 라인을 단어로 분할
3. **단어 정리**: 소문자 변환, 구두점 제거
4. **단어 빈도 계산**: 각 단어의 출현 빈도 계산
5. **결과 정렬**: 빈도순으로 정렬하여 상위 10개 단어 출력

#### 💡 **구현 방법**
```python
# 1. 텍스트 파일 읽기
text_rdd = sc.textFile("sample.txt")

# 2. 단어 분할 및 정리
words_rdd = text_rdd.flatMap(lambda line: 
    line.lower()
        .replace('.', '')
        .replace(',', '')
        .replace('!', '')
        .replace('?', '')
        .split()
)

# 3. 단어 빈도 계산
word_counts = words_rdd.map(lambda word: (word, 1)) \
                      .reduceByKey(lambda a, b: a + b)

# 4. 빈도순 정렬 및 상위 10개 출력
top_words = word_counts.sortBy(lambda x: x[1], ascending=False) \
                      .take(10)

print("상위 10개 단어:")
for word, count in top_words:
    print(f"{word}: {count}")
```

### 🎯 **예상 결과**
- **단어 빈도**: 각 단어의 출현 빈도
- **정렬된 결과**: 빈도순으로 정렬된 단어 목록
- **상위 10개**: 가장 많이 출현한 10개 단어

### 💡 **학습 포인트**
- **완전한 파이프라인**: 텍스트 파일부터 결과 출력까지
- **Spark 연산 활용**: `flatMap()`, `map()`, `reduceByKey()`, `sortBy()`
- **성능 최적화**: Narrow 연산과 Wide 연산의 적절한 활용
- **실제 응용**: 대용량 텍스트 데이터 처리의 실제 사례

- Write the sample text file

- Create the rdd with `SparkContext.textFile method`
- lower, remove dots and split using `rdd.flatMap`
- use `rdd.map` to create the list of key/value pair (word, 1)
- `rdd.reduceByKey` to get all occurences
- `rdd.takeOrdered`to get sorted frequencies of words

All documentation is available [here](https://spark.apache.org/docs/2.1.0/api/python/pyspark.html?highlight=textfile#pyspark.SparkContext) for textFile and [here](https://spark.apache.org/docs/2.1.0/api/python/pyspark.html?highlight=textfile#pyspark.RDD) for RDD. 

For a global overview see the Transformations section of the [programming guide](https://spark.apache.org/docs/latest/rdd-programming-guide.html)


## SparkSession

Since SPARK 2.0.0,  SparkSession provides a single point 
of entry to interact with Spark functionality and
allows programming Spark with DataFrame and Dataset APIs. 

###  $\pi$ computation example

- We can estimate an approximate value for $\pi$ using the following Monte-Carlo method:

1.    Inscribe a circle in a square
2.    Randomly generate points in the square
3.    Determine the number of points in the square that are also in the circle
4.    Let $r$ be the number of points in the circle divided by the number of points in the square, then $\pi \approx 4 r$.
    
- Note that the more points generated, the better the approximation

See [this tutorial](https://computing.llnl.gov/tutorials/parallel_comp/#ExamplesPI).


### Exercise 9.2

Using the same method than the PI computation example, compute the integral
$$
I = \int_0^1 \exp(-x^2) dx
$$
You can check your result with numpy

~~~python
# numpy evaluates solution using numeric computation. 
# It uses discrete values of the function
import numpy as np
x = np.linspace(0,1,1000)
np.trapz(np.exp(-x*x),x)

~~~

numpy and scipy evaluates solution using numeric computation. It uses discrete values of the function

~~~python
import numpy as np
from scipy.integrate import quad
quad(lambda x: np.exp(-x*x), 0, 1)
# note: the solution returned is complex 

~~~

### Correlation between daily stock

- Data preparation

~~~python
import os  # library to get directory and file paths
import tarfile # this module makes possible to read and write tar archives

def extract_data(name, where):
    datadir = os.path.join(where,name)
    if not os.path.exists(datadir):
       print("Extracting data...")
       tar_path = os.path.join(where, name+'.tgz')
       with tarfile.open(tar_path, mode='r:gz') as data:
          data.extractall(where)
            
extract_data('daily-stock','data') # this function call will extract json files

~~~

~~~python
import json
import pandas as pd
import os, glob

here = os.getcwd()
datadir = os.path.join(here,'data','daily-stock')
filenames = sorted(glob.glob(os.path.join(datadir, '*.json')))
filenames

~~~

~~~python
%rm data/daily-stock/*.h5

~~~

~~~python
from glob import glob
import os, json
import pandas as pd

for fn in filenames:
    with open(fn) as f:
        data = [json.loads(line) for line in f]
        
    df = pd.DataFrame(data)
    
    out_filename = fn[:-5] + '.h5'
    df.to_hdf(out_filename, '/data')
    print("Finished : %s" % out_filename.split(os.path.sep)[-1])

filenames = sorted(glob(os.path.join('data', 'daily-stock', '*.h5')))  # data/json/*.json

~~~

### Sequential code

~~~python
filenames

~~~

~~~python
with pd.HDFStore('data/daily-stock/aet.h5') as hdf:
    # This prints a list of all group names:
    print(hdf.keys())

~~~

~~~python
df_test = pd.read_hdf('data/daily-stock/aet.h5')

~~~

~~~python
%%time

series = []
for fn in filenames:   # Simple map over filenames
    series.append(pd.read_hdf(fn)["close"])

results = []

for a in series:    # Doubly nested loop over the same collection
    for b in series:  
        if not (a == b).all():     # Filter out comparisons of the same series 
            results.append(a.corr(b))  # Apply function

result = max(results)
result

~~~

### Exercise 9.3

Parallelize the code above with Apache Spark.

- Change the filenames because of the Hadoop environment.

~~~python
import os, glob

here = os.getcwd()
filenames = sorted(glob.glob(os.path.join(here,'data', 'daily-stock', '*.h5')))
filenames

~~~

If it is not started don't forget the PySpark context

Computation time is slower because there is a lot of setup, workers creation, there is a lot of communications the correlation function is too small

### Exercise 9.4 Fasta file example

Use a RDD to calculate the GC content of fasta file nucleotide-sample.txt:

$$\frac{G+C}{A+T+G+C}\times100 \% $$

Create a rdd from fasta file genome.txt in data directory and count 'G' and 'C' then divide by the total number of bases.

### Another example

Compute the most frequent sequence with 5 bases.
