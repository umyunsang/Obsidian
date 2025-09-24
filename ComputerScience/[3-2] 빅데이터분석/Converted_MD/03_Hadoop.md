# 03. Hadoop - 빅데이터 분산 처리의 핵심

## 📚 개요

Hadoop은 대용량 데이터를 분산 환경에서 처리하기 위한 오픈소스 프레임워크입니다. 이 실습에서는 Hadoop의 핵심 구성요소와 실제 사용법을 학습합니다.

### 🎯 학습 목표
- Hadoop의 핵심 개념과 아키텍처 이해
- HDFS (Hadoop Distributed File System) 활용법
- YARN (Yet Another Resource Negotiator) 이해
- 실제 Hadoop 클러스터에서 작업 실행

### 🎯 Hadoop이 필요한 경우
- **대용량 데이터**: 매우 크거나 복잡한 데이터셋 처리
- **분산 처리**: 여러 서버에서 병렬 소프트웨어 실행
- **빅데이터 4V**: Volume, Variety, Velocity, Variability, Veracity 속성 고려

![빅데이터 개념도](https://github.com/pnavaro/big-data/blob/master/notebooks/images/bigdata.png?raw=1)

![Hadoop 로고](https://github.com/pnavaro/big-data/blob/master/notebooks/images/hadoop.png?raw=1)

## 🏗️ Hadoop 아키텍처

### 📋 핵심 구성요소

#### 1️⃣ **Hadoop 프레임워크**
- **정의**: 대형 클러스터에서 애플리케이션을 실행하는 프레임워크
- **특징**: 투명한 신뢰성과 데이터 이동 제공
- **장점**: 자동화된 노드 장애 처리

#### 2️⃣ **MapReduce 패러다임**
- **작업 분할**: 작업을 작은 단위(fragment)로 분해
- **분산 실행**: 클러스터 내 여러 노드에서 병렬 처리
- **자동 재실행**: 실패한 작업의 자동 재시도

#### 3️⃣ **HDFS (Hadoop Distributed File System)**
- **역할**: 컴퓨팅 노드에서 데이터 저장
- **장점**: 클러스터 간 높은 집계 대역폭 제공
- **내결함성**: 노드 장애 시 자동 복구

### 💡 Hadoop의 핵심 가치
- **확장성**: 수천 개의 노드로 확장 가능
- **내결함성**: 하드웨어 장애에 대한 자동 복구
- **비용 효율성**: 일반적인 하드웨어로 구성 가능
- **유연성**: 다양한 데이터 타입과 처리 방식 지원

## 🗂️ HDFS (Hadoop Distributed File System)

HDFS는 대용량 데이터를 분산 환경에서 안전하게 저장하고 관리하는 분산 파일 시스템입니다.

### 📚 HDFS의 핵심 특징

#### 🎯 **분산 파일 시스템**
- **정의**: 여러 노드에 걸쳐 데이터를 분산 저장하는 파일 시스템
- **목적**: 대용량 데이터의 안전한 저장과 빠른 접근

#### 🛡️ **높은 내결함성**
- **특징**: 하드웨어 장애에 대한 강력한 복구 능력
- **설계**: 저비용 하드웨어에서도 안정적으로 동작
- **복제**: 데이터의 다중 복사본으로 안전성 보장

#### 📊 **대용량 데이터 최적화**
- **적합성**: 대용량 데이터셋을 위한 설계
- **효율성**: 데이터가 있는 곳에서 계산 수행 (Data Locality)
- **성능**: 대용량 데이터 처리 시 높은 효율성

### 🏗️ HDFS 아키텍처

#### 📋 **핵심 구성요소**

##### 1️⃣ **NameNode (네임노드)**
- **역할**: 파일 시스템 메타데이터 관리
- **기능**: 파일/디렉토리 열기, 닫기, 이름 변경
- **웹 인터페이스**: [NameNode 웹 UI](http://svmass2.mass.uhb.fr:50070)

##### 2️⃣ **DataNode (데이터노드)**
- **역할**: 실제 데이터 블록 저장
- **기능**: 블록 생성, 삭제, 복제
- **특징**: HDFS 파일에 대한 직접적인 지식 없음

##### 3️⃣ **Secondary NameNode (보조 네임노드)**
- **역할**: NameNode의 정보 백업
- **기능**: 메타데이터 체크포인트 생성
- **웹 인터페이스**: [Secondary NameNode 웹 UI](http://svmass2.mass.uhb.fr:50090/status.html)

### 🔄 **HDFS 동작 원리**

#### 📝 **파일 저장 과정**
1. **파일 분할**: NameNode가 파일을 블록으로 분할
2. **블록 저장**: DataNode에 블록 저장
3. **복제 관리**: 데이터 안전성을 위한 복제본 생성
4. **최적화**: 신뢰성, 가용성, 네트워크 대역폭 최적화

#### ⚠️ **중요 제약사항**
- **Write-Once**: 파일은 한 번만 쓰기 가능
- **단일 작성자**: 동시에 하나의 작성자만 허용
- **데이터 흐름**: 사용자 데이터는 NameNode를 거치지 않음

## 🔧 HDFS 명령어 활용하기

HDFS는 Unix/Linux 명령어와 유사한 인터페이스를 제공하여 분산 파일 시스템을 쉽게 관리할 수 있습니다.

### 📚 기본 명령어 구조
모든 [HDFS 명령어](http://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/FileSystemShell.html)는 `bin/hdfs` Java 스크립트를 통해 실행됩니다:

```shell
hdfs [SHELL_OPTIONS] COMMAND [GENERIC_OPTIONS] [COMMAND_OPTIONS]
```

### 📁 파일 및 디렉토리 관리

#### 🔍 **파일 목록 조회**
```shell
# 재귀적으로 하위 디렉토리와 사람이 읽기 쉬운 파일 크기로 목록 표시
hdfs dfs -ls -h -R
```

#### 📋 **파일 복사 및 이동**
```shell
# 파일 복사 (소스에서 목적지로)
hdfs dfs -cp source_path destination_path

# 파일 이동 (소스에서 목적지로)
hdfs dfs -mv source_path destination_path
```

#### 📂 **디렉토리 관리**
```shell
# 디렉토리 생성
hdfs dfs -mkdir /foodir

# 디렉토리 삭제 (재귀적)
hdfs dfs -rmr /foodir

# 파일 내용 보기
hdfs dfs -cat /foodir/myfile.txt
```

### 💡 명령어 활용 팁
- **`-h` 옵션**: 사람이 읽기 쉬운 형태로 파일 크기 표시
- **`-R` 옵션**: 재귀적으로 하위 디렉토리까지 처리
- **경로 구분**: HDFS 경로는 `/`로 시작하는 절대 경로 사용

## 🔄 노드 간 데이터 전송

HDFS에서 로컬 파일 시스템과 분산 파일 시스템 간의 데이터 전송은 빅데이터 처리의 핵심 작업입니다.

### 📤 **데이터 업로드 (put 명령어)**

#### 📋 **기본 문법**
```shell
hdfs fs -put [-f] [-p] [-l] [-d] [ - | <localsrc1> .. ]. <dst>
```

#### 🎯 **주요 옵션**
- **`-p`**: 권한과 수정 시간 보존
- **`-f`**: 대상이 이미 존재할 경우 덮어쓰기
- **`-l`**: 로컬 파일 시스템 제한
- **`-d`**: 디렉토리만 복사

#### 💡 **실제 사용 예시**
```shell
# 단일 파일 업로드
hdfs fs -put localfile /user/hadoop/hadoopfile

# 여러 파일 업로드 (덮어쓰기 옵션)
hdfs fs -put -f localfile1 localfile2 /user/hadoop/hadoopdir
```

### 🔄 **다양한 전송 명령어**

#### 📤 **업로드 관련**
- **`put`**: 로컬에서 HDFS로 복사
- **`moveFromLocal`**: 복사 후 로컬 파일 삭제
- **`copyFromLocal`**: 로컬 파일만 복사 (제한적)

#### 📥 **다운로드 관련**
- **`copyToLocal`**: HDFS에서 로컬로 복사 (제한적)

### 🏗️ **HDFS 블록 구조**

![HDFS 블록 구조](https://github.com/pnavaro/big-data/blob/master/notebooks/images/hdfs-fonctionnement.jpg?raw=1)

#### 📚 **중요한 특징**
- **NameNode 역할**: 데이터 경로에 직접 관여하지 않음
- **메타데이터 관리**: 데이터의 위치와 이동 경로만 제공
- **클러스터 맵**: 파일 시스템 메타데이터를 통한 데이터 위치 추적

## 🖥️ Hadoop 클러스터 구성

실습 환경은 8개의 컴퓨터로 구성된 Hadoop 클러스터입니다.

### 📊 **클러스터 구성**
- **총 노드 수**: 8개 컴퓨터 (sve1 ~ sve9)
- **분산 처리**: 여러 노드에서 병렬 작업 수행
- **고가용성**: 노드 장애 시에도 서비스 지속

### 🌐 **웹 인터페이스 접근**

#### 1️⃣ **NameNode 웹 인터페이스 (HDFS 계층)**
- **URL**: http://svmass2.mass.uhb.fr:50070
- **기능**: 
  - 클러스터 요약 정보 (총/남은 용량)
  - 활성/비활성 노드 상태
  - HDFS 네임스페이스 탐색
  - 파일 내용 웹 브라우저에서 확인
  - 로컬 머신의 Hadoop 로그 파일 접근

#### 2️⃣ **Secondary NameNode 정보**
- **URL**: http://svmass2.mass.uhb.fr:50090/
- **역할**: NameNode의 백업 및 체크포인트 관리

#### 3️⃣ **DataNode 정보**
- **노드별 URL**:
  - http://svpe1.mass.uhb.fr:50075/
  - http://svpe2.mass.uhb.fr:50075/
  - ...
  - http://svpe8.mass.uhb.fr:50075/
  - http://svpe9.mass.uhb.fr:50075/

### 💻 **JupyterLab 환경**
실습을 위해 [JupyterLab](https://jupyterlab.readthedocs.io) 환경을 사용할 수 있습니다:
- **접속 주소**: http://localhost:9000/lab
- **장점**: 웹 기반 인터랙티브 개발 환경
- **기능**: 노트북, 터미널, 파일 관리 통합

## 🎯 HDFS 실습 가이드

### 📋 **기본 환경 확인**

#### 1️⃣ **HDFS 홈 디렉토리 확인**
MapReduce 작업 실행에 필요한 HDFS 홈 디렉토리가 존재하는지 확인:
```bash
hdfs dfs -ls /user/${USER}
```

#### 2️⃣ **기본 명령어 연습**
```bash
# 현재 디렉토리 목록
hdfs dfs -ls

# 루트 디렉토리 목록
hdfs dfs -ls /

# 테스트 디렉토리 생성
hdfs dfs -mkdir test
```

### 📝 **파일 생성 및 업로드**

#### 1️⃣ **로컬 파일 생성**
```python
# 사용자 정보와 날짜가 포함된 파일 생성
echo "FirstName LastName" > user.txt
echo `date` >> user.txt 
cat user.txt
```

#### 2️⃣ **HDFS에 파일 업로드**
```bash
# 로컬 파일을 HDFS에 복사
hdfs dfs -put user.txt
```

#### 3️⃣ **업로드 확인**
```bash
# 재귀적으로 디렉토리 목록 확인
hdfs dfs -ls -R 

# 파일 내용 확인
hdfs dfs -cat user.txt 

# 파일 끝부분 확인
hdfs dfs -tail user.txt 
```

### 🔄 **파일 관리 연습**

#### 1️⃣ **파일 삭제 및 재업로드**
```bash
# 파일 삭제
hdfs dfs -rm user.txt

# 다시 업로드
hdfs dfs -copyFromLocal user.txt
```

#### 2️⃣ **디렉토리 이동**
```bash
# books 디렉토리로 파일 이동
hdfs dfs -mv user.txt books/user.txt

# 디렉토리 구조 확인
hdfs dfs -ls -R -h
```

#### 3️⃣ **파일 복사 및 정리**
```bash
# 파일 복사
hdfs dfs -cp books/user.txt books/hello.txt

# 사용자 디렉토리 용량 확인
hdfs dfs -count -h /user/$USER

# 원본 파일 삭제
hdfs dfs -rm books/user.txt
```

## 🎯 HDFS 실습 연습문제

다음 연습문제들을 순서대로 수행하여 HDFS 명령어 사용법을 익혀보세요.

### 📋 **연습문제 목록**

#### 1️⃣ **디렉토리 생성 및 탐색**
```bash
# HDFS에 'files' 디렉토리 생성
hdfs dfs -mkdir files

# 루트 디렉토리(/)의 내용 목록 표시
hdfs dfs -ls /
```

#### 2️⃣ **파일 생성 및 업로드**
```bash
# 오늘 날짜와 사용자 정보가 포함된 파일 생성
date > today.txt
whoami >> today.txt

# HDFS에 today.txt 파일 업로드
hdfs dfs -put today.txt
```

#### 3️⃣ **파일 내용 확인**
```bash
# today.txt 파일의 내용 표시
hdfs dfs -cat today.txt
```

#### 4️⃣ **파일 복사**
```bash
# today.txt 파일을 files 디렉토리로 복사
hdfs dfs -cp today.txt files/today.txt
```

#### 5️⃣ **Java 프로세스 정보 파일 생성**
```bash
# Java 프로세스 목록을 jps.txt 파일로 저장
jps > jps.txt

# 로컬 파일 시스템에서 HDFS로 복사
hdfs dfs -put jps.txt
```

#### 6️⃣ **파일 이동**
```bash
# jps.txt 파일을 files 디렉토리로 이동
hdfs dfs -mv jps.txt files/jps.txt
```

#### 7️⃣ **파일 삭제**
```bash
# 홈 디렉토리에서 today.txt 파일 삭제
hdfs dfs -rm today.txt
```

#### 8️⃣ **파일 끝부분 확인**
```bash
# jps.txt 파일의 마지막 몇 줄 표시
hdfs dfs -tail files/jps.txt
```

#### 9️⃣ **디스크 사용량 확인**
```bash
# du 명령어 도움말 표시
hdfs dfs -help du

# 홈 디렉토리의 총 사용 공간을 사람이 읽기 쉬운 형태로 표시
hdfs dfs -du -h /user/$USER
```

#### 🔟 **파일시스템 용량 확인**
```bash
# df 명령어 도움말 표시
hdfs dfs -help df

# 파일시스템의 사용 가능한 공간을 사람이 읽기 쉬운 형태로 표시
hdfs dfs -df -h
```

#### 1️⃣1️⃣ **파일 권한 변경**
```bash
# today.txt 파일의 권한을 사용자만 읽기/쓰기 가능하도록 변경
hdfs dfs -chmod 600 files/today.txt
```

### 💡 **학습 목표**
- HDFS 기본 명령어 숙련도 향상
- 파일 시스템 관리 능력 개발
- 분산 환경에서의 데이터 처리 이해

## ⚙️ YARN (Yet Another Resource Negotiator)

YARN은 Hadoop의 리소스 관리와 작업 스케줄링/모니터링을 담당하는 핵심 구성요소입니다.

### 📚 **YARN의 역할**
- **리소스 관리**: 클러스터의 CPU, 메모리, 디스크, 네트워크 리소스 관리
- **작업 스케줄링**: 애플리케이션 실행 순서 및 우선순위 관리
- **모니터링**: 실행 중인 작업의 상태 및 성능 모니터링

### 🏗️ **YARN 아키텍처**

#### 📋 **핵심 구성요소**

##### 1️⃣ **ResourceManager (리소스 매니저)**
- **역할**: 시스템 내 모든 애플리케이션 간 리소스 중재의 최종 권한
- **구성요소**: 
  - **Scheduler**: 애플리케이션에 리소스 할당
  - **ApplicationsManager**: 작업 제출 수락, 상태 추적, 진행 모니터링

##### 2️⃣ **NodeManager (노드 매니저)**
- **역할**: 각 머신의 프레임워크 에이전트
- **책임**: 
  - **Container** 관리
  - 리소스 사용량 모니터링 (CPU, 메모리, 디스크, 네트워크)
  - ResourceManager/Scheduler에 보고

##### 3️⃣ **ApplicationMaster (애플리케이션 마스터)**
- **역할**: 애플리케이션별 리소스 관리
- **기능**: 
  - ResourceManager와 리소스 협상
  - NodeManager와 협력하여 작업 실행 및 모니터링

### 🌐 **YARN 웹 인터페이스**

#### 📊 **JobTracker 웹 UI**
- **기능**: 
  - Hadoop 클러스터의 일반적인 작업 통계
  - 실행 중/완료/실패한 작업 정보
  - 작업 히스토리 로그 파일
  - 로컬 머신의 Hadoop 로그 파일 접근

#### 🔗 **접속 URL**
- **모든 애플리케이션**: http://svmass2.mass.uhb.fr:8088

### 🎯 **YARN의 장점**
- **확장성**: 수천 개의 노드로 확장 가능
- **유연성**: 다양한 애플리케이션 프레임워크 지원
- **효율성**: 리소스 사용률 최적화
- **안정성**: 장애 복구 및 내결함성

![YARN 아키텍처](https://github.com/pnavaro/big-data/blob/master/notebooks/images/yarn_architecture.png?raw=1)

*출처: http://hadoop.apache.org/docs/stable/hadoop-yarn/hadoop-yarn-site/yarn_architecture.gif*

## 📝 WordCount 예제 실행

Hadoop의 가장 기본적인 예제인 WordCount를 실행하여 MapReduce 작업의 전체 과정을 학습합니다.

### 📚 **WordCount 예제 개요**
- **구현**: Java로 작성된 [WordCount 예제](https://wiki.apache.org/hadoop/WordCount)
- **참고**: [Hadoop MapReduce 튜토리얼](https://hadoop.apache.org/docs/current/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html)
- **목적**: 텍스트 파일에서 단어 빈도 계산

### 🎯 **실습 준비**

#### 1️⃣ **입력 디렉토리 생성**
MapReduce 작업 실행에 필요한 HDFS 홈 디렉토리에 입력 디렉토리 생성:
```bash
hdfs dfs -mkdir -p /user/${USER}/input
```
> **참고**: `-p` 플래그는 디렉토리가 이미 존재해도 강제로 생성합니다.

#### 2️⃣ **샘플 파일 생성**
lorem Python 패키지를 사용하여 샘플 텍스트 파일들을 생성합니다.

### 🚀 **WordCount 실행**

#### 📋 **연습문제**
1. **필요한 파일들을 HDFS 시스템에 복사**
2. **Java 예제 실행**:
   ```bash
   hadoop jar /export/hadoop-2.7.6/share/hadoop/mapreduce/hadoop-mapreduce-examples-2.7.6.jar wordcount /user/you/input /user/you/output
   ```

#### 🔄 **YARN을 사용한 실행**
1. **출력 디렉토리 제거**:
   ```bash
   hdfs dfs -rm -r /user/you/output
   ```

2. **YARN을 사용한 실행**:
   ```bash
   yarn jar /export/hadoop-2.7.6/share/hadoop/mapreduce/hadoop-mapreduce-examples-2.7.6.jar wordcount /user/you/input /user/you/output
   ```

#### 🌐 **YARN 웹 인터페이스 모니터링**
- **접속**: [YARN 웹 사용자 인터페이스](http://svmass2.mass.uhb.fr:8088/cluster)
- **기능**: 작업 진행 상황, 로그, 성능 지표 확인
- **중요**: 로그를 자세히 읽어보며 작업 실행 과정 이해

### 💡 **학습 목표**
- Hadoop MapReduce 작업의 전체 생명주기 이해
- HDFS와 YARN의 협력 과정 파악
- 분산 환경에서의 데이터 처리 과정 체험

## 🐍 Hadoop에서 Python MapReduce 코드 배포

Hadoop에서 Python 코드를 실행하려면 [Hadoop Streaming API](http://hadoop.apache.org/docs/stable/hadoop-streaming/HadoopStreaming.html)를 사용해야 합니다.

### 📚 **Hadoop Streaming API**
- **목적**: Python의 `sys.stdin`(표준 입력)과 `sys.stdout`(표준 출력)을 통해 Map과 Reduce 코드 간 데이터 전달
- **장점**: Java 외의 언어로 MapReduce 작업 작성 가능
- **특징**: 표준 입출력을 통한 데이터 스트리밍

### 🎯 **구현 요구사항**
- **입력**: `sys.stdin`을 통해 데이터 읽기
- **출력**: `sys.stdout`을 통해 결과 출력
- **형식**: 키-값 쌍을 탭으로 구분하여 출력
- **스트리밍**: 데이터를 한 줄씩 처리

## 🗺️ Map 단계 구현

Map 단계에서는 `sys.stdin`에서 데이터를 읽어서 단어로 분리하고, 각 단어에 대해 `(단어, 1)` 쌍을 즉시 출력합니다.

### 📝 **Mapper 코드 구현**

```python
#!/usr/bin/env python
import sys

# 표준 입력에서 데이터 읽기
for line in sys.stdin:
    # 앞뒤 공백 제거 및 소문자 변환
    line = line.strip().lower()
    
    # 구두점 제거 (마침표를 공백으로 대체)
    line = line.replace(".", " ")
    
    # 라인을 단어로 분리
    for word in line.split():
        # 표준 출력으로 결과 출력
        # 여기서 출력되는 것이 Reduce 단계의 입력이 됩니다
        # 탭으로 구분된 형식; 단어 개수는 1
        print(f'{word}\t1')
```

### 🎯 **코드 설명**

#### 📋 **주요 기능**
- **입력 처리**: `sys.stdin`에서 한 줄씩 읽기
- **텍스트 정리**: 공백 제거, 소문자 변환, 구두점 제거
- **단어 분리**: 공백을 기준으로 단어 분리
- **출력 형식**: `단어\t1` 형태로 탭 구분 출력

#### 💡 **중요한 특징**
- **즉시 출력**: 각 단어에 대해 즉시 `(단어, 1)` 쌍 출력
- **탭 구분**: 키와 값을 탭(`\t`)으로 구분
- **스트리밍**: 한 줄씩 처리하여 메모리 효율성 확보

### 🔧 **실행 권한 설정**

Python 스크립트를 실행 가능하도록 권한을 설정해야 합니다:

```bash
chmod +x mapper.py
```

### 🧪 **로컬 테스트**

#### 📋 **터미널에서 테스트**
```bash
# 파이프를 사용한 테스트
cat sample01.txt | ./mapper.py | sort

# 또는 리다이렉션을 사용한 테스트
./mapper.py < sample01.txt | sort
```

#### 💡 **테스트 결과 확인**
- **입력**: 샘플 텍스트 파일
- **처리**: mapper.py 스크립트 실행
- **정렬**: `sort` 명령어로 알파벳 순 정렬
- **출력**: `(단어, 1)` 쌍의 정렬된 목록

### 🎯 **테스트의 중요성**
- **로컬 검증**: Hadoop 클러스터에 배포하기 전 로컬에서 동작 확인
- **디버깅**: 오류 발생 시 쉽게 수정 가능
- **성능 측정**: 작은 데이터셋으로 성능 확인

## 🔄 Reduce 단계 구현

Reduce 단계에서는 mapper.py의 결과를 읽어서 각 단어의 출현 횟수를 합산하고 최종 결과를 `sys.stdout`으로 출력합니다.

### 📝 **Reducer 코드 구현**

```python
#!/usr/bin/env python
from operator import itemgetter
import sys

# 현재 처리 중인 단어와 개수 초기화
current_word = None
current_count = 0
word = None

for line in sys.stdin:
    # mapper.py에서 받은 입력 파싱
    word, count = line.split('\t', 1)

    # count를 문자열에서 정수로 변환
    try:
        count = int(count)
    except ValueError:
        # count가 숫자가 아닌 경우 조용히 무시
        continue

    # Hadoop이 키(여기서는 단어)로 정렬하여 reducer에 전달하므로
    # 이 IF문이 작동합니다
    if current_word == word:
        current_count += count
    else:
        if current_word:
            # 결과를 sys.stdout에 출력
            print(f'{current_count}\t{current_word}')
        current_count = count
        current_word = word

# 마지막 단어도 출력하는 것을 잊지 마세요!
if current_word == word:
    print(f'{current_count}\t{current_word}')
```

### 🎯 **코드 설명**

#### 📋 **주요 기능**
- **입력 파싱**: `mapper.py`의 출력을 파싱하여 단어와 개수 분리
- **타입 변환**: 문자열 count를 정수로 변환
- **오류 처리**: 잘못된 형식의 입력 무시
- **집계**: 같은 단어의 개수들을 합산
- **출력**: `(개수, 단어)` 형태로 결과 출력

#### 💡 **중요한 특징**
- **정렬 의존성**: Hadoop이 Map 출력을 키로 정렬하므로 동일한 단어들이 연속으로 나타남
- **상태 관리**: `current_word`와 `current_count`로 현재 처리 중인 단어 추적
- **마지막 처리**: 마지막 단어도 반드시 출력해야 함

### 🔧 **실행 권한 설정**

mapper와 마찬가지로 Python 스크립트를 실행 가능하도록 권한을 설정해야 합니다:

```bash
chmod +x reducer.py
```

### 🧪 **전체 파이프라인 테스트**

#### 📋 **터미널에서 전체 테스트**
```bash
# 전체 MapReduce 파이프라인 테스트
cat sample.txt | ./mapper.py | sort | ./reducer.py | sort

# 또는 리다이렉션을 사용한 테스트
./mapper.py < sample01.txt | sort | ./reducer.py | sort
```

#### 🔄 **파이프라인 단계별 설명**
1. **입력**: `sample.txt` 파일 읽기
2. **Map**: `./mapper.py`로 단어 분리 및 `(단어, 1)` 쌍 생성
3. **정렬**: `sort`로 같은 단어들을 그룹화
4. **Reduce**: `./reducer.py`로 단어별 개수 합산
5. **최종 정렬**: `sort`로 결과 정렬
6. **출력**: `(개수, 단어)` 형태의 최종 결과

### 💡 **테스트 결과 확인**
- **입력**: 원본 텍스트 파일
- **중간 결과**: `(단어, 1)` 쌍들의 정렬된 목록
- **최종 결과**: `(개수, 단어)` 형태의 단어 빈도 분석 결과

### 🎯 **전체 파이프라인의 중요성**
- **통합 테스트**: Map과 Reduce 단계가 올바르게 연동되는지 확인
- **성능 측정**: 전체 처리 시간 및 메모리 사용량 측정
- **결과 검증**: 예상된 단어 빈도 결과와 일치하는지 확인

## 🚀 Hadoop 클러스터에서 실행

이제 로컬에서 테스트한 Python MapReduce 코드를 실제 Hadoop 클러스터에서 실행해보겠습니다.

### 📋 **실행 단계**

#### 1️⃣ **모든 파일을 HDFS 클러스터에 복사**
- 입력 데이터 파일들
- mapper.py와 reducer.py 스크립트

#### 2️⃣ **WordCount MapReduce 작업 실행**

### 🔧 **Python 실행 경로 수정**

#### 📍 **Python 경로 확인**
다음 명령어로 Python 실행 파일의 경로를 확인하세요:
```bash
which python
```

예상 결과:
```
/opt/tljh/user/bin/python
```

#### ✏️ **스크립트 수정**
확인된 경로로 다음 라인을 수정해야 합니다:
```python
# 기존
#!/usr/bin/env python

# 수정 후
#!/opt/tljh/user/bin/python
```

**수정 대상 파일**: `mapper.py`와 `reducer.py` 모두

### 🗂️ **출력 디렉토리 정리**

실행 전에 출력 디렉토리가 존재하지 않도록 제거합니다:
```bash
hdfs dfs -rm -r output
```

### 🎯 **Hadoop Streaming 실행**

Hadoop Streaming 라이브러리를 사용하여 HDFS의 파일을 읽고, Python 스크립트로 처리한 후 결과를 HDFS의 output 디렉토리에 저장합니다:

```bash
hadoop jar /export/hadoop-2.7.6/share/hadoop/tools/lib/hadoop-streaming-2.7.6.jar \
-input input/*.txt -output output \
-file ${PWD}/mapper.py -mapper ${PWD}/mapper.py \
-file ${PWD}/reducer.py -reducer ${PWD}/reducer.py
```

### 📊 **결과 확인**

실행 결과를 확인합니다:
```bash
hdfs dfs -cat output/*
```

### 💡 **Hadoop Streaming의 장점**
- **언어 독립성**: Java 외의 언어로 MapReduce 작업 작성 가능
- **표준 입출력**: 기존 Unix/Linux 도구들과 호환
- **디버깅 용이**: 로컬에서 테스트한 코드를 그대로 사용 가능

## 🛠️ Makefile을 사용한 자동화

긴 명령어들을 반복해서 입력하는 것을 피하기 위해 Makefile을 사용하여 작업을 자동화할 수 있습니다.

### 📝 **Makefile 생성**

```makefile
HADOOP_VERSION=2.7.6
HADOOP_HOME=/export/hadoop-${HADOOP_VERSION}
HADOOP_TOOLS=${HADOOP_HOME}/share/hadoop/tools/lib
HDFS_DIR=/user/${USER}
 
SAMPLES = sample01.txt sample02.txt sample03.txt sample04.txt

copy_to_hdfs: ${SAMPLES}
	hdfs dfs -mkdir -p ${HDFS_DIR}/input
	hdfs dfs -put $^ ${HDFS_DIR}/input

run_with_hadoop: 
	hadoop jar ${HADOOP_TOOLS}/hadoop-streaming-${HADOOP_VERSION}.jar \
    -file  ${PWD}/mapper.py  -mapper  ${PWD}/mapper.py \
    -file  ${PWD}/reducer.py -reducer ${PWD}/reducer.py \
    -input ${HDFS_DIR}/input/*.txt -output ${HDFS_DIR}/output-hadoop

run_with_yarn: 
	yarn jar ${HADOOP_TOOLS}/hadoop-streaming-${HADOOP_VERSION}.jar \
	-file  ${PWD}/mapper.py  -mapper  ${PWD}/mapper.py \
	-file  ${PWD}/reducer.py -reducer ${PWD}/reducer.py \
	-input ${HDFS_DIR}/input/*.txt -output ${HDFS_DIR}/output-yarn
```

### 🎯 **Makefile 사용법**

#### 📋 **주요 타겟들**
- **`copy_to_hdfs`**: 샘플 파일들을 HDFS에 복사
- **`run_with_hadoop`**: Hadoop을 사용한 MapReduce 실행
- **`run_with_yarn`**: YARN을 사용한 MapReduce 실행

#### 💡 **사용 예시**
```bash
# 파일들을 HDFS에 복사
make copy_to_hdfs

# Hadoop으로 실행
make run_with_hadoop

# YARN으로 실행
make run_with_yarn
```

### 🎯 **Makefile의 장점**
- **명령어 단순화**: 복잡한 명령어를 간단한 타겟으로 대체
- **재사용성**: 동일한 작업을 반복 실행할 때 편리
- **오류 방지**: 명령어 오타나 누락 방지
- **자동화**: 여러 단계의 작업을 한 번에 실행

### 🧪 **Makefile 실습**

#### 📋 **단계별 실행**
```bash
# 1. 기존 입력 디렉토리 제거
hdfs dfs -rm -r input

# 2. 샘플 파일들을 HDFS에 복사
make copy_to_hdfs

# 3. HDFS의 파일 목록 확인
hdfs dfs -ls input
```

#### 🚀 **Hadoop 실행 및 결과 확인**
```bash
# 1. 기존 출력 디렉토리 제거
hdfs dfs -rm -r -f output-hadoop

# 2. Hadoop Streaming으로 MapReduce 실행
make run_with_hadoop

# 3. 결과 확인
hdfs dfs -cat output-hadoop/*
```

### 🎯 **학습 성과**
- **Hadoop 환경 이해**: 분산 파일 시스템과 리소스 관리자 활용
- **Python MapReduce**: Java 외의 언어로 분산 처리 구현
- **자동화 도구**: Makefile을 통한 작업 자동화
- **실제 배포**: 로컬 코드를 클러스터 환경에서 실행

### 💡 **다음 단계**
- **성능 최적화**: 더 큰 데이터셋으로 성능 테스트
- **고급 기능**: 커스텀 파티셔너, 컴바이너 활용
- **모니터링**: YARN 웹 UI를 통한 작업 모니터링
- **확장**: 다른 MapReduce 알고리즘 구현
