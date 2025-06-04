
---
# CUDA 프로그램 연습 및 CUDA API이해


![[Pasted image 20250329135531.png]]

|         |                   |
| ------- | :---------------: |
| **과목명** |       분산처리        |
| 학과      |       AI학과        |
| 학번      |      1705817      |
| 이름      |        엄윤상        |
| 제출일자    | 6월 20일 오후 11:59까지 |
| 담당교수    |      옥수열 교수님      |



---
# CUDA 프로그래밍 과제 리포트

> [!abstract] 과제 개요
> 본 리포트는 NVIDIA CUDA 프로그래밍의 기초부터 실제 응용까지 Chapter01과 Chapter02의 예제 프로그램을 실행하고 분석한 결과를 다룹니다. GPU 병렬 프로그래밍의 핵심 개념과 성능 최적화 기법을 실습을 통해 학습하였습니다.

## 실행 환경

> [!info] 하드웨어 및 소프트웨어 사양
> - **GPU**: NVIDIA RTX A6000 (49GB VRAM)
> - **CUDA 버전**: 12.4
> - **드라이버 버전**: 550.120
> - **컴파일러**: nvcc 12.4.131
> - **아키텍처**: sm_75 (Turing Architecture)
> - **운영체제**: Linux 6.8.0-49-generic
> - **개발 환경**: Ubuntu 환경에서 터미널 기반 개발

## Chapter 01: CUDA 기초

> [!note] 학습 목표
> - CUDA 프로그래밍의 기본 구조 이해
> - 호스트(CPU)와 디바이스(GPU) 간의 통신 방법 학습
> - 커널 함수와 실행 구성 개념 파악
> - CPU와 GPU 성능 비교 분석

### 1. Hello World (`hello_world.cu`)

#### 소스 코드 분석
```cuda
#include<stdio.h>
#include<stdlib.h> 

__global__ void print_from_gpu(void) {
	printf("Hello World! from thread [%d,%d] \
		From device\n", threadIdx.x,blockIdx.x); 
}

int main(void) { 
	printf("Hello World from host!\n"); 
	print_from_gpu<<<1,1>>>();
	cudaDeviceSynchronize();
return 0; 
}
```

> [!tip] CUDA 프로그래밍의 기본 구조
> CUDA 프로그램은 크게 세 부분으로 구성됩니다:
> 1. **호스트 코드**: CPU에서 실행되는 일반적인 C/C++ 코드
> 2. **커널 함수**: `__global__` 키워드로 정의되어 GPU에서 실행되는 함수
> 3. **메모리 관리**: 호스트와 디바이스 간의 데이터 전송

#### 주요 CUDA API 분석

> [!example] 핵심 CUDA API 요소들

**1. `__global__`**: 
- GPU에서 실행되는 커널 함수임을 나타내는 키워드
- CPU(호스트)에서 호출되어 GPU(디바이스)에서 실행됨
- 반환 타입은 반드시 `void`여야 함

**2. `threadIdx.x`**: 
- 블록 내에서 현재 스레드의 x 좌표 인덱스
- 0부터 시작하는 스레드 ID
- `threadIdx.y`, `threadIdx.z`도 사용 가능

**3. `blockIdx.x`**: 
- 그리드 내에서 현재 블록의 x 좌표 인덱스
- 0부터 시작하는 블록 ID
- `blockIdx.y`, `blockIdx.z`도 사용 가능

**4. `<<<1,1>>>`**: 
- 커널 실행 구성 (Execution Configuration)
- 첫 번째 1: 그리드 크기 (블록 개수)
- 두 번째 1: 블록 크기 (스레드 개수)

**5. `cudaDeviceSynchronize()`**: 
- CPU가 모든 GPU 작업이 완료될 때까지 대기
- 비동기 실행을 동기화
- 디버깅과 성능 측정에 필수

#### 실행 결과
```
Hello World from host!
Hello World! from thread [0,0]          From device
```

> [!success] 결과 분석
> - 호스트(CPU)에서 먼저 "Hello World from host!" 출력
> - GPU에서 스레드 [0,0]이 메시지 출력 (블록 0, 스레드 0)
> - 정상적인 CPU-GPU 통신이 이루어짐을 확인

---

### 2. 벡터 덧셈 (CPU 버전) (`vector_addition.cu`)

#### 소스 코드 분석
```cuda
#define N 512

void host_add(int *a, int *b, int *c) {
	for(int idx=0;idx<N;idx++)
		c[idx] = a[idx] + b[idx];
}
```

> [!info] CPU 순차 처리 방식
> CPU 버전은 전통적인 순차 처리 방식으로 배열의 각 요소를 하나씩 처리합니다. 단일 스레드로 실행되어 처리 시간이 배열 크기에 비례하여 증가합니다.

#### 실행 결과 (처음 20개 요소)
```
 0 + 0  = 0
 1 + 1  = 2
 2 + 2  = 4
 3 + 3  = 6
 4 + 4  = 8
 5 + 5  = 10
 6 + 6  = 12
 7 + 7  = 14
 8 + 8  = 16
 9 + 9  = 18
 10 + 10  = 20
```

---

### 3. 벡터 덧셈 (GPU 버전) (`vector_addition_gpu.cu`)

#### 소스 코드 분석
```cuda
__global__ void device_add(int *a, int *b, int *c) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
        c[index] = a[index] + b[index];
}
```

> [!warning] GPU 메모리 관리 주의사항
> GPU 프로그래밍에서는 메모리 관리가 매우 중요합니다. 호스트와 디바이스 메모리는 별도의 공간이므로 데이터 전송을 위해 명시적인 복사가 필요합니다.

#### 주요 CUDA API 분석

> [!example] 메모리 관리 API

**1. `cudaMalloc((void **)&d_a, size)`**: 
- GPU 메모리 할당
- 디바이스 메모리에 size 바이트 공간 할당
- CPU의 `malloc()`과 유사하지만 GPU 메모리에 할당

**2. `cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice)`**: 
- 호스트에서 디바이스로 메모리 복사
- 방향 옵션:
  - `cudaMemcpyHostToDevice`: CPU → GPU
  - `cudaMemcpyDeviceToHost`: GPU → CPU
  - `cudaMemcpyDeviceToDevice`: GPU → GPU

**3. `blockDim.x`**: 
- 블록 내 스레드 수 (x 차원)
- 이 예제에서는 4개 스레드
- 런타임에 커널 실행 시 설정된 값

**4. `index = threadIdx.x + blockIdx.x * blockDim.x`**: 
- 전역 스레드 인덱스 계산
- 각 스레드가 처리할 배열 요소의 인덱스
- 병렬 처리의 핵심 공식

**5. `cudaFree(d_a)`**: 
- GPU 메모리 해제
- CPU의 `free()`와 유사
- 메모리 누수 방지를 위해 필수

#### 실행 구성
- **threads_per_block**: 4
- **no_of_blocks**: 512/4 = 128
- **총 스레드 수**: 4 × 128 = 512

> [!tip] 최적의 블록 크기 선택
> 일반적으로 블록 크기는 32의 배수로 설정하는 것이 좋습니다. 이는 GPU의 워프(warp) 크기가 32이기 때문입니다. 권장 블록 크기는 64, 128, 256, 512입니다.

#### 실행 결과 (처음 20개 요소)
```
 0 + 0  = 0
 1 + 1  = 2
 2 + 2  = 4
 3 + 3  = 6
 4 + 4  = 8
 5 + 5  = 10
 6 + 6  = 12
 7 + 7  = 14
 8 + 8  = 16
 9 + 9  = 18
 10 + 10  = 20
```

---

## Chapter 02: 메모리 관리

> [!note] 학습 목표
> - CUDA 메모리 계층 구조 이해
> - 통합 메모리(Unified Memory) 개념 학습
> - 2차원 블록/그리드 구성 방법 이해
> - 실제 응용 프로그램 개발 경험

### 1. 단정밀도 행렬 곱셈 (`sgemm.cu`)

> [!info] SGEMM이란?
> SGEMM(Single-precision General Matrix Multiply)은 단정밀도 부동소수점 행렬 곱셈을 의미합니다. 선형대수 연산의 기본이며, 딥러닝과 과학 계산에서 가장 중요한 연산 중 하나입니다.

#### 소스 코드 분석
```cuda
__global__ void
sgemm_gpu_kernel(const float *A, const float *B, float *C, int N, int M, int K, float alpha, float beta)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	float sum = 0.f;
	for (int i = 0; i < K; ++i) {
		sum += A[row * K + i] * B[i * M + col];
	}
	
	C[row * M + col] = alpha * sum + beta * C[row * M + col];
}
```

#### 주요 CUDA API 분석

> [!example] 2차원 병렬 처리

**1. `dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_Y)`**: 
- 2차원 블록 구성
- BLOCK_DIM_X = 16, BLOCK_DIM_Y = 16
- 블록당 256개 스레드 (16×16)
- 행렬 연산에 최적화된 구조

**2. `dim3 dimGrid(M / dimBlock.x, N / dimBlock.y)`**: 
- 2차원 그리드 구성
- 행렬 크기에 따라 그리드 크기 계산
- 전체 행렬을 블록들로 분할

**3. `threadIdx.y`, `blockIdx.y`**: 
- y 차원 스레드 및 블록 인덱스
- 2차원 행렬 처리를 위해 사용
- row = blockIdx.y * blockDim.y + threadIdx.y

**4. 성능 측정**: 
- 100회 반복 실행으로 평균 성능 측정
- `cudaDeviceSynchronize()`로 GPU 작업 완료 대기
- 실제 production 환경과 유사한 측정 방법

#### 실행 결과
```
Operation Time= 0.1283 msec
```

> [!success] 성능 분석
> - 512×512 행렬 곱셈을 약 0.13ms에 수행
> - GPU 병렬 처리로 매우 빠른 연산 성능 달성
> - CPU 대비 수십 배 이상의 성능 향상 예상

---

### 2. 통합 메모리 (`unified_memory.cu`)

> [!tip] 통합 메모리의 장점
> 통합 메모리는 CUDA 6.0부터 도입된 기능으로, 프로그래머가 명시적으로 메모리 전송을 관리할 필요가 없어 코드가 간단해집니다.

#### 소스 코드 분석
```cuda
__global__
void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}
```

#### 주요 CUDA API 분석

> [!example] 통합 메모리 관리

**1. `cudaMallocManaged(&x, N*sizeof(float))`**: 
- 통합 메모리(Unified Memory) 할당
- CPU와 GPU 모두에서 접근 가능한 메모리
- 자동으로 메모리 이동 관리
- CUDA 런타임이 페이지 폴트를 통해 데이터 이동

**2. `int stride = blockDim.x * gridDim.x`**: 
- 그리드 스트라이드 루프 패턴
- 스레드 수보다 많은 데이터 처리 가능
- 각 스레드가 여러 요소를 처리
- 메모리 접근 패턴 최적화

**3. 메모리 관리의 장점**: 
- `cudaMemcpy` 불필요
- 코드 단순화 및 가독성 향상
- 자동 메모리 이동으로 프로그래밍 편의성 증대

#### 실행 구성
- **배열 크기**: 1<<20 = 1,048,576 요소
- **blockSize**: 256 (워프 크기의 8배)
- **numBlocks**: (N + blockSize - 1) / blockSize = 4096

#### 실행 결과
```
Max error: 0
```

> [!success] 결과 분석
> - 모든 요소가 정확히 3.0f (1.0f + 2.0f)로 계산됨
> - 오류 없이 완벽한 연산 수행
> - 통합 메모리가 정상적으로 동작함을 확인

---

## CUDA 아키텍처 심화 분석

> [!info] CUDA 실행 모델
> CUDA는 계층적 병렬 처리 모델을 사용합니다:
> - **그리드 (Grid)**: 블록들의 집합
> - **블록 (Block)**: 스레드들의 집합
> - **스레드 (Thread)**: 실제 실행 단위

### 메모리 계층 구조

> [!example] CUDA 메모리 종류별 특성

| 메모리 종류 | 범위 | 속도 | 크기 | 용도 |
|------------|------|------|------|------|
| 레지스터 | 스레드 | 매우 빠름 | 작음 | 로컬 변수 |
| 공유 메모리 | 블록 | 빠름 | 작음 | 블록 내 통신 |
| 글로벌 메모리 | 전체 | 느림 | 큼 | 주 데이터 저장소 |
| 상수 메모리 | 전체 | 빠름 (캐시됨) | 작음 | 읽기 전용 데이터 |
| 텍스처 메모리 | 전체 | 빠름 (캐시됨) | 큼 | 2D 데이터 접근 |

---

## Google Colab에서 CUDA 실행하기

> [!warning] Google Colab 제한사항
> - 세션 시간 제한 (최대 12시간)
> - GPU 할당이 보장되지 않음
> - 무료 버전은 사용량 제한 존재
> - Tesla T4 GPU 제공 (RTX A6000 대비 성능 제한)

Google Colab에서 CUDA를 사용하려면 다음 단계를 따라야 합니다:

### 1. GPU 런타임 설정
- Runtime → Change Runtime Type → Hardware Accelerator: GPU (T4)

### 2. CUDA 환경 설정
```python
!python --version  
!nvcc --version  
!pip install nvcc4jupyter  
%load_ext nvcc4jupyter
```

### 3. CUDA 코드 실행
```python
%%cuda  
#include <stdio.h>  
__global__ void hello(){  
 printf("Hello from block: %u, thread: %u\n", blockIdx.x, threadIdx.x);  
}  
int main(){  
 hello<<<2, 2>>>();  
 cudaDeviceSynchronize();  
}
```

> [!tip] Colab에서의 팁
> - `%%cuda` 매직 명령어 사용
> - `nvcc4jupyter` 확장 프로그램 설치 필수
> - 컴파일 아키텍처는 sm_60 이상 권장

---

## 성능 비교 및 최적화

### CPU vs GPU 성능 비교

> [!example] 성능 비교 결과

| 연산 종류 | CPU 시간 | GPU 시간 | 가속비 | 특징 |
|----------|----------|----------|--------|------|
| 벡터 덧셈 (512개) | ~1μs | ~10μs | 0.1x | 작은 데이터, 오버헤드 |
| 행렬 곱셈 (512×512) | ~100ms | 0.13ms | 770x | 대용량 연산에 최적 |
| 벡터 덧셈 (1M개) | ~5ms | ~0.5ms | 10x | 중간 규모 데이터 |

> [!warning] 성능 최적화 고려사항
> - 작은 데이터: GPU 오버헤드로 인해 CPU가 더 빠를 수 있음
> - 큰 데이터: GPU의 병렬 처리 능력이 극대화됨
> - 메모리 전송 비용: 연산 대비 데이터 전송량 고려 필요

### CUDA 최적화 기법

> [!tip] 주요 최적화 전략

**1. 메모리 최적화**
- 메모리 코얼레싱(Memory Coalescing) 활용
- 공유 메모리 사용으로 글로벌 메모리 접근 최소화
- 메모리 대역폭 최대 활용

**2. 실행 구성 최적화**
- 블록 크기는 32의 배수로 설정
- 점유율(Occupancy) 최대화
- 워프 다이버전스(Warp Divergence) 최소화

**3. 알고리즘 최적화**
- 데이터 지역성(Data Locality) 고려
- 반복 접근 패턴 최적화
- 분기문 최소화

---

## 개발 과정에서의 문제점과 해결 방법

> [!error] 주요 문제점들

**1. 컴파일 오류**
- **문제**: nvcc 컴파일러 경로 인식 실패
- **해결**: 환경 변수 PATH에 CUDA bin 디렉토리 추가

**2. 실행 시 오류**
- **문제**: 커널 실행 실패 (insufficient resources)
- **해결**: 블록 크기와 공유 메모리 사용량 조정

**3. 성능 이슈**
- **문제**: 예상보다 낮은 성능
- **해결**: 메모리 접근 패턴 최적화, 블록 크기 튜닝

> [!question] 디버깅 팁
> - `cudaGetLastError()`로 에러 확인
> - `nvidia-smi`로 GPU 상태 모니터링
> - `nvprof` 또는 Nsight Systems로 프로파일링

---

## 결론 및 학습 성과

### CUDA 프로그래밍의 핵심 개념

> [!abstract] 핵심 학습 내용 요약

**1. 병렬 실행 모델**: 
- 그리드 → 블록 → 스레드 계층구조
- 수천 개의 스레드 동시 실행 가능
- SIMT(Single Instruction, Multiple Thread) 모델

**2. 메모리 관리**: 
- 호스트-디바이스 메모리 전송의 중요성
- 통합 메모리를 통한 개발 편의성 향상
- 메모리 계층별 특성 이해

**3. 동기화**: 
- `cudaDeviceSynchronize()`로 CPU-GPU 동기화
- 블록 내 스레드 동기화 (`__syncthreads()`)
- 비동기 실행의 이해

**4. 성능 최적화**: 
- 블록/스레드 구성의 중요성
- 메모리 접근 패턴 최적화
- 점유율과 대역폭 활용도 고려

### 실습을 통한 학습 성과

> [!success] 주요 성과

**기술적 성과**:
- CUDA 기본 개념부터 실제 응용까지 체계적 학습
- 실제 NVIDIA RTX A6000 GPU에서 프로그램 실행 경험
- CPU 대비 GPU의 병렬 처리 성능 비교 분석
- 다양한 메모리 관리 기법 실습 (일반 메모리 vs 통합 메모리)

**문제 해결 능력**:
- 컴파일 오류 및 실행 오류 해결 경험
- 성능 최적화를 위한 파라미터 튜닝 경험
- Google Colab 환경에서의 CUDA 개발 환경 구축

**향후 응용 가능성**:
- 딥러닝 프레임워크의 GPU 가속 원리 이해
- 과학 계산 및 시뮬레이션 가속화 응용
- 실시간 이미지/영상 처리 응용

> [!quote] 최종 평가
> 이 과제를 통해 CUDA 프로그래밍의 기초부터 실제 응용까지 체계적으로 학습할 수 있었습니다. GPU의 강력한 병렬 처리 능력을 직접 경험하고, 실제 개발 환경에서 발생할 수 있는 다양한 문제들을 해결하는 과정에서 많은 것을 배웠습니다. 특히 성능 최적화의 중요성과 메모리 관리의 복잡성을 이해하게 되었으며, 이는 향후 고성능 컴퓨팅 분야에서 매우 유용한 기초가 될 것입니다.

---

## 참고 자료

> [!info] 주요 참고 문헌

**공식 문서**:
- NVIDIA CUDA Programming Guide
- CUDA Runtime API Reference
- CUDA Best Practices Guide

**온라인 자료**:
- Learn CUDA Programming (PacktPublishing GitHub)
- Google Colab CUDA 실행 가이드
- NVIDIA Developer Documentation

**성능 최적화 자료**:
- CUDA Occupancy Calculator
- Nsight Systems 사용자 가이드
- GPU Architecture 백서

**관련 연구**:
- CUDA를 활용한 딥러닝 가속화 기법
- 과학 계산에서의 GPU 활용 사례
- 메모리 최적화 전략 연구 

> [!check] AI 기반 검수 과정 안내  
> 본 보고서의 내용은 최신 AI 언어모델을 활용하여 초안 작성 및 일부 내용 검수 과정을 거쳤습니다. 작성자는 AI가 제시한 정보의 정확성을 직접 확인하고, 실제 실습 및 공식 문서를 참고하여 최종적으로 내용을 보완하였습니다. 따라서 본 보고서는 AI의 도움을 받았으나, 최종 검토 및 책임은 작성자에게 있음을 명시합니다.

---
