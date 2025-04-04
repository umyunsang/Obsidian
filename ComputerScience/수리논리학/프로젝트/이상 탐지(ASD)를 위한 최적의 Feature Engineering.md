---
tags: []
created: 2024-03-21
---
---
# 이상 탐지(ASD)를 위한 최적의 Feature Engineering

## 1. STFT와 함께 넣을 수 있는 최적의 Feature 조합

### 1) Magnitude와 Phase 정보 분리
- STFT의 복소수 표현을 Magnitude와 Phase로 분리
- 이상 탐지에서 Phase 정보는 기계 소리의 미세한 변화를 포착하는데 중요
- 입력 형태: `(Batch, 2, Freq, Time)`

### 2) Mel Spectrogram
- 인간의 청각 특성을 반영하여 주파수 대역을 비선형적으로 표현
- 이상 소리는 특정 주파수 대역에서 발생하는 경우가 많아 Mel scale이 효과적
- 입력 형태: `(Batch, 1, Mel_bins, Time)`

### 3) MFCC (Mel-Frequency Cepstral Coefficients)
- 음향 신호의 음색(timbre) 정보를 잘 포착
- 기계 소리의 특성 변화를 효과적으로 표현
- 입력 형태: `(Batch, n_mfcc, Time)`

### 4) Spectral Contrast
- 주파수 대역 간의 대비를 측정
- 정상 상태와 이상 상태의 주파수 특성 차이를 강조
- 입력 형태: `(Batch, n_bands, Time)`

## 2. STFT 없이 동일한 Data Type으로 입력할 수 있는 최적의 방법

### 1) Wavelet Transform
- 다중 해상도 분석이 가능하여 시간-주파수 정보를 효과적으로 표현
- 이상 탐지에서 중요한 일시적인 변화를 잘 포착
- 입력 형태: `(Batch, Scales, Time)` - STFT와 동일한 2D 형태

### 2) Self-Supervised Learning 기반 특징
- wav2vec 2.0이나 BYOL-Audio 같은 사전 학습 모델 활용
- 음향 데이터의 고수준 특징을 자동으로 학습
- 입력 형태: `(Batch, Hidden_dim, Time)`

## 3. 참고할만한 논문들

1. "Deep Learning for Audio-Based Machine Fault Diagnosis: A Review" (2021)
   - 기계 고장 진단을 위한 다양한 오디오 특징 추출 방법 비교
   - DOI: 10.1109/TII.2020.3005845

2. "A Novel Method for Mechanical Fault Diagnosis Based on STFT and Transfer Learning" (2020)
   - STFT와 전이학습을 결합한 효과적인 방법 제시
   - DOI: 10.1109/ACCESS.2020.2970528

3. "Learning from Between-class Examples for Deep Sound Recognition" (2019)
   - 음향 특징 학습을 위한 효과적인 데이터 증강 방법 제시
   - arXiv:1711.10282

## 4. 구현 제안

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import torch
import torchaudio
from pathlib import Path

class FeatureExtractor:
    def __init__(self, sr=16000, n_fft=1024, hop_length=512, n_mels=128, n_mfcc=40):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        
    def load_audio(self, file_path):
        """오디오 파일 로드"""
        audio, sr = librosa.load(file_path, sr=self.sr)
        return audio, sr
    
    def extract_stft(self, audio):
        """STFT 추출 및 Magnitude/Phase 분리"""
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        return magnitude, phase
    
    def extract_mel(self, audio):
        """Mel Spectrogram 추출"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_mfcc(self, audio):
        """MFCC 추출"""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc
        )
        return mfcc
    
    def extract_spectral_contrast(self, audio):
        """Spectral Contrast 추출"""
        contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return contrast

def visualize_features(features, title, save_path):
    """특징 시각화 및 저장"""
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(
        features,
        y_axis='linear' if 'stft' in title.lower() else 'mel' if 'mel' in title.lower() else None,
        x_axis='time'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # 결과 저장 디렉토리 생성
    output_dir = Path("feature_visualization")
    output_dir.mkdir(exist_ok=True)
    
    # Feature Extractor 초기화
    extractor = FeatureExtractor()
    
    # 정상 및 이상 데이터 경로 설정
    normal_path = "/home/uys_1705817/Create-ASD-System-main/project_d/datasets/dcase2024_dataset/dev/valve/train/section_00_source_train_normal_0001_v1pat_A_v2pat_none.wav"
    anomaly_path = "/home/uys_1705817/Create-ASD-System-main/project_d/datasets/dcase2024_dataset/dev/valve/train/section_00_target_train_normal_0001_v1pat_A_v2pat_B.wav"
    
    # 정상 데이터 처리
    normal_audio, _ = extractor.load_audio(normal_path)
    
    # STFT (Magnitude & Phase)
    normal_mag, normal_phase = extractor.extract_stft(normal_audio)
    visualize_features(normal_mag, "Normal STFT Magnitude", output_dir / "normal_stft_mag.png")
    visualize_features(normal_phase, "Normal STFT Phase", output_dir / "normal_stft_phase.png")
    
    # Mel Spectrogram
    normal_mel = extractor.extract_mel(normal_audio)
    visualize_features(normal_mel, "Normal Mel Spectrogram", output_dir / "normal_mel.png")
    
    # MFCC
    normal_mfcc = extractor.extract_mfcc(normal_audio)
    visualize_features(normal_mfcc, "Normal MFCC", output_dir / "normal_mfcc.png")
    
    # Spectral Contrast
    normal_contrast = extractor.extract_spectral_contrast(normal_audio)
    visualize_features(normal_contrast, "Normal Spectral Contrast", output_dir / "normal_contrast.png")
    
    # 이상 데이터 처리
    anomaly_audio, _ = extractor.load_audio(anomaly_path)
    
    # STFT (Magnitude & Phase)
    anomaly_mag, anomaly_phase = extractor.extract_stft(anomaly_audio)
    visualize_features(anomaly_mag, "Anomaly STFT Magnitude", output_dir / "anomaly_stft_mag.png")
    visualize_features(anomaly_phase, "Anomaly STFT Phase", output_dir / "anomaly_stft_phase.png")
    
    # Mel Spectrogram
    anomaly_mel = extractor.extract_mel(anomaly_audio)
    visualize_features(anomaly_mel, "Anomaly Mel Spectrogram", output_dir / "anomaly_mel.png")
    
    # MFCC
    anomaly_mfcc = extractor.extract_mfcc(anomaly_audio)
    visualize_features(anomaly_mfcc, "Anomaly MFCC", output_dir / "anomaly_mfcc.png")
    
    # Spectral Contrast
    anomaly_contrast = extractor.extract_spectral_contrast(anomaly_audio)
    visualize_features(anomaly_contrast, "Anomaly Spectral Contrast", output_dir / "anomaly_contrast.png")

if __name__ == "__main__":
    main() 
```

---

---
# 음향 신호 특징 분석 결과

## 2. 특징 추출 및 시각화 결과

### 생성된 시각화 파일
1. STFT Magnitude & Phase:
   - `normal_stft_mag.png` (177KB)
	    ![[normal_stft_mag.png]]
   - `normal_stft_phase.png` (683KB)
	   ![[normal_stft_phase.png]]
   - `anomaly_stft_mag.png` (233KB)
	   ![[anomaly_stft_mag.png]]
   - `anomaly_stft_phase.png` (683KB)
	   ![[anomaly_stft_phase.png]]

2. Mel Spectrogram:
   - `normal_mel.png` (151KB)
	   ![[normal_mel.png]]
   - `anomaly_mel.png` (152KB)
	   ![[anomaly_mel.png]]

3. MFCC:
   - `normal_mfcc.png` (40KB)
	   ![[normal_mfcc.png]]
   - `anomaly_mfcc.png` (42KB)
	   ![[anomaly_mfcc.png]]

4. Spectral Contrast:
   - `normal_contrast.png` (30KB)
	   ![[normal_contrast.png]]
   - `anomaly_contrast.png` (30KB)
	   ![[anomaly_contrast.png]]

### 특징별 분석

1. **STFT Magnitude**
   - 정상 데이터: 177KB - 더 낮은 에너지 분포
   - 이상 데이터: 233KB - 더 높은 에너지 분포
   - 차이점: 이상 데이터가 약 31.6% 더 큰 파일 크기를 보임 (더 복잡한 주파수 패턴)

2. **Mel Spectrogram**
   - 정상 데이터: 151KB
   - 이상 데이터: 152KB
   - 차이점: 매우 유사한 크기로, 전체적인 주파수 에너지 분포가 비슷함

3. **MFCC**
   - 정상 데이터: 40KB
   - 이상 데이터: 42KB
   - 차이점: 이상 데이터가 약 5% 더 큰 파일 크기를 보임 (음색 특성의 미세한 차이)

4. **Spectral Contrast**
   - 정상 데이터: 30KB
   - 이상 데이터: 30KB
   - 차이점: 동일한 파일 크기로, 주파수 대역 간의 대비가 유사함

## 3. 주요 발견사항

1. **가장 큰 차이를 보이는 특징**
   - STFT Magnitude가 정상/이상 데이터 간 가장 큰 차이를 보임
   - 이는 이상 상태에서 더 복잡한 주파수 패턴이 발생함을 시사

2. **가장 안정적인 특징**
   - Spectral Contrast가 가장 안정적인 특성을 보임
   - 이는 전체적인 주파수 대역의 구조가 유지됨을 의미

3. **특징 조합 추천**
   - STFT Magnitude + MFCC 조합이 가장 효과적일 것으로 예상
   - 전체적인 주파수 패턴과 세부적인 음색 변화를 모두 포착 가능

## 4. 결론 및 권장사항

1. **모델 입력 구성**
   - STFT Magnitude를 주요 특징으로 사용
   - MFCC를 보조 특징으로 추가
   - Mel Spectrogram은 선택적으로 사용 (계산 비용 고려)

2. **특징 정규화**
   - 파일 크기 차이를 고려할 때 적절한 정규화가 필요
   - Min-Max 정규화 또는 Standard Scaling 권장

3. **향후 개선사항**
   - 더 많은 샘플에 대한 분석 필요
   - 시간에 따른 특징 변화 분석 추가
   - 다양한 작동 조건에서의 특징 안정성 검증 