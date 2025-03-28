
---
1. STFT를 제외하고 Anomal, Normal Data를 확연하게 구별할 수 있는 새로운 Feature발굴하기

---
## 🚀 새로운 Feature 발굴 방법

ASD (Anomaly Sound Detection) 시스템에서 **STFT(SHORT-TIME FOURIER TRANSFORM)** 이외의 특징(feature)으로 **Anomalous Data와 Normal Data를 효과적으로 구별**하는 방법을 소개할게.

---

## 🔥 1. 비주파수 도메인에서 새로운 Feature 발굴

STFT는 시간-주파수 분석에 집중하지만, **시간 도메인(time domain)과 비선형적(non-linear) 특징**에서도 이상 탐지에 유용한 정보를 찾을 수 있어.

### 📌 **1-1. 통계적 특징 (Statistical Features)**

- **Mean, Variance, Skewness, Kurtosis**  
    → 이상 데이터는 분포의 평균과 분산이 달라질 가능성이 있음.
    
- **Energy, Entropy**  
    → 비정상적인 소리는 에너지가 급격히 증가하거나 감소할 수 있음.
    
- **Zero Crossing Rate (ZCR)**  
    → 신호가 0을 몇 번이나 지나가는지 확인하면 정상/비정상 패턴을 잡아낼 수 있음.
    

---

### 📌 **1-2. 신호의 시계열 패턴 분석**

- **Autocorrelation**  
    → 정상적인 소리는 특정 주기성을 가지지만, 이상 데이터는 무작위성을 띨 가능성이 있음.
    
- **Peak Analysis (Envelope)**  
    → 정상 소리는 특정 패턴을 가지지만, 이상 데이터는 급격한 피크(peak)나 변화가 나타날 수 있음.
    
- **Short-Time Energy (STE)**  
    → 특정 구간에서 에너지를 계산하면 이상 데이터가 급격히 변할 가능성이 높음.
    

---

### 📌 **1-3. 비선형 특징 활용**

- **Fractal Dimension**  
    → 비정상 소리는 복잡도가 높거나 낮게 변할 수 있음.
    
- **Lyapunov Exponent (카오스 이론 활용)**  
    → 시스템의 민감도를 측정하면 정상/비정상 구분이 가능할 수 있음.
    
- **Hurst Exponent**  
    → 정상적인 신호는 특정 자기 유사성을 가지지만, 이상 데이터는 그렇지 않을 가능성이 있음.
    

---

## 🔥 2. 머신러닝을 활용한 Feature Selection

1. **Feature Engineering 후 다양한 Feature 추출**
    
    - 위에서 소개한 특징을 전부 계산한 후 벡터화
        
2. **차원 축소 기법 적용**
    
    - PCA(주성분 분석) 또는 LDA(선형판별분석)를 통해 주요 Feature 선정
        
3. **Feature Importance 평가**
    
    - Random Forest, XGBoost 등을 활용하여 중요한 Feature를 선별
        

---

## 🔥 3. 딥러닝을 활용한 Feature 학습

만약 수작업 Feature Selection이 어렵다면 **딥러닝을 활용**하는 것도 가능함.

1. **CNN으로 Mel-Spectrogram 활용 (STFT 대체)**
    
    - STFT 대신 **Mel-Spectrogram**을 활용하여 CNN 기반 모델을 훈련
        
2. **Autoencoder 기반 Feature 학습**
    
    - 정상 데이터만 학습한 후, 이상 데이터에서 Reconstruction Error가 커지는지 확인
        
3. **Contrastive Learning 활용**
    
    - 정상/이상 데이터를 Contrastive Loss를 활용하여 Feature Representation 학습
        

---

## ✅ 결론

STFT 없이 이상 데이터를 탐지하려면 **통계적 특징, 시계열 패턴 분석, 비선형 특징**을 활용하여 새로운 Feature를 발굴하는 것이 중요해!  
또한 머신러닝을 활용한 Feature Selection과 딥러닝 기반 Feature 학습도 효과적인 방법이 될 수 있음.

이 방법 중에서 프로젝트에 적용하기 적절한 방법을 선택해서 실험해 보면 좋을 것 같아! 🚀

---

아래는 새로운 Feature 발굴을 위한 라이브러리 및 코드 샘플을 추천해줄게. 주로 **Python**을 활용하며, 여러 통계적, 시계열적, 비선형적 특징을 추출할 수 있는 라이브러리를 소개할 거야.

---

## 1. **통계적 특징 추출 라이브러리**

### 📌 **`scipy`, `numpy`**

`scipy`와 `numpy`를 활용하면 기본적인 통계적 특징을 쉽게 계산할 수 있어.

```python
import numpy as np
from scipy.stats import kurtosis, skew

def statistical_features(signal):
    mean = np.mean(signal)
    variance = np.var(signal)
    skewness = skew(signal)
    kurt = kurtosis(signal)
    return mean, variance, skewness, kurt
```

### 📌 **`librosa`**

음악/소리 신호 처리를 위한 라이브러리로 **Zero Crossing Rate, Energy, Entropy 등**을 쉽게 추출할 수 있어.

```python
import librosa
import numpy as np

def signal_features(file_path):
    signal, sr = librosa.load(file_path)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(signal).mean()

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr).mean()

    # Spectral Roll-Off
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr).mean()

    # Spectral Flatness
    flatness = librosa.feature.spectral_flatness(y=signal).mean()

    return zcr, centroid, rolloff, flatness
```

---

## 2. **시계열 패턴 분석**

### 📌 **`statsmodels`**

시계열 분석에 유용한 `autocorrelation`을 계산할 수 있어.

```python
import statsmodels.api as sm

def autocorrelation(signal, lags=40):
    acf = sm.tsa.acf(signal, nlags=lags)
    return acf
```

### 📌 **`pywt` (Wavelet Transform)**

시계열 신호의 패턴을 분석하는 데 유용한 **Wavelet Transform**을 활용할 수 있어.

```python
import pywt
import numpy as np

def wavelet_transform(signal):
    coeffs = pywt.wavedec(signal, 'db1')  # db1 is Daubechies wavelet
    return coeffs
```

---

## 3. **비선형적 특징 활용**

### 📌 **`nolds` (Fractal Dimension)**

프랙탈 차원을 계산할 수 있는 라이브러리 `nolds`.

```python
import nolds

def fractal_dimension(signal):
    return nolds.dfa(signal)
```

### 📌 **`chaospy` (Lyapunov Exponent)**

카오스 이론을 적용할 수 있는 **Lyapunov Exponent**를 계산할 수 있어.

```python
import chaospy as cp

def lyapunov_exponent(signal):
    # Lyapunov exponent 계산 방법은 시스템에 따라 달라질 수 있음.
    # 예시로 간단히 구현한 방식.
    return cp.stat.pearson(signal)  # Example, replace with actual Lyapunov calculation
```

---

## 4. **머신러닝 기반 Feature Selection**

### 📌 **`sklearn`**

**Random Forest**, **XGBoost** 등을 활용해 중요 Feature를 선택하는 방법을 소개할게.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

def feature_importance(X_train, y_train):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    model = SelectFromModel(clf, threshold="mean", max_features=5)
    X_selected = model.transform(X_train)
    
    return X_selected, clf.feature_importances_
```

### 📌 **`xgboost`**

`XGBoost`를 활용한 feature importance 계산 예시.

```python
import xgboost as xgb
from xgboost import plot_importance

def xgboost_feature_importance(X_train, y_train):
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    
    plot_importance(model, importance_type='weight')  # 중요 feature 시각화
    return model.feature_importances_
```

---

## 5. **딥러닝을 활용한 Feature 학습**

### 📌 **`tensorflow`, `keras`**

Autoencoder 기반으로 특징을 학습하고 Reconstruction Error를 활용하여 이상 여부를 판단하는 방법.

```python
import tensorflow as tf
from tensorflow.keras import layers

def autoencoder_model(input_dim):
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(128, activation='relu')(input_layer)
    encoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(128, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = tf.keras.models.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    return autoencoder

# 학습 후 Reconstruction Error 계산
def reconstruction_error(model, X_test):
    reconstructed = model.predict(X_test)
    error = np.mean(np.square(X_test - reconstructed), axis=1)
    return error
```

---

## 6. **Feature Selection과 모델 학습을 결합한 전체 프로세스 예시**

```python
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 신호에서 특징 추출
def extract_features(file_paths):
    features = []
    labels = []
    for file_path in file_paths:
        signal, sr = librosa.load(file_path)
        zcr, centroid, rolloff, flatness = signal_features(file_path)
        features.append([zcr, centroid, rolloff, flatness])
        labels.append(0 if "normal" in file_path else 1)  # 0: Normal, 1: Anomalous
    return np.array(features), np.array(labels)

# 데이터 로딩 및 특징 추출
file_paths = ["normal_1.wav", "normal_2.wav", "anomaly_1.wav"]
X, y = extract_features(file_paths)

# 훈련 데이터와 테스트 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest 모델 학습
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

이 코드들은 **특징 추출**과 **머신러닝 모델**을 결합하여 ASD 시스템의 성능을 향상시킬 수 있도록 도와줄 거야. 각 방법을 조합하거나 필요에 따라 수정해 가며 적용해 보기를 권장해! 😊