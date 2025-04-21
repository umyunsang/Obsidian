
---
# Quadratic Programming 기반 SVM(Support Vector Machine)

## 1. scikit-learn 라이브러리를 활용한 SVM 구현
scikit-learn의 SVC 클래스를 사용하여 SVM 모델을 구현하고 학습시킵니다.

```python
from sklearn.svm import SVC
# 선형 커널을 사용하는 SVM 모델 정의 및 학습
clf = SVC(kernel='linear') # kernel:linear,rbf,poly,sigmoid 등
clf.fit(X, y)

# 시각화 실행
visualize_svm(clf.coef_[0], clf.intercept_)
```

![[Pasted image 20250415173546.png]]

## 2. 마진 계산
학습된 SVM 모델의 마진 크기를 계산합니다.

```python
margin = 2 / np.sqrt(np.dot(clf.coef_[0].T, clf.coef_[0]))
print(margin)
```




