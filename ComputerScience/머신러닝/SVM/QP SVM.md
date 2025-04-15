
---
#### Quadratic Programming 기반으로 SVM

```
from sklearn.svm import SVC
# 선형 커널을 사용하는 SVM 모델 정의 및 학습
clf = SVC(kernel='linear') # kernel:linear,rbf,poly,sigmoid 등
clf.fit(X, y)

# 시각화 실행
visualize_svm(clf.coef_[0], clf.intercept_)
```
![[Pasted image 20250415173546.png]]

```
margin = 2 / np.sqrt(np.dot(clf.coef_[0].T, clf.coef_[0]))
print(margin)
```

