
---

#### 1. **파드 생성**

ClusterIP 서비스를 위한 파드 생성:

```bash
kubectl run web --image nginx:1.12 --labels "app=websvr" --port 80
```

---

#### 2. **ClusterIP 서비스 매니페스트 작성**

ClusterIP 서비스 매니페스트 파일(`cluster.yaml`) 작성:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: svc  # 서비스 이름
spec:
  selector:
    app: websvr  # 'websvr' 라벨을 가진 파드를 선택
  type: ClusterIP  # 클러스터 내부에서만 접근 가능한 서비스
  ports:
  - port: 80        # 서비스가 노출하는 포트
    targetPort: 80  # 파드 내부 컨테이너 포트
    protocol: TCP
```

---

#### 3. **ClusterIP 서비스 적용 및 확인**

1. **매니페스트 파일 적용**:
    
    ```bash
    kubectl apply -f cluster.yaml
    ```
    
2. **서비스 및 파드 상태 확인**:
    
    ```bash
    kubectl get svc
    kubectl get po -o wide
    ```
    
    - 서비스의 클러스터 IP 및 포트를 확인합니다.
3. **서비스의 상세 정보 확인**:
    
    ```bash
    kubectl describe svc svc
    ```
    
    - `Endpoints` 섹션에서 서비스와 연결된 파드의 IP 주소를 확인합니다.

---

#### 4. **클라이언트 파드 생성**

클러스터 내부에서 서비스를 테스트하기 위한 클라이언트 파드 생성:

```bash
kubectl run clientpod --image nginx
```

---

#### 5. **클라이언트 파드에서 서비스 테스트**

1. **클라이언트 파드에 접속**:
    
    ```bash
    kubectl exec -it clientpod -- bash
    ```
    
2. **서비스 통신 테스트**:
    
    - **클러스터 IP 사용**:
        
        ```bash
        curl http://<서비스의-클러스터-IP-주소>
        ```
        
    - **서비스 이름 사용**:
        
        ```bash
        curl svc
        ```
        

---

#### 6. **클러스터 내 DNS 서비스 확인**

1. **CoreDNS 관련 서비스 확인**:
    
    ```bash
    kubectl get svc -n kube-system --show-labels
    ```
    
2. **CoreDNS 파드 확인**:
    
    ```bash
    kubectl get po -n kube-system | grep coredns
    ```
    
    - CoreDNS는 클러스터 내의 DNS 서비스로, 서비스 이름을 IP 주소로 변환하는 역할을 합니다.

---

#### 7. **ClusterIP의 구조**

- **동작 원리**:
    - ClusterIP 서비스는 클러스터 내부에서만 접근 가능하며, 외부에서 직접 접근할 수 없습니다.
    - CoreDNS를 통해 서비스 이름을 사용하여 파드와 통신할 수 있습니다.
- **DNS 역할**:
    - 클러스터 내에서 서비스 이름을 통해 통신이 가능하며, CoreDNS가 이를 처리합니다.

---

#### 8. **Endpoints 확인**

서비스와 연결된 파드의 IP 주소를 확인:

```bash
kubectl describe svc svc
```

- `Endpoints` 섹션에서 서비스와 매핑된 파드의 IP 주소를 확인할 수 있습니다.

---

###### Quiz. **아래의 조건에 해당하는 deployment 와  service 설정을  quiz.yaml  파일에 설정하세요.**
```
deployment 이름:  quiz-deploy
pod 의 초기 개수: 5
pod 의 label:   nginx-testbed
pod 의 이름:  www
pod 의 이미지: nginx:1.14
pod 의 port 번호:  80 / TCP
  
service 의 이름: quiz-svc
service 의 type: clusterip
targetport: 80
```

>[! deployment와 service설정을 quiz..yaml파일로 생성]
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quiz-deploy
spec:
  replicas: 5  # 초기 파드 개수
  selector:
    matchLabels:
      app: nginx-testbed  # 파드의 라벨
  template:
    metadata:
      labels:
        app: nginx-testbed  # 파드의 라벨
    spec:
      containers:
      - name: www  # 파드 이름
        image: nginx:1.14  # 파드의 이미지
        ports:
        - containerPort: 80  # 파드의 포트 번호

---
apiVersion: v1
kind: Service
metadata:
  name: quiz-svc  # 서비스 이름
spec:
  selector:
    app: nginx-testbed  # 'nginx-testbed' 라벨을 가진 파드를 선택
  type: ClusterIP  # 클러스터 내부에서만 접근 가능한 서비스
  ports:
    - port: 80        # 서비스가 노출하는 포트
      targetPort: 80  # 파드 내부 컨테이너 포트
      protocol: TCP

```

>[!적용 확인]
```bash
# quiz.yaml 파일을 클러스터에 적용
kubectl apply -f quiz.yaml
# 생성된 서비스 목록 확인
kubectl get svc
# 생성된 파드 목록 및 상세 정보 확인
kubectl get po -o wide

# 클라이언트 역할을 할 파드 생성
kubectl run clientpod --image nginx
# 클라이언트 파드 터미널 접속
kubectl exec -it clientpod -- bash
# curl로 연결 확인
@clientpod:/~$ curl quiz-svc
```
---
