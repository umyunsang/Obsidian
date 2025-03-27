
---
### NodePort 서비스 설정 및 가이드

#### 1. **파드 생성**

NodePort 서비스를 위한 파드 생성:

```bash
kubectl run myweb2 --image nginx:1.12 --labels "app=web-svc2" --port 80
```

---

#### 2. **NodePort 서비스 매니페스트 작성**

NodePort를 사용하는 서비스 매니페스트 파일(`nodeport.yaml`) 작성:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: svc-nodeport  # 서비스 이름
spec:
  selector:
    app: web-svc2  # 'web-svc2' 라벨을 가진 파드를 선택
  type: NodePort  # 클러스터 외부에서 접근 가능한 서비스
  ports:
  - port: 80          # 서비스가 노출하는 포트
    targetPort: 80    # 파드 내부 컨테이너 포트
    nodePort: 30080   # 노드의 특정 포트 (30000~32767 범위에서 선택 가능)
    protocol: TCP
```

---

#### 3. **NodePort 서비스 적용 및 확인**

1. **매니페스트 파일 적용**:
    
    ```bash
    kubectl apply -f nodeport.yaml
    ```
    
2. **서비스 상태 확인**:
    
    ```bash
    kubectl get svc
    kubectl describe svc svc-nodeport
    ```
    
    - `NodePort`에 설정된 포트를 확인합니다.

---

#### 4. **서비스 테스트**

1. **서비스 및 파드 확인**:
    
    ```bash
    kubectl get svc
    kubectl get po -o wide
    ```
    
    - 각 노드의 IP 주소와 `NodePort`를 확인.
2. **외부 접속 테스트**:
    
    - 브라우저 또는 `curl` 명령어를 사용하여 서비스 확인:
        
        ```bash
        curl http://<노드의-외부-IP>:30080
        ```
        
    - `<노드의-외부-IP>`는 클러스터 노드의 IP 주소.

---

#### 5. **클라이언트 파드 생성**

클러스터 내부에서 통신을 테스트하기 위한 클라이언트 파드 생성:

```bash
kubectl run clientpod --image nginx
```

---

#### 6. **클라이언트 파드에서 서비스 테스트**

1. **클라이언트 파드에 접속**:
    
    ```bash
    kubectl exec -it clientpod -- bash
    ```
    
2. **서비스 통신 테스트**:
    
    - **클러스터 IP 사용**:
        
        ```bash
        curl http://<서비스의-클러스터-IP>
        ```
        
    - **NodePort 사용**:
        
        ```bash
        curl http://<노드의-외부-IP>:30080
        ```
        

---

#### 7. **NodePort 서비스의 구조**

- **동작 원리**:
    - 클러스터의 모든 노드에서 지정된 `NodePort`를 통해 서비스를 노출.
    - 외부에서 클러스터 노드의 IP 주소와 `NodePort`를 사용하여 접근.
- **DNS 역할**:
    - 클러스터 내부에서는 CoreDNS를 통해 서비스 이름으로 통신.
    - 외부에서는 클러스터 IP 대신 노드 IP와 `NodePort`를 사용.

---

#### 8. **Endpoints 확인**

서비스와 연결된 파드의 IP를 확인:

```bash
kubectl describe svc svc-nodeport
```

- `Endpoints` 섹션에서 서비스와 매핑된 파드의 IP를 확인할 수 있습니다.

---
