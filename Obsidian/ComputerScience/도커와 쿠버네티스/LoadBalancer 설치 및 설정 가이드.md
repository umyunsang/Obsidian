
---
### LoadBalancer 설치 및 설정 가이드

#### 1. **Kube-Proxy 설정 변경**

`kube-proxy`의 설정을 편집하여 `strictARP`를 활성화합니다.

```bash
kubectl edit configmap -n kube-system kube-proxy
```

- `ipvs` 섹션에서 `strictARP`를 `true`로 변경:
    
    ```yaml
    ipvs:
      excludeCIDRs: null
      minSyncPeriod: 0s
      scheduler: ""
      strictARP: true  # 기존 false에서 true로 변경
    ```
    

---

#### 2. **MetalLB 다운로드 및 설치**

1. **MetalLB 아카이브 다운로드**
    
    ```bash
    wget https://github.com/metallb/metallb/archive/refs/tags/v0.12.1.tar.gz
    ```
    
2. **아카이브 압축 해제 및 디렉토리 이동**
    
    ```bash
    tar -xvzf v0.12.1.tar.gz
    cd metallb-0.12.1/manifests
    ```
    
3. **MetalLB 네임스페이스 및 구성 적용**
    
    ```bash
    kubectl apply -f namespace.yaml
    kubectl apply -f metallb.yaml
    ```
    
4. **MetalLB 리소스 상태 확인**
    
    ```bash
    kubectl get all -n metallb-system
    ```
    

---

#### 3. **Layer 2 Configuration 설정**

1. **`example-layer2-config.yaml` 파일 편집**
    
    - `addresses`를 네트워크 대역에 맞게 수정:
        
        ```yaml
        addresses:
          - 192.168.11.230-192.168.11.233  # 네트워크 대역에 맞게 수정
        ```
        
2. **Layer 2 Configuration 적용**
    
    ```bash
    kubectl apply -f example-layer2-config.yaml
    ```
    

---

#### 4. **LoadBalancer 서비스 생성**

1. **테스트 파드 생성**
    
    ```bash
    kubectl run myweb3 --image nginx:1.12 --labels "app=web-svc3" --port 80
    ```
    
2. **`load.yaml` 파일 작성**
    
    ```yaml
    apiVersion: v1
    kind: Service
    metadata:
      name: sv3
    spec:
      selector:
        app: web-svc3
      type: LoadBalancer
      ports:
      - port: 80
        targetPort: 80
    ```
    
3. **LoadBalancer 서비스 적용**
    
    ```bash
    kubectl apply -f load.yaml
    ```
    
4. **서비스 확인**
    
    ```bash
    kubectl get svc
    ```
    

---
