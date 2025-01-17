
---
### **Kubernetes Ingress 설정 정리**

### 1. **Ingress-NGINX 설치**

#### 1.1. **Ingress Controller 설치**

- Bare Metal 환경에서 Ingress-NGINX 설치:

```bash
wget https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.11.2/deploy/static/provider/baremetal/deploy.yaml
```

- `deploy.yaml` 파일의 **366행**에서 `type` 수정:

```yaml
type: LoadBalancer  # NodePort에서 LoadBalancer로 변경
```

- 수정한 파일 적용:

```bash
kubectl apply -f deploy.yaml
```

#### 1.2. **설치 상태 확인**

```bash
kubectl get ns                   # 네임스페이스 확인
kubectl get all -n ingress-nginx # 리소스 확인
kubectl get svc -n ingress-nginx # 서비스 정보 확인
```

- **EXTERNAL-IP** 확인 (예: `192.168.11.231`):

```plaintext
NAME                        TYPE           CLUSTER-IP       EXTERNAL-IP      PORT(S)                      AGE
ingress-nginx-controller    LoadBalancer   10.102.195.216   192.168.11.231   80:32218/TCP,443:30533/TCP   101s
```

---

### 2. **디플로이먼트와 서비스 생성**

#### 2.1. **기본 NGINX 서비스**

```bash
kubectl create deploy nginx-main --image=nginx
kubectl expose deploy nginx-main --name nginx-main-svc --port 80
```

#### 2.2. **색상별 NGINX 서비스**

- **블루 버전**:

```bash
kubectl create deploy nginx-blue --image=thekoguryo/nginx-hello:blue
kubectl expose deploy nginx-blue --name nginx-blue-svc --port 80
```

- **그린 버전**:

```bash
kubectl create deploy nginx-green --image=thekoguryo/nginx-hello:green
kubectl expose deploy nginx-green --name nginx-green-svc --port 80
```

#### 2.3. **HTTPD 서비스**

```bash
kubectl create deploy httpd-main --image=httpd
kubectl expose deploy httpd-main --name httpd-main-svc --port 80
```

- **상태 확인**:

```bash
kubectl get deploy,svc,po
```

---

### 3. **Ingress 설정**

#### 3.1. **Ingress 매니페스트 작성**

`ig.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: myingress
spec:
  rules:
  - host: www.uys1998.com
    http:
      paths:
      - pathType: Prefix
        path: "/"
        backend:
          service:
            name: nginx-main-svc
            port:
              number: 80
      - pathType: Prefix
        path: "/blue"
        backend:
          service:
            name: nginx-blue-svc
            port:
              number: 80
      - pathType: Prefix
        path: "/green"
        backend:
          service:
            name: nginx-green-svc
            port:
              number: 80
  - host: www.guru2025.co.kr
    http:
      paths:
      - pathType: Prefix
        path: "/"
        backend:
          service:
            name: httpd-main-svc
            port:
              number: 80
```

#### 3.2. **Ingress 리소스 적용**

```bash
kubectl apply -f ig.yaml
kubectl get ing                  # 생성된 Ingress 확인
kubectl describe ing myingress   # 상세 정보 확인
```

---

### 4. **로컬 네임 해석 설정**

#### 4.1. **호스트 파일 수정**

- **Windows**: `C:\Windows\System32\drivers\etc\hosts`
    
- **Linux/Mac**: `/etc/hosts`
    
- 파일의 **마지막 줄**에 다음 내용 추가:
    
```
192.168.11.231 www.uys1998.com www.guru2025.co.kr
```

- 수정 후 저장.

---

### 5. **Ingress 동작 확인**

- **URL 접속**:
    - [http://www.uys1998.com](http://www.uys1998.com/) → 기본 `nginx-main`
    - [http://www.uys1998.com/blue](http://www.uys1998.com/blue) → `nginx-blue`
    - [http://www.uys1998.com/green](http://www.uys1998.com/green) → `nginx-green`
    - [http://www.guru2025.co.kr](http://www.guru2025.co.kr/) → `httpd-main`

---
