
---
https://kubernetes.io/docs/concepts/services-networking/ingress-controllers/
에 가서
Kubernetes as a project supports and maintains [AWS](https://github.com/kubernetes-sigs/aws-load-balancer-controller#readme), [GCE](https://git.k8s.io/ingress-gce/README.md#readme), and [nginx](https://git.k8s.io/ingress-nginx/README.md#readme) ingress controllers.
사용하려는 곳을 링크로 가서 (nginx 설치 할 예정)
https://github.com/kubernetes/ingress-nginx/blob/main/README.md#readme
get started 링크 접속 
https://kubernetes.github.io/ingress-nginx/deploy/
bare metal 선택

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.12.0/deploy/static/provider/baremetal/deploy.yaml
```
를 v1.11.2로 변경해서 리눅스에 설치

```bash
wget https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.11.2/deploy/static/provider/baremetal/deploy.yaml
```
deploy.yaml 파일에 366행 
```
type: NodePort <- LoadBalancer 로 변경
```

```bash
# 적용하고 확인하기
kubectl apply -f deploy.yaml 
kubectl get ns
kubectl get all -n ingress-nginx
kubectl get svc -n ingress-nginx
```
kubectl get svc -n ingress-nginx 로 나온 내용중 이터널 ip 기억
```
NAME                                 TYPE           CLUSTER-IP       EXTERNAL-IP      PORT(S)                      AGE
ingress-nginx-controller             LoadBalancer   10.102.195.216   192.168.11.231   80:32218/TCP,443:30533/TCP   101s
```
디플로이먼트 생성하고
```
kubectl create deploy nginx-main --image=nginx
kubectl expose deploy nginx-main --name nginx-main-svc --port 80
kubectl get deploy,svc,po
```
연결하기 위한 파드와 서비스 생성하기
```

kubectl create deploy nginx-blue --image=thekoguryo/nginx-hello:blue
kubectl expose deploy nginx-blue --name nginx-blue-svc --port 80
kubectl get deply,svc,po
```
```
kubectl create deploy nginx-green --image=thekoguryo/nginx-hello:green
kubectl expose deploy nginx-green --name nginx-green-svc --port 80
kubectl get deply,svc,po
```
```
kubectl create deploy httpd-main --image=httpd
kubectl expose deploy httpd-main --name httpd-main-svc --port 80
kubectl get deploy,svc,po
```

인그레스 yaml 작성
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

```bash
# 실행
kubectl apply -f ig.yaml
# 확인 
kubectl get ing
kubectl describe ing myingress
```

```
C:\Windows\System32\drivers\etc 에 있는 hosts
```
hosts파일의 젤 마지막줄에 
인그레스 이터널 ip 아까 기억해둔 <192.168.11.231> 항목 추가
