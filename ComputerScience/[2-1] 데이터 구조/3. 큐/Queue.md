
---
## 큐

- 큐(Queue): 삽입과 삭제가 양 끝에서 각각 수행되는 자료구조 

- 일상생활의 관공서, 은행, 우체국, 병원 등에서 번호표를 이용한 줄서기 

- 선입 선출(First-In First-Out, FIFO)


#### `파이썬 리스트 큐`

```python
def add(item): # 삽입연산
	q.append(item)

def remove(): # 삭제연산
	if len(q) != 0:
		item = q.pop(0)
		return item

def print_q(): # 큐출력
	print('front -> ', end='')
	for i in range(len(q)):
		print('{!s:<8}'.format(q[i]), end='')
	print(' <- rear')
```

---
#### `단순 연결 리스트 큐`

```python
class Node:  
    def __init__(self, item, n):  
        self.item = item  
        self.next = n  
  
  
front = None  
rear = None  
size = 0  
  
  
def add(item):  # 삽입 연산  
    global size  
    global front  
    global rear  
    new_node = Node(item, None)  
    if size == 0:  
        front = new_node  
    else:  
        rear.next = new_node  
    rear = new_node  
    size += 1  
  
  
def remove():  # 삭제 연산  
    global size  
    global front  
    global rear  
    if size != 0:  
        fitem = front.item  
        front = front.next  
        size -= 1  
        if size == 0:  
            rear = None  
        return fitem  
  
  
def print_q():  # 큐출력  
    p = front  
    print('front: ', end='')  
    while p:  
        if p.next is not None:  
            print(p.item, '-> ', end='')  
        else:  
            print(p.item, end='')  
        p = p.next  
    print(' : reear')
```

#### 수행 시간

- 리스트로 구현한 큐의 add와 remove 연산: 각각 O(1) 시간 
	- 리스트 크기를 확대/축소시키는 경우에 큐의 모든 항목을 새 리스트에 복사해야 하므로 O(n) 시간 
- 단순 연결 리스트 큐의 add와 remove 연산은 각각 O(1) 시간 
	- 삽입 또는 삭제 연산이 rear 와 front로 인해 연결 리스트의 다른 노드를 방문할 필요 없음

---
## 데크

![](../../../../image/Pasted%20image%2020240816190547.png)

- 데크(Double-ended Queue, Deque): 양쪽 끝에서 삽입과 삭제를 허용하는 자료구조 

- 데크는 스택과 큐 자료구조를 혼합한 자료구조 

- 따라서 데크는 스택과 큐를 동시에 구현하는데 사용

- 데크를 이중 연결 리스트로 구현하는 것이 편리

- 단순 연결 리스트는 노드의 이전 노드의 레퍼런스를 알아야 삭제

- 파이썬에는 데크가 Collections 패키지에 정의되어 있음

- 삽입, 삭제 등의 연산은 파이썬의 리스트의 연산과 매우 유사

```python
from collections import deque     
  
dq = deque('data')  
for elem in dq:  
    print(elem.upper(), end='')  
print()  
  
dq.append('r')  
dq.appendleft('k')  
print(dq)  
  
dq.pop()  
dq.popleft()  
print(dq[-1])  
print('x' in dq)      
  
dq.extend('structure')  
dq.extendleft(reversed('python'))  
print(dq)
```

#### 수행 시간

- 데크를 배열이나 이중 연결 리스트로 구현한 경우, 스택과 큐의 수행 시간과 동일 
- 양 끝에서 삽입과 삭제가 가능하므로 프로그램이 다소 복잡
- 이중 연결 리스트로 구현한 경우는 더 복잡함
---
## 요약

- 큐는 삽입과 삭제가 양 끝에서 각각 수행되는 선입 선출(FIFO) 자료구조 
- 큐는 CPU의 태스크 스케줄링, 네트워크 프린터, 실시간 시스템의 인터럽트 처리, 다양한 이벤트 구동 방식 컴퓨터 시뮬레이션, 콜 센터의 전화 서비스 처리 등에 사용되며, 이진트리의 레벨순회와 그래프의 BFS에 사용 
- 데크는 양쪽 끝에서 삽입과 삭제를 허용하는 자료구조로서 스택과 큐 자료구조를 혼합한 자료구조 
- 데크는 스크롤, 문서 편집기의 undo 연산, 웹 브라우저의 방문 기록 등에 사용