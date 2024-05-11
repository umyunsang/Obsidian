
---
#### 트리(TREE) 정의 
- 트리는 나무 모양의 자료구조 
	- 계층적인 관계를 가진 자료의 표현에 매우 유용 
	- 비선형 자료구조, 계층구조 
	![[Pasted image 20240511132732.png]]
- 활용 예시: 운영체제의 파일시스템, 탐색 트리, 우선순위 큐, 결정 트리 등

#### 트리(TREE) 관련 용어

![[Pasted image 20240511132912.png]]

| 용어     | 내용                                                        |
| ------ | --------------------------------------------------------- |
| 부모 노드  | 다른 노드들과 연결된 상위 노드                                         |
| 자식 노드  | 부모 노드와 직접 연결된 하위 노드                                       |
| 형제 노드  | 같은 부모 노드를 가진 다른 노드들                                       |
| 조상 노드  | 어떤 노드에서 루트 노드까지의 경로상에 있는 모든 상위 노드들                        |
| 자손 노드  | 어떤 노드 하위에 연결된 모든 하위 노드들                                   |
| 단말 노드  | 자식 노드가 없는 노드로, 자식이 있으면 비단말 노드로 분류됨                        |
| 노드의 차수 | 노드가 가지고 있는 자식의 수, 단말 노드는 항상 0                             |
| 트리의 차수 | 트리에 포함된 모든 노드의 차수 중에서 가장 큰 수                              |
| 레벨     | 트리의 각 층에 번호를 매기는 것, 루트 노드의 레벨은 1이고, 한 층씩 내려갈 수록 레벨은 1씩 증가 |
| 트리의 높이 | 트리가 가지고 있는 최대 레벨                                          |
#### 트리(TREE) 표기 방법
- 노드와 간선의 연결 관계
	- 중첩된 집합
	- 중첩된 괄호
	- 들여쓰기 (indentation)
	![[Pasted image 20240511133126.png]]

#### 트리(TREE) 표현법
- 방법1 : N-링크표현
	- 자식의 개수에 제한이 없는 트리 (genderal TREE)
		![[Pasted image 20240511133321.png]]

- 방법2: 왼쪽 자식 – 오른쪽 형제

![[Pasted image 20240511133507.png]]


#### 이진 트리 (Binary TREE)
- 모든 노드가 최대 2개의 자식만을 가질 수 있는 트리
	- 모든 노드의 차수가 2 이하로 제한 
	- 자식 노드에도 순서가 존재 
	-  컴퓨터 분야에서 널리 활용되는 기본적인 자료 구조 
	-  데이터의 구조적인 관계를 잘 반영 
	-  효율적인 삽입과 탐색 가능 
	-  일반적인 TREE에 비해 계층적인 관계를 가지는 모든 자료형을 표현하기에는 부족
- 활용예시
	- 이진 탐색트리 (Binary search tree) 
	- 우선순위 큐를 효과적으로 구현하는 힙 트리(heap tree) – 컴퓨터 프로세스 과정의 중요한 알고리즘 중 하나 
	-  수식을 트리 형태로 표현하여 계산하는 수식 트리

#### 이진 트리 (Binary TREE) 종류

![[Pasted image 20240511133810.png]]

- 포화 이진 트리 (full binary tree)
	- 트리의 각 레벨에 노드가 꽉 차 있는 이진 트리
- 완전 이진 트리 (Complete binary tree)
	- 마지막 레벨을 제외한 각 레벨이 노드들로 꽉 차 있는 있는 트리를 말하며, 마지막 레벨에서는 왼쪽부터 오른쪽으로 노드가 순서대로 채워져 있는 이진 트리 
	- 마지막 레벨에서는 노드가 꽉 차 있지 않아도 되지만 중간에 빈 곳이 있으면 안 됨 
	- “포화 이진 트리는 항상 완전 이진 트리”는 성립, “완전 이진 트리는 항상 포화 이진 트리”는 성립되지 않음
	
	![[Pasted image 20240511134048.png]]
	
- 균형 이진 트리 (balanced binary tree)
	- 높이 균형 이진 트리 (height-balanced binary tree) 
	- 모든 노드에서 좌우 서브 트리의 높이 차이가 1 이하인 트리를 말하며, 높이 차이가 1 초과할 경우 경사트리
	
	![[Pasted image 20240511134217.png]]

#### 이진 트리 (Binary TREE) 와 배열 자료형의 관계

![[Pasted image 20240511134312.png]]

- 이진 트리의 특성을 활용한 배열 자료형에 저장할 경우 탐색 및 활용이 용이함 
- 배열의 첫 인덱스는 건너띄고, 두번째 인덱스 부터 저장 
- 루트 노드부터 각 레벨로 내려오고, 각 레벨은 왼쪽에서 오른쪽으로 순차적으로 저장

![[Pasted image 20240511134458.png]]

**Quiz
	문제 1. "D"의 부모 노드는? -> B
	문제 2. "D"의 왼쪽 자식노드와 오른쪽 자식노드는? -> H, I


#### 이진 트리 구현을 위한 클래스

![[Pasted image 20240511135134.png]]
```python
class TreeNode:
	def __init__(self, value, left=None, right=None):
		self.value = value
		self.left = left
		self.right = right
```

#### 이진 트리 구현
- 노드를 이용한 이진 트리 구현은 복잡
	- 주로 배열 자료형을 이용 
	- 레벨이 증가할 수록 비교 해야 할 노드가 기하급수적으로 증가 
	- 이전(앞) 노드에 대한 정보가 없으므로 매번 root 노드부터 다시 검색
- 구현방법
	- 왼쪽 자손 노드 부터 생성 
	- 단말 노드부터 생성
		```python
		d = TreeNode('D') 
		e = TreeNode('E') 
		b = TreeNode('B', d, e) 
		f = TreeNode('F') 
		c = TreeNode('C', f) 
		root = TreeNode('A', b, c)
```

#### 이진 트리 출력방법

**1. 전위순회 출력 (Preorder traversal) : V(root)L(left)R(right)

```python
def preorder_recursive(node):
	if node is not None:
		print(node.value, end=' ')
		preorder_recursive(node.left)
		preorder_recursive(node.right)
```
![[Pasted image 20240511140018.png]]

**2. 중위순회 출력 (inorder traversal) : L(left)V(root)R(right)

```python
def inorder_recursive(node):
	if node is not None:
		inorder_recursive(node.left)
		print(node.value, end=' ')
		inorder_recursive(node.right)
```

![[Pasted image 20240511140621.png]]

**3. 후위순회 출력 (postorder traversal) : L(left)R(right)V(root)

```python
def postorder_recursive(node):
	if node is not None:
		postorder_recursive(node.left)
		postorder_recursive(node.right)
		print(node.value, end=' ')
```

![[Pasted image 20240511140823.png]]

#### 전체 노드의 수 구하기
- 왼쪽 서브 트리의 노드 수와 오른쪽 서브 트리의 노드 수의 합
```python
def count_node(node):
	if node is None:
		return 0
	else:
		return count_node(node.left) + count_node(node.right) + 1
```

#### 이진 트리의 높이 구하기
- 좌우 트리의 높이 중에서 큰 값에 1을 더한 값
```python
def calc_height(node):
    if node is None:
        return 0
    else:
        left_height = calc_height(node.left)
        right_height = calc_height(node.right)
        return max(left_height, right_height) + 1
```

#### 이진탐색트리(Binary Search Tree, BST)
- 이진 탐색 트리 속성
	- 