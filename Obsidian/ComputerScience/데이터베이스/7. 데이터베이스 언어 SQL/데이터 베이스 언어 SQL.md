
---
## 1. SQL의 소개
#### SQL의 분류

```SQL
create 생성
alter 변경
drop 삭제
```
- 데이터 정의어(DDL) 
	- 테이블을 **생성**하고 **변경,삭제**하는 기능을 제공 
```sql
insert 삽입
update 수정
delete 삭제
select 검색
```
- 데이터 조작어(DML) 
	- 테이블에 새 데이터를 **삽입**하거나, 테이블에 저장된 데이터를 **수정,삭제,검색** 하는 기능을 제공 
```sql
revoke grant commit rollback
```
- 데이터 제어어(DCL) 
	- 보안을 위해 데이터에 대한 접근 및 사용 권한을 사용자별로 부여하거나 취소 하는 기능을 제공

#### ==아마 시험 ?==
![](../../../../image/Pasted%20image%2020241007173812.png)
## 2. SQL을 이용한 데이터 정의
#### ==테이블 생성 : CREATE TABLE 문 (암기)==
```SQL
CREATE TABLE 테이블_이름 (
	속성_이름 데이터_타입 [NOT NULL] [DEFAULT 기본_값]
	[PRIMARY KEY (속성_리스트)]
	[UNIQUE (속성_리스트)]
	[FOREIGN KEY (속성_리스트) REFERENCES 테이블_이름(속성_리스트)]
	[ON DELETE 옵션] [ON UPDATE 옵션]
	[CONSTRANINT 이름] [CHECK(조건)]
);
```
-  [ ]의 내용은 생략이 가능 
- SQL 문은 세미콜론(;)으로 문장의 끝을 표시 
- SQL 문은 대소문자를 구분하지 않음
- **속성의 정의** 
	- 테이블을 구성하는 각 속성의 데이터 타입을 선택한 후에 널 값 허용 여부와 기본 값 필요 여부를 결정 
	- NOT NULL 
		- 속성이 널 값을 허용하지 않음을 의미하는 키워드 
		- 예) 고객아이디 VARCHAR(20) NOT NULL 
	- DEFAULT 
		- 속성의 기본 값을 지정하는 키워드 
		- 예) 적립금 INT DEFAULT 0 
		- 예) 담당자 VARCHAR(10) DEFAULT ‘방경아'
- 속성의 데이터 타입
	![](../../../../image/Pasted%20image%2020241010104327.png)
- **키의 정의** 
	- PRIMARY KEY 
		- 기본키를 지정하는 키워드 
		- 예) PRIMARY KEY(고객아이디) 
		- 예) PRIMARY KEY(주문고객, 주문제품) 
	- UNIQUE 
		- 대체키를 지정하는 키워드 
		- 대체키로 지정된 속성의 값은 유일성을 가지며, 기본키와 달리 널 값이 허용됨 
		- 예) UNIQUE(고객이름)
	- FOREIGN KEY 
		- 외래키를 지정하는 키워드 
		- 외래키가 어떤 테이블의 무슨 속성을 참조하는지 REFERENCES 키워드 다음에 제시 
		- 참조 무결성 제약조건 유지를 위해 참조되는 테이블에서 투플 삭제 시 처리 방법을 지정하는 옵션 
			- ON DELETE NO ACTION : 투플을 삭제하지 못하게 함 
			- ON DELETE CASCADE : 관련 투플을 함께 삭제함 
			- ON DELETE SET NULL : 관련 투플의 외래키 값을 NULL로 변경함 
			- ON DELETE SET DEFAULT : 관련 투플의 외래키 값을 미리 지정한 기본 값으로 변경함
			- ON UPDATE NO ACTION : 투플을 변경하지 못하게 함 
			- ON UPDATE CASCADE : 관련 투플에서 외래키 값을 함께 변경함 
			- ON UPDATE SET NULL : 관련 투플의 외래키 값을 NULL로 변경함 
			- ON UPDATE SET DEFAULT : 관련 투플의 외래키 값을 미리 지정한 기본 값으로 변경함 
		- 예) FOREIGN KEY(소속부서) REFERENCES 부서(부서번호) 
		- 예) FOREIGN KEY(소속부서) REFERENCES 부서(부서번호) ON DELETE CASCADE ON UPDATE CASCADE
- **데이터 무결성 제약조건의 정의** 
	- CHECK 
		- 테이블에 정확하고 유효한 데이터를 유지하기 위해 특정 속성에 대한 제약조건을 지정 
		- CONSTRAINT 키워드와 함께 고유의 이름을 부여할 수도 있음 
		- 예) CHECK(재고량 >= 0 AND 재고량 <= 10000) 
		- 예) CONSTRAINT CHK_CPY CHECK(제조업체 = ‘한빛제과’)

#### 참조 무결성 제약조건 유지를 위한 투플 삭제 예
![](../../../../image/Pasted%20image%2020241010104711.png)
- ON DELETE NO ACTION : 부서 테이블의 투플을 삭제하지 못하게 함 
- ON DELETE CASCADE : 사원 테이블에서 홍보부에 근무하는 정소화 사원 투플도 함께 삭제 
- ON DELETE SET NULL : 사원 테이블에서 정소화 사원의 소속부서 속성 값을 NULL로 변경 
- ON DELETE SET DEFAULT : 사원 테이블에서 정소화 사원의 소속부서 속성 값을 기본 값으로 변경
#### 고객 테이블 생성을 위한 CREATE TABLE 문 작성 예
![](../../../../image/Pasted%20image%2020241010104947.png)
```sql
CREATE TABLE 고객 (
	고객아이디 VARCHAR(20) NOT NULL,
	고객이름 VARCHAR(10) NOT NULL,
	나이 INT,
	등급 VARCHAR(10) NOT NULL,
	직업 VARCHAR(10),
	적립금 INT DEFAULT 0,
	PRIMARY KEY(고객아이디)
);
```

![](../../../../image/Pasted%20image%2020241010105005.png)
```sql
CREATE TABLE 제품 (
	제품번호 CHAR(3) NOT NULL,
	제품명 VARCHAR(20),
	재고량 INT,
	단가 INT,
	제조업체 VARCHAR(20),
	PRIMARY KEY(제품번호),
	CHECK (재고량 >=0 AND 재고량 <=10000)
);
```

![](../../../../image/Pasted%20image%2020241010105048.png)
```sql
CREATE TABLE 주문 (
	주문번호 CHAR(3) NOT NULL,
	주문고객 VARCHAR(20),
	주문제품 CHAR(3),
	수량 INT,
	배송지 VARCHAR(30),
	주문일자 DATETIME,
	PRIMARY KEY(주문번호),
	FOREIGN KEY(주문고객) REFERENCES 고객(고객아이디),
	FOREIGN KEY(주문제품) REFERENCES 제품(제품번호)
);
```

![](../../../../image/Pasted%20image%2020241010105104.png)
```sql
CREATE TABLE 배송업체 (
	업체번호 CHAR(3) NOT NULL,
	업체명 VARCHAR(20),
	주소 VARCHAR(100),
	전화번호 VARCHAR(20),
	PRIMARY KEY(업체번호)
);
```
####  테이블 변경 : ALTER TABLE 문
```SQL
ALTER TABLE 테이블이름
	[ADD 속성이름 데이터타입]
	[DROP COLUMN 속성이름]
	[ALTER COLUMN 속성이름 데이터타입]
	[ALTER COLUMN 속성이름 [NULL|NOT NULL]]
	[ADD PRIMARY KEY(속성이름)]
	[[ADD|DROP] 제약이름]
```
---
- 새로운 속성 추가
```SQL
ALTER TABLE 테이블_이름
	ADD 속성_이름 데이터_타입 [NOT NULL] [DEFAULT 기본_값];
```
![](../../../../image/Pasted%20image%2020241010111745.png)
```SQL
ALTER TABLE 고객 ADD 가입날짜 DATETIME;
```
---
- 기존 속성 삭제
```SQL
ALTER TABLE 테이블_이름 DROP COLUMN 속성_이름;
```
- 만약, 삭제할 속성과 관련된 제약조건이나 참조하는 다른 속성이 존재한다면? 
	- 속성 삭제가 수행되지 않음 
	- 관련된 제약조건이나 참조하는 다른 속성을 먼저 삭제해야 함

![](../../../../image/Pasted%20image%2020241010112120.png)
```SQL
ALTER TABLE 고객 DROP COLUMN 가입날짜;
```
---
- 기존 제약조건의 삭제
```SQL
ALTER TABLE 테이블
```

---
## ==3. SQL을 이용한 데이터 조작(시험)==
#### 데이터 검색 : SELECT 문 
```SQL
SELECT [ALL; DISTINCT] 속성_리스트
FROM 테이블_리스트;
```
- 기본 검색 
	- SELECT 키워드와 함께 검색하고 싶은 속성의 이름을 나열 
	- FROM 키워드와 함께 검색하고 싶은 속성이 있는 테이블의 이름을 나열 
	- 검색 결과는 테이블 형태로 반환됨
	- ALL 
		- 결과 테이블이 투플의 중복을 허용하도록 지정, 생략 가능
	- DISTINCT 
		- 결과 테이블이 투플의 중복을 허용하지 않도록 지정
	- AS 키워드를 이용해 결과 테이블에서 속성의 이름을 바꾸어 출력 가능 
		- **새로운 이름에 공백이 포함되어 있으면 큰따옴표나 작은따옴표로 묶어주어야 함** 
			- 오라클에서는 큰따옴표, MS SQL 서버에서는 작은따옴표 사용 
		- AS 키워드는 생략 가능
			- 조건 : **공백이 있으면 안된다**
	- 산술식을 이용한 검색 
		- SELECT 키워드와 함께 산술식 제시 
			- 산술식: 속성의 이름과 +, -, *, / 등의 산술 연산자와 상수로 구성 
		- 결과 테이블에서만 계산된 결과 값이 출력됨 
			- 속성의 값이 실제로 변경되는 것은 아님

![](../../../../image/Pasted%20image%2020241010112625.png)
```SQL
SELECT 고객아이디, 고객이름, 등급 FROM 고객;
```
![](../../../../image/Pasted%20image%2020241010112720.png)
```SQL
SELECT * FROM 고객;
```
- 모든 속성을 검색할 때는 속성의 이름을 전부 나열하지 않고 * 사용 가능

![](../../../../image/Pasted%20image%2020241010112812.png)
```SQL
SELECT ALL 제조업체 FROM 제품;
```
- 결과 테이블에서 제조업체가 중복 됨

![](../../../../image/Pasted%20image%2020241010112928.png)
```SQL
SELECT DISTINCT 제조업체 FROM 제품;
```
- 결과 테이블에서 제조업체가 한 번씩만 나타남

![](../../../../image/Pasted%20image%2020241010113120.png)
```SQL
SELECT 제품명, 단가 AS 가격 FROM 제품;
```
![](../../../../image/Pasted%20image%2020241010113356.png)
```SQL
SELECT 제품명, 단가 +500 AS "조정 단가" FROM 제품;
```
---
#### 데이터 검색 : SELECT 문 (조건 검색)
```SQL
SELECT [ALL; DISTINCT] 속성_리스트
FROM 테이블_리스트
[WHERE 조건];
```
- 조건 검색 
	- 조건을 만족하는 데이터만 검색
	- WHERE 키워드와 함께 비교 연산자와 논리 연산자를 이용한 검색 조건 제시 
		- 숫자뿐만 아니라 문자나 날짜 값을 비교하는 것도 가능 
			- 예) ‘A’ < ‘C’ 
			- 예) ‘2022-12-01’ < ‘2022-12-02’ 
		- 조건에서 문자나 날짜 값은 작은따옴표로 묶어서 표현

![](../../../../image/Pasted%20image%2020241010113729.png)
```SQL
SELECT 제품명, 재고량, 단가 FROM 제품 WHERE 제조업체='한빛제과';
```
![](../../../../image/Pasted%20image%2020241010113822.png)
```SQL
SELECT 주문제품, 수량, 주문일자 
FROM 주문 
WHERE 주문고객='apple' AND 수량>=15;
```
![](../../../../image/Pasted%20image%2020241010113932.png)
```sql
SELECT 주문제품, 수량, 주문일자, 주문고객 
FROM 주문 
WHERE 주문고객='apple' OR 수량>=15;
```
![](../../../../image/Pasted%20image%2020241010114055.png)
```sql
SELECT 제품명, 단가, 제조업체 FROM 제품 WHERE 단가>=2000 AND 단가<=3000;
```
---
- LIKE를 이용한 검색
	- LIKE 키워드를 이용해 **부분적으로 일치하는 데이터** 검색 
	- 문자열을 이용하는 조건에만 LIKE 키워드 사용 가능

![](../../../../image/Pasted%20image%2020241014163042.png)
![](../../../../image/Pasted%20image%2020241014163024.png)
![](../../../../image/Pasted%20image%2020241015155755.png)
```sql
SELECT 고객이름, 나이, 등급, 적립금
FROM 고객
WHERE 고객이름 LIKE '김%';
```
![](../../../../image/Pasted%20image%2020241015155857.png)
```SQL
SELECT 고객아이디, 고객이름, 등급
FROM 고객
WHERE 고객아이디 LIKE '_____'; /* Access 에서는 '?????'
```

---
- NULL을 이용한 검색
	- IS NULL 키워드를 이용해 특정 속성의 값이 널 값인지를 비교 
	- IS NOT NULL 키워드를 이용해 특정 속성의 값이 널 값이 아닌지를 비교 
	- 검색 조건에서 널 값은 다른 값과 크기를 비교하면 결과가 모두 거짓이 됨

![](../../../../image/Pasted%20image%2020241015160037.png)
```SQL
SELECT 고객이름
FROM 고객
WHERE 나이 IS NULL;
```
![](../../../../image/Pasted%20image%2020241015160113.png)
```SQL
SELECT 고객이름
FROM 고객
WHERE 나이 IS NOT NULL;
```
---
```SQL
SELECT [ ALL ; DISTINCT ] 속성_리스트
FROM 테이블_리스트
[ WHERE 조건 ]
[ ORDER BY 속성_리스트 [ ASC ; DESC ]]
```
- 정렬 검색 
	- ORDER BY 키워드를 이용해 결과 테이블 내용을 사용자가 원하는 순서로 출력 
	- ORDER BY 키워드와 함께 정렬 기준이 되는 속성과 정렬 방식을 지정 
		-  오름차순(기본): ASC / 내림차순: DESC 
		-  널 값은 오름차순에서는 맨 마지막에 출력되고, 내림차순에서는 맨 먼저 출력됨 
		-  여러 기준에 따라 정렬하려면 정렬 기준이 되는 속성들을 차례대로 제시

![](../../../../image/Pasted%20image%2020241015160255.png)
```SQL
SELECT 고객이름, 등급, 나이
FROM 고객
ORDER BY 나이 DESC; /* DESC : 내림차순 */
```
![](../../../../image/Pasted%20image%2020241015160349.png)
```SQL
SELECT 주문고객, 주문제품, 수량, 주문일자
FROM 주문
WHERE 수량 >= 10
ORDER BY 주문제품 ASC, 수량 DESC; /* ASC : 오름차순, DESC : 내림차순 */
```
- P01 제품이 맨 먼저 출력되고, P03 제품 중에는 수량이 22인 제품이 먼저 출력됨됨
---
![](../../../../image/Pasted%20image%2020241014165859.png)
- 집계 함수를 이용한 검색
	- 집계 함수 사용 시 주의 사항 
		- 집계 함수는 널인 속성 값은 제외하고 계산함 
		- 집계 함수는 WHERE 절에서는 사용할 수 없고, SELECT 절이나 HAVING 절에서만 사용 가능

![](../../../../image/Pasted%20image%2020241015160558.png)
```SQL
SELECT AVG(단가) FROM 제품; /* AVG : 속성 값의 평균 */
```
![](../../../../image/Pasted%20image%2020241015160652.png)
```SQL
SELECT SUM(재고량) AS '재고량 합계'
FROM 제품
WHERE 제조업체 = '한빛제과'; /* SUM : 속성 값의 합계 , AS : 속성명 지정 */
```
![](../../../../image/Pasted%20image%2020241015160833.png)
```SQL
/* 1. 고객아이디 속성을 이용해 계산하는 경우 */
SELECT COUNT(고객아이디)  AS 고객수 FROM 고객;
/* 2. 나이 속성을 이용해 계산하는 경우 */
SELECT COUNT(나이) AS 고객수 FROM 고객;
/* 3. *를 이용해 계산하는 경우 */
SELECT COUNT(*) AS 고객수 FROM 고객;
```
- 널인 속성 값은 제외하고 개수 계산
- 정확한 개수를 계산하기 위해서는 보통 **기본키 속성이나 별을 주로 사용**

![](../../../../image/Pasted%20image%2020241015161105.png)
```SQL
SELECT COUNT(DISTINCT 제조업체) AS '제조업체 수' FROM 제품;
```
- DISTNCT 키워드를 이용해 중복을 없애고 서로 다른 제조업체의 개수만 계산

---
```sql
SELECT [ALL;DISTINCT] 속성_리스트
FROM 테이블_리스트
[WHERE 조건]
[GROUP BY 속성_리스트[HAVING 조건]]
[ORDER BY 속성_리스트[ASC;DESC]];
```
- 그룹별 검색 
	- GROUP BY 키워드를 이용해 특정 속성의 값이 같은 투플을 모아 그룹을 만들고, 그룹별로 검색 
		- – GROUP BY 키워드와 함께 그룹을 나누는 기준이 되는 속성을 지정 
	- HAVING 키워드와 함께 그룹에 대한 조건 작성 가능 
	- 그룹을 나누는 기준이 되는 속성을 SELECT 절에도 작성하는 것이 좋음

![](../../../../image/Pasted%20image%2020241015161222.png)
```SQL
SELECT 주문제품, SUM(수량) AS 총주문수량
FROM 주문
GROUP BY 주문제품;
```
- 그룹을 나누는 기준이 되는 '주문제품' 속성을 SELECT 절에도 작성하는 것이 좋음
- 동일 제품을 주문한 투플을 모아 그룹으로 만들고, 그룹별로 수량의 합계를 계산

![](../../../../image/Pasted%20image%2020241015162146.png)
```SQL
SELECT 제조업체, COUNT(*) AS 제품수, MAX(단가) AS 최고가
FROM 제품
GROUP BY 제조업체;
```
![](../../../../image/Pasted%20image%2020241015162306.png)
```SQL
SELECT 제조업체, COUNT(*) AS 제품수, MAX(단가) AS 최고가
FROM 제품
GROUP BY 제조업체 HAVING COUNT(*) >= 3;
```
- 집계 함수를 이용한 조건은 WHERE 절에는 작성할 수 없고, HAVING 절에서 작성 가능

![](../../../../image/Pasted%20image%2020241015162438.png)
```SQL
SELECT 등급, COUNT(*) AS 고객수, AVG(적립금) AS 평균적립금
FROM 고객
GROUP BY 등급 HAVING AVG(적립금) >= 1000;
```
![](../../../../image/Pasted%20image%2020241015162538.png)
```SQL
SELECT 주문제품, 주문고객, SUM(수량) AS 총주문수량
FROM 주문
GROUP BY 주문제품, 주문고객;
```
- 집계 함수나 GROUP BY 절에 명시된 속성 외의 속성은 SELECT 절에 작성 불가
---
- 여러 테이블에 대한 조인 검색 
	- 조인 검색: 여러 개의 테이블을 연결하여 데이터를 검색하는 것 
	- 조인 속성: 조인 검색을 위해 테이블을 연결해주는 속성 
		- 연결하려는 테이블 간에 조인 속성의 이름은 달라도 되지만 도메인은 같아야 함 
		- 일반적으로 외래키를 조인 속성으로 이용함 
	- FROM 절에 검색에 필요한 모든 테이블을 나열 
	- WHERE 절에 조인 속성의 값이 같아야 함을 의미하는 조인 조건을 제시 
	- 속성 이름 앞에 해당 속성이 소속된 테이블의 이름을 표시 
		- 같은 이름의 속성이 서로 다른 테이블에 존재할 수 있기 때문 
		- 예) 주문.주문고객
	- 표준 SQL에서는 INNER JOIN과 ON 키워드를 이용해 작성하는 방법도 제공
```SQL
SELECT 속성_리스트
FROM 테이블1 INNER JOIN 테이블2 ON 조인조건
[ WHERE 검색조건 ]
```

![](../../../../image/Pasted%20image%2020241015162656.png)
```SQL
SELECT 제품.제품명
FROM 제품, 주문
WHERE 주문.주문고객 ='banana' AND 제품.제품번호 = 주문.주문제품;
```
![](../../../../image/Pasted%20image%2020241015162841.png)
```SQL
SELECT 주문.주문제품, 주문.주문일자
FROM 고객, 주문
WHERE 고객.나이 >= 30 AND 고객.고객아이디 = 주문.주문고객;
/* 다른 방법 */
SELECT o.주문제품, o.주문일자
FROM 고객 c, 주문 o
WHERE c.나이 >= 30 AND c.고객아이디 = o.주문고객;
```
- FROM 절에서 테이블의 이름을 대신하는 단순한 별명을 제시하여 질의문을 작성하는 것도 좋음

![](../../../../image/Pasted%20image%2020241015163302.png)
```SQL
SELECT 제품.제품명
FROM 고객, 제품, 주문
WHERE 고객.고객이름 ='고명석' 
AND 고객.고객아이디 = 주문.주문고객 
AND 제품.제품번호 = 주문.주문제품;
```
---
```SQL
SELECT 속성_리스트
FROM 테이블1 LEFT; RIGHT; FULL OUTER JOIN 테이블2 ON 조인조건
[ WHERE 검색조건]
```
- 외부 조인 검색
	- 조인 조건을 만족하지 않는 투플에 대해서도 검색을 수행 
	- OUTER JOIN과 ON 키워드를 이용해 작성
	- 분류 
		- 모든 투플을 검색 대상으로 하는 테이블이 무엇이냐에 따라 분류 
		- 왼쪽 외부 조인, 오른쪽 외부 조인, 완전 외부 조인

---
![[Pasted image 20241017103926.png]]
- 부속 질의문을 이용한 검색
	- SELECT 문 안에 또 다른 SELECT 문을 포함하는 **질의**
		- 상위 질의문(주 질의문): 다른 SELECT 문을 포함하는 SELECT 문 
		- 부속 질의문(서브 질의문): 다른 SELECT 문 안에 들어 있는 SELECT 문
			- 괄호로 묶어서 작성, ORDER BY 절을 사용할 수 없음 
			- 단일 행 부속 질의문: 하나의 행을 결과로 반환 
			- 다중 행 부속 질의문: 하나 이상의 행을 결과로 반환 
	- 부속 질의문을 먼저 수행하고, 그 결과를 이용해 상위 질의문 수행 
	- 부속 질의문과 상위 질의문을 연결하는 연산자 필요 
		- 단일 행 부속 질의문은 비교 연산자(=, <>, >, >=, <, <=) 사용 가능 
		- 다중 행 부속 질의문은 비교 연산자 사용 불가

![[Pasted image 20241017103944.png]]
```sql
SELECT 제품명, 단가
FROM 제품
/*상위 질의문과 부속 질의문*/
WHERE 제조업체 = (SELECT 제조업체 FROM 제품 WHERE 제품명 ='달콤비스킷');
```
- ‘달콤비스킷’의 제조업체는 ‘한빛제과’만 존재 → **단일 행 부속 질의문 (비교 연산자 = 이용)**

![[Pasted image 20241017104135.png]]
```SQL
SELECT 고객이름, 적립금
FROM 고객
WHERE 적립금 = (SELECT MAX(적립금) FROM 고객);
```
- 최대 적립금은 단일 값이므로 단일 행 부속 질의문 **(비교 연산자 = 이용)**

![[Pasted image 20241017104237.png]]
```SQL
SELECT 제품명, 제조업체
FROM 제품
WHERE 제품번호 IN (SELECT 주문제품 FROM 주문 WHERE 주문고객 ='banana');
```
- ‘banana’ 고객이 주문한 제품은 여러 개이므로 → **다중 행 부속 질의문 (IN 연산자 이용)**

![[Pasted image 20241017104504.png]]
```SQL
SELECT 제품명, 제조업체
FROM 제품
WHERE 제품번호 NOT IN (SELECT 주문제품 FROM 주문 WHERE 주문고객 ='banana');
```
- 부속 질의문의 결과 값 중에서 일치하는 것이 없어야 조건이 참이 되는 **NOT IN** 연산자 이용

![[Pasted image 20241017105135.png]]
```SQL
SELECT 제품명, 단가, 제조업체
FROM 제품
WHERE 단가 > ALL (SELECT 단가 FROM 제품 WHERE 제조업체='대한식품');
/* ALL과 반대의 개념으로 ANY */
WHERE 단가 > ANY (SELECT 단가 FROM 제품 WHERE 제조업체='대한식품');
```
- 대한식품이 제조한 제품은 단가가 4,500원인 그냥만두와 1,200원인 얼큰라면

![[Pasted image 20241017105753.png]]
```SQL
SELECT 고객이름
FROM 고객
WHERE EXISTS (SELECT * FROM 주문
			 WHERE 주문일자 ='2022-03-15'
			 AND 주문.주문고객 = 고객.고객아이디);
/* EXISTS의 반대 개념 NOT EXISTS*/
```
- Access에서 날짜를 입력할 때 : #2022-03-15 이라 입력해야함
- AND 주문고객 = 고객아이디 생략도 가능 함

---
#### 데이터 삽입 : INSERT 문
```SQL
INSERT
INTO 테이블_이름[(속성_리스트)]
VALUES (속성값_리스트);
```
- 데이터 직접 삽입 
	- 테이블에 투플을 직접 삽입
	- INTO 키워드와 함께 투플을 삽입할 테이블의 이름과 속성의 이름을 나열 
		- 속성 리스트를 생략하면 테이블을 정의할 때 지정한 속성의 순서대로 값이 삽입됨 
	- VALUES 키워드와 함께 삽입할 속성 값들을 나열 
	- **INTO 절의 속성 이름과 VALUES 절의 값은 순서대로 일대일 대응되어야 함**

![[Pasted image 20241017110724.png]]
```SQL
INSERT
INTO 고객(고객아이디, 고객이름, 나이, 등급, 직업, 적립금)
VALUES ('strawberry', '최유경', 30, 'vip', '공무원', 100);
/* 테이블의 속성리스트는 생략가능 */
INSERT
INTO 고객
VALUES ('strawberry', '최유경', 30, 'vip', '공무원', 100);
```

![[Pasted image 20241017111052.png]]
```SQL
INSERT 
INTO 고객(고객아이디, 고객이름, 나이, 등급, 적립금)
VALUES ('tomato', '정은심', 36, 'gold', 4000);
/* 속성리스트를 생략할때는 null값을 넣어줘야 한다 */
INSERT
INTO 고객
VALUES ('tomato', '정은심', 36, 'gold', NULL, 4000);
```
- 직업 속성에 널(NULL) 값을 삽입

---
```SQL
INSERT
INTO 테이블_이름[(속성_리스트)]
SELECT 문;
```
- SELECT 문을 이용해 다른 테이블에서 검색한 데이터를 삽입

```SQL
CREATE TABLE 한빛제품 AS SELECT * FROM 제품
WHERE 1 = 2 ; /* 조건문을 틀리게 작성해서 한빛제품의 구조만 복사함*/
```
- WHERE절 전까지는 전체복사
- WHERE절을 틀리게 작성해 구조만 복사

```SQL
INSERT
INTO 한빛제품(제품번호, 제품명, 재고량, 단가)
SELECT 제품번호, 제품명, 재고량, 단가
FROM 제품
WHERE 제조업체='한빛제과';
```
- SELECT 문을 이용해 다른 테이블에서 검색한 데이터를 삽입
- 한빛제과에서 제조한 제품의 제품명, 재고량, 단가를 제품 테이블에서 검색하여 한빛제품 테이블에 삽입

---
####  데이터 수정 : UPDATE 문
```SQL
UPDATE 테이블_이름
SET 속성_이름1 = 값1, ...
[WHERE 조건]
```
- 테이블에 저장된 투플에서 특정 속성의 값을 수정
- SET 키워드 다음에 속성 값을 어떻게 수정할 것인지를 지정 
- WHERE 절에 제시된 조건을 만족하는 투플만 속성 값을 수정 
	- WHERE 절을 생략하면 테이블에 존재하는 모든 투플을 대상으로 수정

---
#### 데이터 삭제 : DELETE 문(UPDATE문과 구조가 다르니 꼭 기억)
```SQL
DELETE
FROM 테이블_이름
[WHERE 조건];
```
- 테이블에 저장된 데이터를 삭제
- WHERE 절에 제시한 조건을 만족하는 투플만 삭제 
	- WHERE 절을 생략하면 테이블에 존재하는 모든 투플을 삭제해 빈 테이블이 됨

---
관계대수 SQL(SELECT?) 문장 

주관식으로 나옴
결과 테이블을 그려라 할수도 있음
테이블은 교제에 있는 테이블을 시험지에 줄거임


OX문제
4지 선다

책을 외워야 함
책의 문제와 조금 다르게 냄
한문제는 다른 테이블을 주어짐

월요일 4시 시험 60분 