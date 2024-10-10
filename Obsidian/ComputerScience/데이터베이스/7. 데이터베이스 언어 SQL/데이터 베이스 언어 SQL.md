
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
![](../../../../image/Pasted%20image%2020241010112812.png)
```SQL
SELECT ALL 제조업체 FROM 제품;
```
![](../../../../image/Pasted%20image%2020241010112928.png)
```SQL
SELECT DISTINCT 제조업체 FROM 제품;
```
![](../../../../image/Pasted%20image%2020241010113120.png)
```SQL
SELECT 제품명, 단가 AS 가격 FROM 제품;
```
![](../../../../image/Pasted%20image%2020241010113356.png)
```SQL
SELECT 제품명, 단가 +500 AS "조정 단가" FROM 제품;
```
---
#### 조건 검색
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
SELECT 주문제품, 수량, 주문일자 FROM 주문 WHERE 주문고객='apple' AND 수량>=15;
```
![](../../../../image/Pasted%20image%2020241010113932.png)
```sql
SELECT 주문제품, 수량, 주문일자, 주문고객 FROM 주문 WHERE 주문고객='apple' OR 수량>=15;
```
![](../../../../image/Pasted%20image%2020241010114055.png)
```sql
SELECT 제품명, 단가, 제조업체 FROM 제품 WHERE 단가>=2000 AND 단가<=3000;
```
