
---
## MYSQL

MySQL을 사용하여 데이터베이스를 조작하는 데에는 여러 가지 명령어가 있습니다. 각각의 명령어를 간단히 설명하고 주의해야 할 점을 알려드리겠습니다.

1. **CREATE**: 데이터베이스, 테이블, 뷰, 인덱스 등을 생성하는 데 사용됩니다.

   ```sql
   CREATE DATABASE my_database; -- 데이터베이스 생성
   CREATE TABLE my_table (id INT, name VARCHAR(50)); -- 테이블 생성
   ```

	### **데이터 타입(Data Types)**
	
	#### 정수(Integer)
	- **INT**: 정수형 데이터를 저장합니다. 대표적으로 32비트로 표현됩니다.
	- **BIGINT**: 대용량의 정수를 저장할 때 사용됩니다. 64비트로 표현됩니다.
	- **SMALLINT**: 작은 범위의 정수를 저장할 때 사용됩니다. 주로 16비트로 표현됩니다.
	- **TINYINT**: 아주 작은 범위의 정수를 저장할 때 사용됩니다. 주로 8비트로 표현됩니다.
	
	#### 문자열(String)
	- **VARCHAR(size)**: 가변 길이 문자열을 저장합니다. 최대 길이를 지정할 수 있습니다.
	- **CHAR(size)**: 고정 길이 문자열을 저장합니다. 지정된 크기보다 작은 문자열을 저장할 때는 공백 문자로 채워집니다.
	- **TEXT**: 대용량의 문자열을 저장할 때 사용됩니다. 최대 길이를 따로 지정하지 않습니다.
	
	#### 날짜 및 시간(DateTime)
	- **DATE**: 날짜 값을 저장합니다. 형식은 'YYYY-MM-DD'입니다.
	- **TIME**: 시간 값을 저장합니다. 형식은 'HH:MM:SS'입니다.
	- **DATETIME**: 날짜와 시간 값을 저장합니다. 형식은 'YYYY-MM-DD HH:MM:SS'입니다.
	- **TIMESTAMP**: 시간 정보를 저장합니다. 보통 현재 시간을 자동으로 기록합니다.

	### **CREATE TABLE 문법**
	
	```sql
	CREATE TABLE table_name (
	    column1_name column1_datatype [constraints],
	    column2_name column2_datatype [constraints],
	    ...
	    PRIMARY KEY (one or more columns),
	    FOREIGN KEY (column_name) REFERENCES other_table(column_name),
	    ...
	);
	```
	
	- **table_name**: 생성할 테이블의 이름입니다.
	- **column_name**: 열(컬럼)의 이름입니다.
	- **column_datatype**: 열의 데이터 타입입니다.
	- **constraints**: 열에 적용할 제약 조건입니다. 예를 들어, NOT NULL, UNIQUE, DEFAULT 등이 있습니다.
	- **PRIMARY KEY**: 테이블의 기본 키를 설정합니다. 기본 키는 해당 열의 값이 고유하게 식별될 수 있도록 합니다. (중복 x)
	- **FOREIGN KEY**: 외래 키를 설정합니다. 다른 테이블의 열을 참조하는데 사용됩니다.

---

2. **DROP**: 데이터베이스, 테이블, 뷰 등을 삭제합니다.

   ```sql
   DROP DATABASE my_database; -- 데이터베이스 삭제
   DROP TABLE my_table; -- 테이블 삭제
   ```

---

3. **INSERT**: 테이블에 새로운 레코드를 삽입합니다.
	```sql
	INSERT INTO table_name (column1, column2, column3, ...)
	VALUES (value1, value2, value3, ...);
	```
	
	- **table_name**: 데이터를 삽입할 테이블의 이름입니다.
	- **column1, column2, column3, ...**: 데이터를 삽입할 열(컬럼)의 이름입니다. 선택적으로 지정할 수 있습니다.
	- **VALUES**: 삽입할 값의 목록을 지정합니다.
	- **value1, value2, value3, ...**: 각 열에 삽입할 값입니다. 열의 순서와 값의 순서가 일치해야 합니다.
	
	### 설명
	
	- INSERT 문은 테이블에 새로운 레코드를 추가합니다. 각각의 값은 해당 열의 데이터 타입과 일치해야 합니다.
	- 열 이름을 명시하지 않으면 VALUES 문에 지정된 값들이 테이블의 열 순서대로 삽입됩니다. 그러나 열 이름을 지정하는 것이 더 명확하고 오류를 방지할 수 있습니다.

---

4. **SELECT**: 데이터를 조회합니다.

   ```sql
   SELECT * FROM my_table; -- 모든 레코드 조회
   SELECT name FROM my_table WHERE id = 1; -- 조건에 맞는 레코드 조회
   ```

	**SELECT 문법**
	
	```sql
	SELECT column1, column2, ...
	FROM table_name
	WHERE condition;
	```
	
	- **column1, column2, ...**: 조회할 열(컬럼)의 이름입니다. '*'를 사용하면 모든 열을 조회할 수 있습니다.
	- **table_name**: 데이터를 조회할 테이블의 이름입니다.
	- **condition**: 선택적으로 지정할 수 있는 조건입니다. 조건에 맞는 행만을 조회할 때 사용됩니다.
	
	### 설명
	
	- SELECT 문은 데이터베이스에서 데이터를 검색하고 반환합니다.
	- 조회할 열을 명시하고, 데이터를 검색할 테이블을 지정합니다. WHERE 절을 사용하여 특정 조건을 지정하여 원하는 데이터만을 검색할 수 있습니다.
	
	예시:
	
	```sql
	SELECT * FROM employees;
	```
	위 예제에서는 'employees' 테이블의 모든 열을 조회합니다.
	
	```sql
	SELECT name FROM employees WHERE id = 1;
	```
	위 예제에서는 'employees' 테이블에서 'id'가 1인 레코드의 'name' 열만을 조회합니다.

---


5. **UPDATE**: 테이블의 레코드를 수정합니다.

   ```sql
   UPDATE my_table SET name = 'Jane' WHERE id = 1; -- 레코드 수정
   ```

	 **UPDATE 문법**
	
	```sql
	UPDATE table_name
	SET column1 = value1, column2 = value2, ...
	WHERE condition;
	```
	
	- **table_name**: 레코드를 수정할 테이블의 이름입니다.
	- **column1 = value1, column2 = value2, ...**: 수정할 열과 해당하는 값을 지정합니다.
	- **condition**: 선택적으로 지정할 수 있는 조건입니다. 조건에 맞는 레코드만을 수정할 때 사용됩니다.
	
	### 설명
	
	- UPDATE 문은 테이블의 레코드를 수정합니다.
	- SET 절에서는 수정할 열과 해당하는 값을 지정합니다. WHERE 절을 사용하여 특정 조건을 지정하여 원하는 레코드만을 수정할 수 있습니다.
	
	예시:
	
	```sql
	UPDATE employees SET name = 'Jane' WHERE id = 1;
	```
	위 예제에서는 'employees' 테이블에서 'id'가 1인 레코드의 'name' 열 값을 'Jane'으로 수정합니다.

---

6. **DELETE**: 테이블에서 레코드를 삭제합니다.

   ```sql
   DELETE FROM my_table WHERE id = 1; -- 레코드 삭제
   ```

	### **DELETE 문법**
	
	```sql
	DELETE FROM table_name
	WHERE condition;
	```
	
	- **table_name**: 레코드를 삭제할 테이블의 이름입니다.
	- **condition**: 선택적으로 지정할 수 있는 조건입니다. 조건에 맞는 레코드만을 삭제할 때 사용됩니다.
	
	### 설명
	
	- DELETE 문은 테이블에서 레코드를 삭제합니다.
	- WHERE 절을 사용하여 특정 조건을 지정하여 해당 조건에 맞는 레코드만을 삭제할 수 있습니다. 조건을 생략하면 테이블의 모든 레코드가 삭제됩니다.
	
	예시:
	
	```sql
	DELETE FROM employees WHERE id = 1;
	```
	위 예제에서는 'employees' 테이블에서 'id'가 1인 레코드를 삭제합니다.

---
