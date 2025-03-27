
---
### **1. Scanner 클래스 개요**

- **패키지**: `java.util.Scanner`
- **역할**: 다양한 데이터 소스로부터 입력을 읽고, 이를 원하는 데이터 타입으로 파싱(Parsing)하는 클래스.
- **주요 특징**:
    - 정규식을 사용하여 입력을 분리.
    - 다양한 데이터 타입(`int`, `float`, `String` 등)으로 변환 가능.
    - 키보드 입력, 파일, 문자열 등 다양한 입력 소스를 지원.

---

### **2. 주요 생성자**

```java
// System.in으로 키보드 입력 받기
Scanner scanner = new Scanner(System.in);

// 파일로부터 입력 받기
File file = new File("input.txt");
Scanner scanner = new Scanner(file);

// 문자열을 소스로 사용
String data = "42 3.14 Hello";
Scanner scanner = new Scanner(data);
```

---

### **3. 주요 메서드**

#### **a) 입력 확인 메서드**

- 입력 데이터를 확인하는 데 사용하며, 반환값은 `boolean`:
    - `hasNext()`: 다음 입력이 존재하는지 확인.
    - `hasNextInt()`, `hasNextDouble()` 등: 다음 입력이 특정 타입으로 변환 가능한지 확인.

#### **b) 데이터 읽기 메서드**

- 데이터 타입에 맞는 메서드 사용:
    - `next()`: 다음 단어(공백으로 구분된 문자열) 읽기.
    - `nextLine()`: 한 줄 전체 읽기.
    - `nextInt()`, `nextDouble()`: 입력 값을 정수 또는 실수로 변환.
- **예시**:
    
    ```java
    Scanner scanner = new Scanner("100 3.14 Hello");
    int a = scanner.nextInt();  // 100
    double b = scanner.nextDouble();  // 3.14
    String c = scanner.next();  // "Hello"
    ```
    

---

### **4. 정규식과 Scanner**

`Scanner`는 정규식을 사용해 입력을 구분하거나 특정 패턴에 따라 데이터를 처리합니다.

- **기본 구분자**: 공백, 탭, 줄바꿈 등.
- **사용자 정의 구분자**: `useDelimiter()` 메서드로 설정 가능.
    
    ```java
    Scanner scanner = new Scanner("apple,banana,grape");
    scanner.useDelimiter(",");
    while (scanner.hasNext()) {
        System.out.println(scanner.next());
    }
    // 출력:
    // apple
    // banana
    // grape
    ```
    

---

### **5. 내부 동작**

- `Scanner`는 내부적으로 입력 소스를 **`Readable` 인터페이스**를 통해 처리하며, 입력을 **버퍼**로 읽고 데이터를 분석합니다.
- 입력 데이터를 분리하기 위해 정규식을 사용하며, 데이터를 분석 후 파싱합니다.

---

### **6. 한계 및 단점**

1. **버퍼링 문제**:
    
    - `Scanner`는 입력을 줄 단위로 버퍼링하지 않으므로 속도가 느릴 수 있습니다.
    - 대량의 데이터를 처리할 때는 `BufferedReader`를 사용하는 것이 더 효율적입니다.
2. **입력 에러 처리**:
    
    - 부적절한 입력 값이 주어지면 예외(`InputMismatchException`)가 발생합니다.
        
        ```java
        Scanner scanner = new Scanner("abc");
        int num = scanner.nextInt();  // 예외 발생
        ```
        
    - 이를 방지하려면 `hasNextInt()` 등을 사용하여 확인 후 처리해야 합니다.
3. **동기화 문제**:
    
    - `Scanner`는 멀티스레드 환경에서 안전하지 않습니다. 동기화가 필요한 경우 별도의 처리 필요.
4. **구분자 변경의 영향**:
    
    - 구분자를 변경하면 기본적인 데이터 읽기가 어려워질 수 있습니다.

---

### **7. Scanner와 BufferedReader 비교**

|특징|Scanner|BufferedReader|
|---|---|---|
|**구분자 사용**|공백(기본) 또는 사용자 정의 정규식|라인 단위로 입력 읽기|
|**데이터 타입 파싱**|직접 지원 (`nextInt()`, `nextDouble()`)|파싱은 추가 작업 필요 (`Integer.parseInt()`)|
|**성능**|느림|빠름 (버퍼링 최적화)|
|**사용 편의성**|쉬움|상대적으로 복잡|

---

### **8. 사용 사례**

1. **파일 처리**:
    
    ```java
    Scanner scanner = new Scanner(new File("input.txt"));
    while (scanner.hasNextLine()) {
        System.out.println(scanner.nextLine());
    }
    ```
    
2. **콘솔 입력**:
    
    ```java
    Scanner scanner = new Scanner(System.in);
    System.out.print("Enter your name: ");
    String name = scanner.nextLine();
    System.out.println("Hello, " + name + "!");
    ```
    
3. **데이터 파싱**:
    
    ```java
    String data = "10,20,30";
    Scanner scanner = new Scanner(data);
    scanner.useDelimiter(",");
    while (scanner.hasNextInt()) {
        System.out.println(scanner.nextInt());
    }
    ```
    

---

### **9. 결론**

`Scanner`는 간단한 입력 작업에 적합한 도구로, 사용 편의성과 다양한 기능을 제공합니다. 하지만 대규모 데이터 처리나 성능이 중요한 경우, 더 최적화된 도구(예: `BufferedReader`, `Stream API`)를 고려해야 합니다. **입력 소스와 요구 사항에 따라 적절한 도구를 선택하는 것이 중요합니다.**