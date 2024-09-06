

---
## QUIZ.1

### HTML 부분

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>오늘 실습</title>
    <style>
        body {
            padding: 20px;
        }
        h1 {
            color: blue;
        }
    </style>
</head>
<body>
    <h1>영어 퀴즈</h1>
    고양이 영어로? <input type="text" id="a1"> <br><br>
    돌고래 영어로? <input type="text" id="a2"> <br><br>
    오리 영어로? <input type="text" id="a3"> <br><br>
    <button onclick="checkAnswers()">제출하기</button> <br><br>
    <h3 id="result"></h3>
</body>
</html>
```

#### 주요 설명:
1. **DOCTYPE 선언**: 웹 페이지가 HTML5로 작성되었음을 명시.
2. **meta charset="UTF-8"**: HTML 문서에서 사용할 문자 인코딩을 UTF-8로 지정. (한글 포함 다양한 문자를 표현 가능)
3. **h1**: "영어 퀴즈"라는 큰 제목을 표시.
4. **input**: 텍스트 입력 필드. `id`를 통해 각 입력 필드의 고유 식별자를 지정 (JavaScript에서 해당 요소를 선택할 때 사용).
5. **button**: 버튼을 클릭하면 JavaScript 함수 `checkAnswers()`가 실행됨.
6. **h3**: 점수를 출력할 위치를 나타내며, `id="result"`로 지정하여 JavaScript로 텍스트를 삽입 가능.

### JavaScript 부분

```javascript
<script>
    function checkAnswers() {
        let score = 0;

        // 사용자가 입력한 값 가져오기
        let a1 = document.getElementById('a1').value.trim();
        let a2 = document.getElementById('a2').value.trim();
        let a3 = document.getElementById('a3').value.trim();

        // 채점
        if (a1.toLowerCase() === "cat") score++;
        if (a2.toLowerCase() === "dolphin") score++;
        if (a3.toLowerCase() === "duck") score++;

        // 결과 출력
        document.getElementById('result').textContent = score + "개 맞았습니다.";
    }
</script>
```

#### 주요 설명:
1. **function checkAnswers()**: 함수 선언. 사용자가 퀴즈를 제출할 때 실행되는 함수.
2. **let score = 0;**: 퀴즈 점수를 저장할 변수를 선언하고 초기값으로 0을 지정.
3. **document.getElementById('a1').value.trim()**:
   - **document.getElementById()**: HTML 문서에서 특정 `id`를 가진 요소를 선택하는 메서드.
   - **value**: 사용자가 입력한 텍스트 값을 가져오는 속성.
   - **trim()**: 사용자가 입력한 값에서 앞뒤 공백을 제거하는 메서드.
4. **if 문**: 사용자가 입력한 값이 정답과 일치하는지 확인. `toLowerCase()`는 대소문자 구분 없이 비교할 수 있도록 입력을 소문자로 변환.
5. **document.getElementById('result').textContent**: `result`라는 id를 가진 요소의 내용을 변경하여 점수를 표시.

---

### 필기 예시

1. **HTML과 JavaScript 연동**: HTML에서 요소를 클릭할 때 자바스크립트 함수가 실행되도록 할 수 있음.
   - `onclick="checkAnswers()"`: 버튼을 클릭하면 `checkAnswers()` 함수 실행.
  
2. **DOM 접근**: `document.getElementById()`를 사용하여 HTML의 특정 요소에 접근하고, 해당 요소의 속성을 가져오거나 변경할 수 있음.

3. **조건문 사용**: `if` 문을 사용하여 조건에 따라 특정 코드를 실행할 수 있음.
   - `if (a1.toLowerCase() === "cat") score++;`: 사용자의 입력이 "cat"인지 확인한 후 점수를 증가시킴.

4. **결과 출력**: HTML 요소의 내용을 변경하는 방법으로 `textContent`를 사용함.
   - `document.getElementById('result').textContent = score + "개 맞았습니다."`: 점수를 화면에 출력.

---
### `.value` vs `.textContent` 비교

#### 1. `.value`
- **용도**: 주로 **`input`**, **`textarea`** 같은 **입력 요소의 값**을 가져오거나 설정할 때 사용.
- **적용 대상**: `<input>`, `<textarea>`, `<select>`와 같은 사용자 입력을 받는 요소들.
- **읽기/쓰기 가능**: 사용자가 입력한 값을 읽거나 수정할 수 있음.

```javascript
let userInput = document.getElementById('a1').value;
```

위 코드에서 `.value`는 사용자가 `<input>` 필드에 입력한 값을 가져옵니다.

#### 예시:
```html
<input type="text" id="a1" value="hello">
```
```javascript
let inputValue = document.getElementById('a1').value;  // "hello" 반환
```

#### 2. `.textContent`
- **용도**: 요소의 **텍스트 내용을 가져오거나 설정**할 때 사용.
- **적용 대상**: 주로 **일반적인 HTML 요소들**(`<div>`, `<span>`, `<p>`, `<h1>`, `<h3>` 등)에 적용. 해당 요소 내부의 텍스트 내용을 가져오거나 설정함.
- **읽기/쓰기 가능**: 요소의 텍스트 내용을 읽고 수정할 수 있음.

```javascript
document.getElementById('result').textContent = "점수: 3개";
```

위 코드에서 `.textContent`는 `<h3>` 요소의 텍스트 내용을 변경하여 화면에 표시합니다.

#### 예시:
```html
<h3 id="result">초기 값</h3>
```
```javascript
document.getElementById('result').textContent = "새로운 값";  // "새로운 값"으로 변경됨
```

---
