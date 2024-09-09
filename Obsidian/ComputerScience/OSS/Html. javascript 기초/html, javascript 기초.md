

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
### 1. **`let` 키워드**

- **변경 가능성**: `let`으로 선언한 변수는 재할당이 가능합니다. 즉, 변수를 초기화한 후에 다른 값으로 변경할 수 있습니다.
### 2. **`const` 키워드**

- **변경 불가능성**: `const`로 선언된 변수는 재할당이 불가능합니다. 한 번 값을 할당하면, 이후에 값을 변경할 수 없습니다.

---
