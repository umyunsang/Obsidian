

---
# QUIZ.1

```html
<body>
    <h1>영어 퀴즈</h1>
    고양이 영어로? <input type="text" id="a1"> <br><br>
    돌고래 영어로? <input type="text" id="a2"> <br><br>
    오리 영어로? <input type="text" id="a3"> <br><br>
    <button onclick="checkAnswers()">제출하기</button> <br><br>
    <h3 id="result"></h3>
</body>
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
</html>
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

#### 2. `.textContent`
- **용도**: 요소의 **텍스트 내용을 가져오거나 설정**할 때 사용.
- **적용 대상**: 주로 **일반적인 HTML 요소들**(`<div>`, `<span>`, `<p>`, `<h1>`, `<h3>` 등)에 적용. 해당 요소 내부의 텍스트 내용을 가져오거나 설정함.
- **읽기/쓰기 가능**: 요소의 텍스트 내용을 읽고 수정할 수 있음.

```javascript
document.getElementById('result').textContent = "점수: 3개";
```

위 코드에서 `.textContent`는 `<h3>` 요소의 텍스트 내용을 변경하여 화면에 표시합니다.

---
### 1. **`let` 키워드**

- **변경 가능성**: `let`으로 선언한 변수는 재할당이 가능합니다. 즉, 변수를 초기화한 후에 다른 값으로 변경할 수 있습니다.
### 2. **`const` 키워드**

- **변경 불가능성**: `const`로 선언된 변수는 재할당이 불가능합니다. 한 번 값을 할당하면, 이후에 값을 변경할 수 없습니다.

---
# who.html

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>예제 2</title>
    <style>
        body {text-align: center}
        img {max-width: 300px; max-height: 300px;}
    </style>
</head>
<body>
    <h1>사진 가져오기</h1>
    <input type="text" id="who">
    <button onclick="showImage()">사진</button>
    <br><br>
    <img id="image" src="static/나는.png" alt="사진을 가져오지 못했습니다.">
    <script>
        function showImage() {
            const image = document.getElementById('image');
            const who = document.getElementById('who').value.trim();

            if (who == "당근" || who == "당") 
            image.src = 'static/당근.png';
            else if (who == "양파" || who == "양") 
            image.src = 'static/양파.png';
            else if (who == "식빵" || who == "식") 
            image.src = 'static/식빵.png';
            else image.src = 'static/쏘리.png';
        }
    </script>
</body>
</html>
```
- **`<input type="text" id="who">`**: 사용자가 텍스트를 입력할 수 있는 입력 필드입니다. 여기서 사용자가 입력한 값에 따라 특정 이미지를 변경하게 됩니다.
- **`<button onclick="showImage()">사진</button>`**: "사진" 버튼을 클릭하면 `showImage()` 함수가 실행됩니다.
- **`<img id="image" src="static/나는.png" alt="사진을 가져오지 못했습니다.">`**: 초기 이미지를 표시하는 요소입니다. `id="image"`는 자바스크립트에서 이 요소에 접근할 수 있도록 해줍니다. `src` 속성은 이미지 파일의 경로를 지정하며, `alt` 속성은 이미지를 불러올 수 없을 때 대신 표시될 텍스트를 정의합니다.

```js
function showImage() {     const image = document.getElementById('image');      const who = document.getElementById('who').value.trim();
```

- **`document.getElementById('image')`**: `id`가 "image"인 `<img>` 요소에 접근하여 변수 `image`에 저장합니다.
- **`document.getElementById('who').value.trim()`**: 사용자가 입력한 텍스트 값을 가져와 `who` 변수에 저장합니다. `trim()` 메서드는 사용자가 입력한 값의 앞뒤 공백을 제거합니다.

```js
if (who == "당근" || who == "당")          
image.src = 'static/당근.png';      
else if (who == "양파" || who == "양")          
image.src = 'static/양파.png';      
else if (who == "식빵" || who == "식")          
image.src = 'static/식빵.png';     
else          
image.src = 'static/쏘리.png'; }
```

- **조건문**: 사용자가 입력한 값(`who`)에 따라 `image` 요소의 `src` 속성을 변경합니다.
    - `who`가 "당근" 또는 "당"일 경우 `image.src = 'static/당근.png'`로 설정하여 "당근" 이미지로 변경합니다.
    - `who`가 "양파" 또는 "양"일 경우 `image.src = 'static/양파.png'`로 설정합니다.
    - `who`가 "식빵" 또는 "식"일 경우 `image.src = 'static/식빵.png'`로 설정합니다.
    - 위의 조건에 맞지 않으면 `image.src = 'static/쏘리.png'`로 설정하여 기본적으로 "쏘리" 이미지를 표시합니다.


---
# ex3.js

```js
function calculate(op) {
    // html에서 입력한 숫자 2개 가져와서 숫자로 변환 후 num1, num2에 넣음
    const num1 = parseFloat(document.getElementById('number1').value);
    const num2 = parseFloat(document.getElementById('number2').value);
    let result = 0;

    // 숫자 아니면 리턴(종료)
    if (isNaN(num1) || isNaN(num2)) {
        alert("두 개 모두 숫자로 입력해 주세요!");
        return;
    }

    switch (op) {
        case '+': result = num1 + num2; break;
        case '-': result = num1 - num2; break;
        case '*': result = num1 * num2; break;
        case '/': if (num2 === 0) {
                alert("0으로 나눌 수 없어요!"); return;
            }
            result = num1 / num2; break;
    }

    // 결과 보여줌
    document.getElementById('result').value = result;
}

```
### 1. **`parseFloat()`**
`parseFloat()` 함수는 문자열을 실수(float)로 변환하는 역할을 합니다. 문자열 안에 숫자 형식이 있을 경우, 그 숫자를 반환하며, 문자열에 포함된 숫자 외의 문자는 무시됩니다.

#### 예시:
```javascript
parseFloat("123.45");    // 123.45
parseFloat("123.45abc"); // 123.45 (문자 'abc'는 무시)
parseFloat("abc123.45"); // NaN (숫자가 앞에 없으므로 변환 불가)
```

- **역할:** 문자열을 실수로 변환. 숫자가 포함된 문자열만을 처리하며, 숫자로 변환할 수 없는 경우 `NaN`을 반환합니다.

#### 주의점:
- `parseFloat()`은 정수도 변환할 수 있지만, 항상 실수로 처리합니다.
- 숫자 이외의 문자가 뒤에 붙어있으면 그 문자는 무시되지만, 앞에 있으면 변환할 수 없습니다.

### 2. **`isNaN()`**
`isNaN()` 함수는 주어진 값이 `NaN`(Not-a-Number)인지 확인하는 역할을 합니다. 이 함수는 숫자가 아닌 값이나, 숫자로 변환할 수 없는 값을 만나면 `true`를 반환합니다.

#### 예시:
```javascript
isNaN(123);          // false (숫자이므로)
isNaN("123");        // false (숫자로 변환 가능하므로)
isNaN("abc");        // true (숫자가 아니므로)
isNaN(NaN);          // true (NaN은 자신도 NaN이므로)
isNaN(parseFloat("abc")); // true (변환 실패로 NaN 반환)
```

- **역할:** 입력값이 숫자가 아닌지를 검사하여, `NaN`일 경우 `true`를 반환합니다.

#### 주의점:
- `isNaN()`은 숫자가 아닌 값인지 확인할 때만 사용해야 합니다. 예를 들어, `isNaN(null)`은 `false`를 반환합니다. 이는 자바스크립트의 암묵적 타입 변환으로 `null`이 `0`으로 변환되기 때문입니다.

---
# ex4.js
```js
let n, sum;

n = prompt("숫자를 입력하세요.");
document.write("n ==> " + n + "<hr>");

sum = n + 100;
document.write("n + 100 ==> " + sum + "<hr>");

sum = parseInt(n) + 100;
document.write("정수변환n + 100 ==> " + sum + "<hr>");
```
- `document.write()`는 HTML 문서에 실시간으로 내용을 출력합니다.
- `prompt()`는 사용자의 입력을 받고, 그 결과는 문자열입니다.
- 문자열과 숫자를 더하면 문자열 연결이 되고, `parseInt()`를 사용하면 문자열을 정수로 변환할 수 있습니다.

# ex5.js
```js
function changeColor() {
    const co = document.getElementById("co").value;
    // 입력된 색상을 body의 글자색으로 설정
    document.body.style.color = co;
    // 입력된 색상이 잘못되었을 경우 팝업창
    if (!isValidColor(co)) {
      alert(co + "는 존재하지 않는 색입니다. 다시 작성해 주세요 ^^");
    }
  }
  
  function isValidColor(color) {
    // 유효한 색상인지 확인
    const st = new Option().style;
    st.color = color;
    if (st.color == "") return false;
    else return true;
  }
```
- 