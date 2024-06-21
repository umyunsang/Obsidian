
---
### HTML 기초 실습 정리

#### 노란색 형광펜 효과
- **CSS를 사용한 배경색 설정**
  - 특정 텍스트를 강조하고 싶을 때 노란색 배경을 설정할 수 있습니다.
  - 예시:
    ```html
    <span style="background-color: yellow;">강조할 텍스트</span>
    ```
  - CSS 클래스 사용:
    ```html
    <style>
        .highlight {
            background-color: yellow;
        }
    </style>
    <span class="highlight">강조할 텍스트</span>
    ```

#### 위첨자와 아래첨자
- **위첨자 (Superscript)**
  - 숫자나 문자의 위첨자 형태를 표현할 때 사용합니다.
  - 예시: `x<sup>2</sup> + y<sup>2</sup> = 1` ➔ x² + y² = 1
- **아래첨자 (Subscript)**
  - 숫자나 문자의 아래첨자 형태를 표현할 때 사용합니다.
  - 예시: `H<sub>2</sub>O` ➔ H₂O

#### HTML 표 실습
- colspan
	-  열을 합치는 방법입니다.
	- 예시:
```html
    <table border="1">
        <tr> <td colspan="3">여러 칸 셀 합치기
        <tr> <td>2행 1열
             <td>2행 2열
             <td>2행 3열
    </table>
```
![[Pasted image 20240610170328.png]]
-  **rowspan**
	- 행을 합치는 방법입니다.
	- 예시:
```html
    <table border="1">
        <tr> <td rowspan="2">1열 합치기
             <td>1행 2열
             <td>1행 3열
        <tr> <td>2행 2열
             <td>2행 3열
    </table>
```
![[Pasted image 20240610170137.png]]

### HTML `input` 태그의 `type` 속성: `range`와 `number`

#### `input type="range"`
- **설명**: 
  - `type="range"`는 슬라이더 컨트롤을 생성합니다. 사용자가 지정된 범위 내에서 값을 선택할 수 있습니다.
  - 슬라이더는 시각적으로 직관적이며, 특히 값의 범위를 빠르게 선택해야 하는 경우에 유용합니다.
- **속성**:
  - `min`: 슬라이더의 최소값을 설정합니다.
  - `max`: 슬라이더의 최대값을 설정합니다.
  - `step`: 슬라이더가 이동할 때의 간격을 설정합니다.
  - `value`: 슬라이더의 초기값을 설정합니다.
- **예시**:
  ```html
  <label for="volume">Volume:</label>
  <input type="range" id="volume" name="volume" min="0" max="100" step="1" value="50">
  ```
  - 이 예제는 0에서 100까지의 범위에서 값을 선택할 수 있는 슬라이더를 생성하며, 초기값은 50입니다.

#### `input type="number"`
- **설명**:
  - `type="number"`는 숫자를 입력할 수 있는 텍스트 필드를 생성합니다. 사용자는 직접 숫자를 입력하거나 스핀 버튼(위아래 화살표)을 사용하여 값을 증가시키거나 감소시킬 수 있습니다.
  - 폼 제출 시 숫자 형식을 자동으로 검증하므로 숫자가 아닌 값이 입력되지 않습니다.
- **속성**:
  - `min`: 입력할 수 있는 최소값을 설정합니다.
  - `max`: 입력할 수 있는 최대값을 설정합니다.
  - `step`: 증가 또는 감소할 때의 간격을 설정합니다.
  - `value`: 필드의 초기값을 설정합니다.
- **예시**:
  ```html
  <label for="quantity">Quantity:</label>
  <input type="number" id="quantity" name="quantity" min="1" max="10" step="1" value="5">
  ```
  - 이 예제는 1에서 10까지의 숫자를 입력할 수 있는 필드를 생성하며, 초기값은 5입니다.

### HTML 속성: `required`, `checked`, `selected`, `disabled`

#### `required`
- **설명**:
  - `required` 속성은 폼의 입력 필드가 필수 항목임을 지정합니다.
  - 사용자가 해당 입력 필드를 채우지 않으면 폼이 제출되지 않으며, 브라우저는 사용자에게 경고 메시지를 표시합니다.
- **사용 가능한 요소**:
  - `<input>` (특히 `type="text"`, `type="email"`, `type="password"`, `type="number"`, 등)
  - `<textarea>`
  - `<select>`

#### `checked`
- **설명**:
  - `checked` 속성은 체크박스(`input type="checkbox"`)나 라디오 버튼(`input type="radio"`)이 기본적으로 선택되어 있음을 지정합니다.
- **사용 가능한 요소**:
  - `<input type="checkbox">`
  - `<input type="radio">`

#### `selected`
- **설명**:
  - `selected` 속성은 드롭다운 목록(`<select>`)의 옵션(`<option>`)이 기본적으로 선택되어 있음을 지정합니다.
- **사용 가능한 요소**:
  - `<option>`

#### `disabled`
- **설명**:
  - `disabled` 속성은 폼 요소가 비활성화되어 사용자가 상호작용할 수 없음을 지정합니다.
  - 비활성화된 폼 요소는 폼 제출 시 전송되지 않습니다.
- **사용 가능한 요소**:
  - `<input>`
  - `<textarea>`
  - `<select>`
  - `<button>`
  - `<fieldset>`
  - `<option>`
  - `<optgroup>`

| 속성         | 설명                                         | 사용 가능한 요소                                                                                                  |
| ---------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `required` | 입력 필드가 필수 항목임을 지정. 사용자가 입력하지 않으면 폼 제출이 안됨. | `<input>` (`type="text"`, `type="email"`, `type="password"`, `type="number"`, 등), `<textarea>`, `<select>` |
| `checked`  | 체크박스 또는 라디오 버튼이 기본적으로 선택되어 있음을 지정.         | `<input type="checkbox">`, `<input type="radio">`                                                          |
| `selected` | 드롭다운 목록의 옵션이 기본적으로 선택되어 있음을 지정.            | `<option>`                                                                                                 |
| `disabled`  | 폼 요소가 비활성화되어 사용자가 상호작용할 수 없음을 지정. 폼 제출 시 전송되지 않음.                   | `<input>`, `<textarea>`, `<select>`, `<button>`, `<fieldset>`, `<option>`, `<optgroup>`                     |

---
![[Pasted image 20240610170610.png]]

```html
<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Insert title here</title></head>
<body>
<strong>설문조사</strong><p>
<form action="/iam/answer" method="post">
<fieldset>
	<legend>성별과 나이를 입력하세요.</legend>
	성별: <input type="radio" name="gender" value="남" checked>남자
	<input type="radio" name="gender" value="여">여자<br>
	나이 : <input type="number" name="age" required>
</fieldset><p>

<fieldset>
	<legend>관심있는 식문화를 선택하세요.(1개 이상)</legend>
	<input type="checkbox" name="foods" value="소주에 삼겹살">소주에 삼겹살
	<input type="checkbox" name="foods" value="딸기케이크">딸기 케이크
	<input type="checkbox" name="foods" value="치맥">치맥
	<input type="checkbox" name="foods" value="킹크랩">킹크랩
</fieldset><p>

<fieldset>
	<legend>나의 자신감은 어느 정도인가?</legend>
	나는 무지 잘 생겼다. 
	<input type="range" name="face" min="1" max="5" value="1"><br>
	나는 맘만 먹으면 A+ 이다. 
	<input type="range" name="grade" min="1" max="5" value="1"> <br>
	강의실에서 산만해보여도 머리속에는 내용이 잘 정리되어 있다.<br>
	<input type="range" name="head" min="1" max="5" value="1">
</fieldset><p>

Promote yourself<br>
<textarea name="promote" rows="4" cols="40"></textarea><p>
<input type="submit" value="확인">
<input type="reset" value="취소">
</form></body></html>
```