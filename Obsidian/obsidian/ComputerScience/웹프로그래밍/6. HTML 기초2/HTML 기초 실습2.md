
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
- **표 생성**
  - HTML에서 표를 생성하고, 여러 셀을 합치는 방법을 배웁니다.
  - 예시:
    ```html
    <table border="1">
        <tr> <td colspan="3">여러 칸 셀 합치기
        <tr> <td>2행 1열
             <td>2행 2열
             <td>2행 3열
    </table>
    ```
- **rowspan**
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
  - `rowspan="2"` 속성을 사용하여 두 행을 합칩니다.

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

---

```html
<form method="post" action="/mybook">
    성명: <input type="text" name="person">
    <p> 회원여부: 
        <input type="radio" name="member" value="yes" checked>회원 
        <input type="radio" name="member" value="no">비회원
    <p> 직업: 
        <select name="job" size="1">
            <option>학생 
            <option selected>창업CEO 
            <option>예술가 
            <option>기타 
        </select>
    <p> 
        <fieldset>
            <legend> 구입희망분야(복수선택 가능)</legend> 
            <input type="checkbox" name="books" value="computer">컴퓨터 
            <input type="checkbox" name="books" value="economy">주식 
            <input type="checkbox" name="books" value="animation">애니메이션 
            <input type="checkbox" name="books" value="common">상식 
        </fieldset>
    <p> 비고: <br> 
        <textarea name="comments" rows="4" cols="40">...하고픈 말...</textarea>
    </p> 
    <button>신청</button> 
</form>

```