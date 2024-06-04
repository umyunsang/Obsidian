
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