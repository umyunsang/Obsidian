
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